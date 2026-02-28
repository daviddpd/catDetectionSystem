from __future__ import annotations

import json
import platform
import shutil
from pathlib import Path
from typing import Any

from cds.training.constants import LEGACY_RKNN_CHIPS, TOOLKIT2_CHIPS

SUPPORTED_TARGETS = {"pytorch", "onnx", "coreml", "tensorrt", "rknn"}


def available_export_targets() -> dict[str, dict[str, Any]]:
    system = platform.system().lower()

    def _module(name: str) -> bool:
        try:
            __import__(name)
            return True
        except Exception:
            return False

    has_ultralytics = _module("ultralytics")
    has_onnx = _module("onnx")
    has_coremltools = _module("coremltools")
    has_tensorrt = _module("tensorrt")
    has_rknn_toolkit = _module("setuptools") and (_module("rknn") or _module("rknn_toolkit2"))
    has_rknn_runtime = _module("rknnlite")

    return {
        "pytorch": {"supported": True, "reason": "checkpoint copy"},
        "onnx": {
            "supported": has_ultralytics and has_onnx,
            "reason": "ultralytics+onnx required",
        },
        "coreml": {
            "supported": has_ultralytics and has_coremltools and system == "darwin",
            "reason": "macOS with coremltools",
        },
        "tensorrt": {
            "supported": has_ultralytics and has_tensorrt,
            "reason": "ultralytics+tensorrt runtime",
        },
        "rknn": {
            "supported": has_onnx,
            "reason": "ONNX export required; conversion can run offline on RKNN host",
            "toolkit_present": has_rknn_toolkit,
            "runtime_present": has_rknn_runtime,
        },
    }


def _normalize_targets(raw_targets: str) -> list[str]:
    parts = [item.strip().lower() for item in raw_targets.split(",") if item.strip()]
    if not parts:
        return ["all"]
    return parts


def _write_rknn_conversion_bundle(onnx_path: Path, output_root: Path) -> dict[str, str]:
    rknn_dir = output_root / "rknn"
    rknn_dir.mkdir(parents=True, exist_ok=True)

    toolkit2_script = rknn_dir / "convert_toolkit2.py"
    legacy_script = rknn_dir / "convert_legacy.py"
    calibration_file = rknn_dir / "calibration.txt"
    calibration_helper = rknn_dir / "make_calibration_txt.py"

    toolkit2_output = rknn_dir / "model.toolkit2.rknn"
    legacy_output = rknn_dir / "model.legacy.rknn"

    toolkit2_script.write_text(
        f"""#!/usr/bin/env python3
from __future__ import annotations

# Toolkit2 conversion template. Run on an RKNN Toolkit2-capable host.
# Update target_platform and dataset calibration file before execution.
# The default mean/std settings below match CDS runtime preprocessing
# (RGB float32 normalized to 0..1 before inference).

try:
    from rknn.api import RKNN
except ModuleNotFoundError as exc:
    if exc.name == "pkg_resources":
        try:
            import setuptools
            version = getattr(setuptools, "__version__", "installed")
        except Exception:
            version = None
        version_note = f"setuptools {{version}} is installed, " if version else ""
        raise SystemExit(
            version_note
            + "but pkg_resources is unavailable. "
            + "RKNN Toolkit2 currently requires pkg_resources. "
            + "Pin setuptools below 82 in this environment: "
            + "python3 -m pip install 'setuptools<82'"
        ) from exc
    raise

from pathlib import Path

ONNX_PATH = r\"{onnx_path}\"
OUTPUT_PATH = r\"{toolkit2_output}\"
TARGET_PLATFORM = "RK3588"
MEAN_VALUES = [[0, 0, 0]]
STD_VALUES = [[255, 255, 255]]
DO_QUANTIZATION = True
CALIBRATION_DATASET = Path(__file__).with_name("calibration.txt")


def _resolve_calibration_dataset() -> str | None:
    if not DO_QUANTIZATION:
        return None
    if not CALIBRATION_DATASET.exists():
        raise SystemExit(
            "Calibration dataset file not found: "
            + str(CALIBRATION_DATASET)
            + ". Generate it with: "
            + f"python3 {{Path(__file__).with_name('make_calibration_txt.py')}} /path/to/images"
        )
    lines = [
        line.strip()
        for line in CALIBRATION_DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise SystemExit(
            "Calibration dataset file is empty: "
            + str(CALIBRATION_DATASET)
            + ". Add one absolute image path per line, or set DO_QUANTIZATION = False "
            + "for a non-quantized conversion."
        )
    return str(CALIBRATION_DATASET)

rknn = RKNN(verbose=True)
rknn.config(
    target_platform=TARGET_PLATFORM,
    mean_values=MEAN_VALUES,
    std_values=STD_VALUES,
)
assert rknn.load_onnx(model=ONNX_PATH) == 0
assert rknn.build(
    do_quantization=DO_QUANTIZATION,
    dataset=_resolve_calibration_dataset(),
) == 0
assert rknn.export_rknn(OUTPUT_PATH) == 0
rknn.release()
print("Exported", OUTPUT_PATH)
""",
        encoding="utf-8",
    )
    legacy_script.write_text(
        f"""#!/usr/bin/env python3
from __future__ import annotations

# Legacy RKNN toolkit conversion template. Run on legacy-compatible host.
# The default mean/std settings below match CDS runtime preprocessing
# (RGB float32 normalized to 0..1 before inference).

try:
    from rknn.api import RKNN
except ModuleNotFoundError as exc:
    if exc.name == "pkg_resources":
        try:
            import setuptools
            version = getattr(setuptools, "__version__", "installed")
        except Exception:
            version = None
        version_note = f"setuptools {{version}} is installed, " if version else ""
        raise SystemExit(
            version_note
            + "but pkg_resources is unavailable. "
            + "The RKNN Python conversion tool currently requires pkg_resources. "
            + "Pin setuptools below 82 in this environment: "
            + "python3 -m pip install 'setuptools<82'"
        ) from exc
    raise

from pathlib import Path

ONNX_PATH = r\"{onnx_path}\"
OUTPUT_PATH = r\"{legacy_output}\"
TARGET_PLATFORM = "RK1808"
MEAN_VALUES = [[0, 0, 0]]
STD_VALUES = [[255, 255, 255]]
DO_QUANTIZATION = True
CALIBRATION_DATASET = Path(__file__).with_name("calibration.txt")


def _resolve_calibration_dataset() -> str | None:
    if not DO_QUANTIZATION:
        return None
    if not CALIBRATION_DATASET.exists():
        raise SystemExit(
            "Calibration dataset file not found: "
            + str(CALIBRATION_DATASET)
            + ". Generate it with: "
            + f"python3 {{Path(__file__).with_name('make_calibration_txt.py')}} /path/to/images"
        )
    lines = [
        line.strip()
        for line in CALIBRATION_DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise SystemExit(
            "Calibration dataset file is empty: "
            + str(CALIBRATION_DATASET)
            + ". Add one absolute image path per line, or set DO_QUANTIZATION = False "
            + "for a non-quantized conversion."
        )
    return str(CALIBRATION_DATASET)

rknn = RKNN(verbose=True)
rknn.config(
    target_platform=TARGET_PLATFORM,
    mean_values=MEAN_VALUES,
    std_values=STD_VALUES,
)
assert rknn.load_onnx(model=ONNX_PATH) == 0
assert rknn.build(
    do_quantization=DO_QUANTIZATION,
    dataset=_resolve_calibration_dataset(),
) == 0
assert rknn.export_rknn(OUTPUT_PATH) == 0
rknn.release()
print("Exported", OUTPUT_PATH)
""",
        encoding="utf-8",
    )
    calibration_helper.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _find_repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "src" / "cds").is_dir():
            return candidate
    return None


def _default_model_path() -> str | None:
    exports_dir = Path(__file__).resolve().parents[1] / "exports"
    if not exports_dir.exists():
        return None
    for candidate in (
        exports_dir / "best.mlpackage",
        exports_dir / "best.mlmodel",
        exports_dir / "best.pt",
        exports_dir / "best.onnx",
        exports_dir / "model.mlpackage",
        exports_dir / "model.mlmodel",
        exports_dir / "model.pt",
        exports_dir / "model.onnx",
    ):
        if candidate.exists():
            return str(candidate)
    return None


def _iter_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _load_backend(
    *,
    model_path: str,
    labels_path: str | None,
    backend_name: str,
    imgsz: int,
    min_confidence: float,
    nms: float,
):
    repo_root = _find_repo_root()
    if repo_root is None:
        raise SystemExit(
            "Could not locate repository root containing src/cds. "
            "Run this helper from the catDetectionSystem checkout."
        )
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from cds.config.models import BackendPolicyConfig
    from cds.detector.models.model_spec import ModelSpec
    from cds.detector.selector import select_backend

    model_spec = ModelSpec(
        name="rknn-calibration",
        model_path=str(Path(model_path).expanduser().resolve()),
        labels_path=(
            str(Path(labels_path).expanduser().resolve())
            if labels_path
            else None
        ),
        confidence=float(min_confidence),
        nms=float(nms),
        imgsz=max(32, int(imgsz)),
    )
    selection = select_backend(
        model_spec,
        BackendPolicyConfig(
            requested=str(backend_name or "auto").strip().lower() or "auto",
            allow_darknet_fallback=True,
            allow_rknn=True,
            allow_tensorrt=True,
        ),
    )
    print(
        "Using backend="
        + selection.backend.name()
        + " device="
        + selection.backend.device_info()
        + " reason="
        + selection.reason
    )
    selection.backend.warmup()
    return selection.backend, model_spec


def _score_images(
    images: list[Path],
    detector: Any,
) -> tuple[list[tuple[Path, float, tuple[str, ...], int]], int]:
    import cv2

    ranked: list[tuple[Path, float, tuple[str, ...], int]] = []
    unreadable = 0
    for index, image_path in enumerate(images, start=1):
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            unreadable += 1
            continue
        detections = detector.infer(frame)
        if not detections:
            continue
        best_confidence = max(float(det.confidence) for det in detections)
        labels = tuple(sorted({str(det.label) for det in detections}))
        ranked.append((image_path, best_confidence, labels, index))
    return ranked, unreadable


def _select_ranked(
    ranked: list[tuple[Path, float, tuple[str, ...], int]],
    *,
    limit: int,
    coverage_per_label: int,
) -> list[Path]:
    ordered = sorted(
        ranked,
        key=lambda item: (-item[1], -len(item[2]), item[3], str(item[0])),
    )
    if limit <= 0:
        return [item[0] for item in ordered]

    selected: list[Path] = []
    used: set[Path] = set()
    label_counts: dict[str, int] = {}
    if coverage_per_label > 0:
        for path, _score, labels, _index in ordered:
            if path in used:
                continue
            needed = [
                label
                for label in labels
                if label_counts.get(label, 0) < coverage_per_label
            ]
            if not needed:
                continue
            selected.append(path)
            used.add(path)
            for label in labels:
                if label in needed:
                    label_counts[label] = label_counts.get(label, 0) + 1
            if len(selected) >= limit:
                return selected

    for path, _score, _labels, _index in ordered:
        if path in used:
            continue
        selected.append(path)
        used.add(path)
        if len(selected) >= limit:
            break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an RKNN calibration.txt file from a directory of images."
    )
    parser.add_argument("source", help="Directory scanned recursively for images")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("calibration.txt")),
        help="Output text file path (default: alongside this script)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of image paths to write (0 = all, default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Shuffle seed before truncation (default: 1337)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Optional model used to score images and keep only high-confidence positives "
            "(default: disabled)"
        ),
    )
    parser.add_argument(
        "--use-bundle-model",
        action="store_true",
        help=(
            "Auto-discover best.mlpackage/best.pt/best.onnx beside this bundle and use it "
            "for model-assisted selection"
        ),
    )
    parser.add_argument(
        "--labels-path",
        default=None,
        help="Optional labels file passed to the detector backend",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        help="Detector backend policy for model-assisted mode (default: auto)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size for model-assisted mode (default: 640)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.90,
        help="Minimum detection confidence to include an image in model-assisted mode (default: 0.90)",
    )
    parser.add_argument(
        "--nms",
        type=float,
        default=0.50,
        help="NMS IoU threshold for model-assisted mode (default: 0.50)",
    )
    parser.add_argument(
        "--coverage-per-label",
        type=int,
        default=1,
        help=(
            "When model-assisted mode is active, reserve up to this many top images per detected label "
            "before filling remaining slots (0 disables, default: 1)"
        ),
    )
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.exists() or not source.is_dir():
        raise SystemExit(f"Source directory not found: {source}")

    images = _iter_images(source)
    if not images:
        raise SystemExit(
            "No calibration images found under "
            + str(source)
            + ". Supported extensions: "
            + ", ".join(sorted(IMAGE_EXTENSIONS))
        )

    selected: list[Path]
    model_path = args.model_path
    if not model_path and args.use_bundle_model:
        model_path = _default_model_path()
        if not model_path:
            raise SystemExit(
                "No bundle-local model found under ./exports. "
                "Pass --model-path explicitly or omit --use-bundle-model."
            )

    if model_path:
        detector, _model_spec = _load_backend(
            model_path=model_path,
            labels_path=args.labels_path,
            backend_name=args.backend,
            imgsz=args.imgsz,
            min_confidence=args.min_confidence,
            nms=args.nms,
        )
        ranked, unreadable = _score_images(images, detector)
        if not ranked:
            raise SystemExit(
                "No qualifying calibration images found after model scoring. "
                "Lower --min-confidence, verify --model-path, or use --model-path '' "
                "to fall back to random sampling."
            )
        selected = _select_ranked(
            ranked,
            limit=args.limit,
            coverage_per_label=max(0, int(args.coverage_per_label)),
        )
        selected_meta = {
            path: (score, labels)
            for path, score, labels, _index in ranked
        }
        selected_labels = Counter(
            label
            for path in selected
            for label in selected_meta.get(path, (0.0, tuple()))[1]
        )
        print(
            "Scored "
            + str(len(images))
            + " images, unreadable="
            + str(unreadable)
            + ", qualifying="
            + str(len(ranked))
            + ", selected="
            + str(len(selected))
            + ", labels="
            + str(dict(sorted(selected_labels.items())))
        )
    else:
        rng = random.Random(args.seed)
        rng.shuffle(images)
        selected = images if args.limit <= 0 else images[: args.limit]

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(f"{path.resolve()}\\n" for path in selected),
        encoding="utf-8",
    )
    print(f"Wrote {len(selected)} calibration entries to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )
    calibration_file.write_text(
        "# One absolute image path per line for RKNN quantization calibration.\n"
        "# Generate with:\n"
        "#   python3 ./make_calibration_txt.py /path/to/images --limit 200\n",
        encoding="utf-8",
    )

    toolkit2_script.chmod(0o755)
    legacy_script.chmod(0o755)
    calibration_helper.chmod(0o755)

    chip_notes = rknn_dir / "chip_families.txt"
    chip_notes.write_text(
        "Toolkit2 family:\n- "
        + "\n- ".join(TOOLKIT2_CHIPS)
        + "\n\nLegacy family:\n- "
        + "\n- ".join(LEGACY_RKNN_CHIPS)
        + "\n",
        encoding="utf-8",
    )

    return {
        "toolkit2_script": str(toolkit2_script),
        "legacy_script": str(legacy_script),
        "calibration_file": str(calibration_file),
        "calibration_helper": str(calibration_helper),
        "chip_notes": str(chip_notes),
    }


def _stage_export_artifact(artifact: Path, exports_dir: Path) -> Path:
    if not artifact.exists():
        return artifact

    dest = exports_dir / artifact.name
    if dest == artifact:
        return artifact

    if artifact.is_dir():
        if dest.exists() and not dest.is_dir():
            dest.unlink()
        shutil.copytree(artifact, dest, dirs_exist_ok=True)
        return dest

    if dest.exists() and dest.is_dir():
        shutil.rmtree(dest)
    shutil.copy2(artifact, dest)
    return dest


def export_model_artifacts(
    model_path: Path,
    output_root: Path,
    targets_csv: str,
    imgsz: int = 640,
    half: bool = False,
) -> dict[str, Any]:
    target_list = _normalize_targets(targets_csv)
    availability = available_export_targets()

    if "all" in target_list:
        target_list = ["pytorch", "onnx", "coreml", "tensorrt", "rknn"]

    for target in target_list:
        if target not in SUPPORTED_TARGETS:
            raise ValueError(f"Unsupported export target: {target}")

    output_root.mkdir(parents=True, exist_ok=True)
    exports_dir = output_root / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "source_model": str(model_path),
        "requested_targets": target_list,
        "results": [],
    }

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    source_is_onnx = model_path.suffix.lower() == ".onnx"
    source_is_pt = model_path.suffix.lower() in {".pt", ".pth"}

    checkpoint_dest = exports_dir / "model.pt"
    if source_is_pt:
        shutil.copy2(model_path, checkpoint_dest)
        report["results"].append(
            {
                "target": "pytorch",
                "status": "ok",
                "artifact": str(checkpoint_dest),
            }
        )
    else:
        report["results"].append(
            {
                "target": "pytorch",
                "status": "skipped",
                "message": "source model is not a PyTorch checkpoint",
            }
        )

    need_ultralytics = any(t in target_list for t in ("onnx", "coreml", "tensorrt"))
    yolo_model = None
    if need_ultralytics and source_is_pt:
        try:
            from ultralytics import YOLO

            yolo_model = YOLO(str(model_path))
        except Exception as exc:
            yolo_model = None
            report["results"].append(
                {
                    "target": "ultralytics-load",
                    "status": "error",
                    "message": str(exc),
                }
            )

    onnx_artifact: Path | None = None
    if source_is_onnx:
        onnx_artifact = exports_dir / model_path.name
        shutil.copy2(model_path, onnx_artifact)
        report["results"].append(
            {
                "target": "onnx",
                "status": "ok",
                "artifact": str(onnx_artifact),
                "message": "source model is already ONNX",
            }
        )

    for target in target_list:
        if target == "pytorch":
            continue
        if target == "onnx" and source_is_onnx:
            continue

        target_available = availability.get(target, {}).get("supported", False)
        if not target_available and target != "rknn":
            report["results"].append(
                {
                    "target": target,
                    "status": "skipped",
                    "message": availability.get(target, {}).get("reason", "unsupported on host"),
                }
            )
            continue

        if target in {"onnx", "coreml", "tensorrt"}:
            if yolo_model is None:
                report["results"].append(
                    {
                        "target": target,
                        "status": "skipped",
                        "message": "ultralytics model unavailable",
                    }
                )
                continue

            fmt = {
                "onnx": "onnx",
                "coreml": "coreml",
                "tensorrt": "engine",
            }[target]
            try:
                export_path = yolo_model.export(format=fmt, imgsz=imgsz, half=half)
                artifact = Path(str(export_path)).resolve()
                artifact = _stage_export_artifact(artifact, exports_dir)
                report["results"].append(
                    {
                        "target": target,
                        "status": "ok",
                        "artifact": str(artifact),
                    }
                )
                if target == "onnx":
                    onnx_artifact = artifact
            except Exception as exc:
                report["results"].append(
                    {
                        "target": target,
                        "status": "error",
                        "message": str(exc),
                    }
                )
            continue

        if target == "rknn":
            if onnx_artifact is None:
                onnx_candidates = sorted(exports_dir.glob("*.onnx"))
                if onnx_candidates:
                    onnx_artifact = onnx_candidates[0]

            if onnx_artifact is None:
                report["results"].append(
                    {
                        "target": "rknn",
                        "status": "skipped",
                        "message": "ONNX artifact required before RKNN conversion bundle generation",
                    }
                )
                continue

            bundle = _write_rknn_conversion_bundle(onnx_artifact, output_root)
            report["results"].append(
                {
                    "target": "rknn",
                    "status": "ok",
                    "artifact": str(output_root / "rknn"),
                    "bundle": bundle,
                    "message": "Generated RKNN conversion scripts, calibration template/helper, and chip-family notes",
                }
            )

    report_path = output_root / "reports" / "export_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
