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
rknn.config(target_platform=TARGET_PLATFORM)
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
rknn.config(target_platform=TARGET_PLATFORM)
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
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _iter_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


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
