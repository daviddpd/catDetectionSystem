from __future__ import annotations

import copy
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


def _export_rknn_wrapper_onnx(yolo_model: Any, exports_dir: Path, imgsz: int) -> Path:
    """Export an RKNN-oriented ONNX variant with boxes and scores split into separate outputs.

    Ultralytics' standard ONNX export flattens decoded boxes and class scores into one
    `(1, 4+nc, anchors)` tensor. That is convenient for general runtimes, but it mixes
    large pixel-scale box values with small class scores in one output tensor, which
    appears to quantize poorly for RKNN on some models. This wrapper keeps the decoded
    boxes and class scores as separate outputs so RKNN quantizes them independently.
    """

    try:
        import torch
    except Exception as exc:
        raise RuntimeError("RKNN export wrapper requires torch") from exc

    try:
        import onnx
    except Exception as exc:
        raise RuntimeError("RKNN export wrapper requires onnx") from exc

    class _RKNNExportWrapper(torch.nn.Module):
        def __init__(self, model: Any) -> None:
            super().__init__()
            self.model = model

        def forward(self, images: Any) -> tuple[Any, Any]:
            outputs = self.model(images)
            if not isinstance(outputs, tuple) or len(outputs) != 2:
                raise RuntimeError(
                    "Unexpected Ultralytics forward output while building RKNN-specific ONNX export"
                )
            merged, _preds = outputs
            if not isinstance(merged, torch.Tensor):
                raise RuntimeError(
                    "Ultralytics detect export wrapper expected a tensor prediction output"
                )
            if merged.ndim != 3 or int(merged.shape[1]) < 5:
                raise RuntimeError(
                    f"Ultralytics detect export wrapper expected shape (1,4+nc,N), got {tuple(merged.shape)}"
                )
            return merged[:, :4, :], merged[:, 4:, :]

    export_model = copy.deepcopy(yolo_model.model).cpu().eval().float()
    if hasattr(export_model, "fuse"):
        export_model = export_model.fuse(verbose=False)

    detect_head = getattr(export_model, "model", [None])[-1]
    if detect_head is None or detect_head.__class__.__name__ != "Detect":
        raise RuntimeError(
            "RKNN-specific ONNX export currently supports Ultralytics Detect models only"
        )

    detect_head.export = False
    detect_head.dynamic = False
    detect_head.xyxy = False
    if hasattr(detect_head, "shape"):
        detect_head.shape = None

    wrapper = _RKNNExportWrapper(export_model).eval()
    export_size = max(32, int(imgsz))
    sample = torch.zeros((1, 3, export_size, export_size), dtype=torch.float32)
    artifact = (exports_dir / "best.rknn.onnx").resolve()

    torch.onnx.export(
        wrapper,
        sample,
        str(artifact),
        export_params=True,
        do_constant_folding=True,
        opset_version=19,
        input_names=["images"],
        output_names=["boxes", "scores"],
    )

    model_onnx = onnx.load(str(artifact))
    metadata = {
        "names": str(getattr(export_model, "names", {}) or {}),
        "imgsz": str([export_size, export_size]),
        "stride": str(list(getattr(export_model, "stride", []))),
        "task": "detect",
        "cds_rknn_wrapper": "boxes_scores_split",
        "cds_rknn_output_order": "boxes,scores",
    }
    for key, value in metadata.items():
        meta = model_onnx.metadata_props.add()
        meta.key = str(key)
        meta.value = str(value)
    if getattr(model_onnx, "ir_version", 0) > 10:
        model_onnx.ir_version = 10
    onnx.save(model_onnx, str(artifact))
    return artifact


def _write_rknn_conversion_bundle(onnx_path: Path, output_root: Path) -> dict[str, str]:
    rknn_dir = output_root / "rknn"
    rknn_dir.mkdir(parents=True, exist_ok=True)

    toolkit2_script = rknn_dir / "convert_toolkit2.py"
    toolkit2_vendor_script = rknn_dir / "convert_toolkit2_vendor.py"
    legacy_script = rknn_dir / "convert_legacy.py"
    calibration_file = rknn_dir / "calibration.txt"
    calibration_helper = rknn_dir / "make_calibration_txt.py"
    smoke_test_script = rknn_dir / "smoke_test_rknn.py"
    one_shot_wrapper = rknn_dir / "run_vendor_quant_smoke.sh"

    toolkit2_output = rknn_dir / "model.toolkit2.rknn"
    toolkit2_vendor_output = rknn_dir / "model.toolkit2.vendor.rknn"
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
    toolkit2_vendor_script.write_text(
        f"""#!/usr/bin/env python3
from __future__ import annotations

# Vendor-style Toolkit2 conversion wrapper for easy comparison testing.
# This keeps the input contract explicit and matches the smoke test:
# raw RGB uint8 image data, NHWC, batched.

import argparse

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

DEFAULT_ONNX_PATH = Path(r\"{onnx_path}\")
DEFAULT_OUTPUT_PATH = Path(r\"{toolkit2_vendor_output}\")
DEFAULT_CALIBRATION_DATASET = Path(__file__).with_name("calibration.txt")
DEFAULT_TARGET_PLATFORM = "RK3588"


def _csv_triplet(raw: str, name: str) -> list[int]:
    parts = [item.strip() for item in str(raw).split(",")]
    if len(parts) != 3:
        raise SystemExit(f"Expected three comma-separated values for {{name}}, got: {{raw}}")
    values: list[int] = []
    for item in parts:
        try:
            values.append(int(item))
        except Exception as exc:
            raise SystemExit(f"Invalid integer in {{name}}={{raw}}") from exc
    return values


def _resolve_calibration_dataset(path: Path, do_quantization: bool) -> str | None:
    if not do_quantization:
        return None
    if not path.exists():
        raise SystemExit(
            "Calibration dataset file not found: "
            + str(path)
            + ". Generate it with: "
            + f"python3 {{Path(__file__).with_name('make_calibration_txt.py')}} /path/to/images"
        )
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise SystemExit(
            "Calibration dataset file is empty: "
            + str(path)
            + ". Add one absolute image path per line, or pass --no-quant."
        )
    return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert ONNX to RKNN using an explicit vendor-style Toolkit2 wrapper."
    )
    parser.add_argument(
        "--onnx",
        default=str(DEFAULT_ONNX_PATH),
        help="Input ONNX model path (default: bundle ONNX)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output RKNN path (default: bundle-local model.toolkit2.vendor.rknn)",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET_PLATFORM,
        help="Target platform, for example RK3588 (default: RK3588)",
    )
    parser.add_argument(
        "--calibration",
        default=str(DEFAULT_CALIBRATION_DATASET),
        help="Calibration txt path (default: bundle-local calibration.txt)",
    )
    parser.add_argument(
        "--mean",
        default="0,0,0",
        help="Mean values as r,g,b (default: 0,0,0)",
    )
    parser.add_argument(
        "--std",
        default="255,255,255",
        help="Std values as r,g,b (default: 255,255,255)",
    )
    parser.add_argument(
        "--no-quant",
        action="store_true",
        help="Skip quantization and export a floating-point RKNN model",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce RKNN Toolkit2 verbosity",
    )
    args = parser.parse_args()

    onnx_path = Path(args.onnx).expanduser().resolve()
    if not onnx_path.exists():
        raise SystemExit(f"ONNX model not found: {{onnx_path}}")

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    calibration_path = Path(args.calibration).expanduser().resolve()
    do_quantization = not bool(args.no_quant)
    dataset = _resolve_calibration_dataset(calibration_path, do_quantization)
    mean_values = [_csv_triplet(args.mean, "mean")]
    std_values = [_csv_triplet(args.std, "std")]

    rknn = RKNN(verbose=not args.quiet)
    rknn.config(
        target_platform=str(args.target),
        mean_values=mean_values,
        std_values=std_values,
    )
    assert rknn.load_onnx(model=str(onnx_path)) == 0
    assert rknn.build(
        do_quantization=do_quantization,
        dataset=dataset,
    ) == 0
    assert rknn.export_rknn(str(output_path)) == 0
    rknn.release()
    print("Exported", output_path)
    print("Expected runtime input: NHWC uint8 batched (raw RGB image data).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
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
    smoke_test_script.write_text(
        f"""#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception as exc:
    raise SystemExit("smoke_test_rknn.py requires opencv-python / cv2") from exc

DEFAULT_RKNN_MODEL = Path(__file__).with_name("model.toolkit2.vendor.rknn")
DEFAULT_ONNX_MODEL = Path(r\"{onnx_path}\")


def _infer_hw_from_onnx(onnx_path: Path | None) -> tuple[int, int] | None:
    if onnx_path is None or not onnx_path.exists():
        return None
    try:
        import onnx
    except Exception:
        return None
    try:
        model = onnx.load(str(onnx_path))
        if not model.graph.input:
            return None
        dims: list[int] = []
        for dim in model.graph.input[0].type.tensor_type.shape.dim:
            dims.append(int(dim.dim_value) if dim.dim_value else 0)
        if len(dims) != 4:
            return None
        if dims[1] in {{1, 3, 4}} and dims[2] > 0 and dims[3] > 0:
            return dims[2], dims[3]
        if dims[3] in {{1, 3, 4}} and dims[1] > 0 and dims[2] > 0:
            return dims[1], dims[2]
    except Exception:
        return None
    return None


def _parse_input_size(raw: str, onnx_path: Path | None) -> tuple[int, int]:
    text = str(raw).strip().lower()
    if text in {{"", "auto"}}:
        inferred = _infer_hw_from_onnx(onnx_path)
        if inferred is not None:
            return inferred
        return (640, 640)
    parts = text.replace("x", " ").split()
    if len(parts) != 2:
        raise SystemExit(f"Invalid --input-size value: {{raw}}")
    try:
        height = max(32, int(parts[0]))
        width = max(32, int(parts[1]))
    except Exception as exc:
        raise SystemExit(f"Invalid --input-size value: {{raw}}") from exc
    return height, width


def _letterbox(frame: np.ndarray, input_h: int, input_w: int) -> np.ndarray:
    shape = frame.shape[:2]
    gain = min(input_h / shape[0], input_w / shape[1])
    new_unpad = (
        int(round(shape[1] * gain)),
        int(round(shape[0] * gain)),
    )
    dw = input_w - new_unpad[0]
    dh = input_h - new_unpad[1]
    dw /= 2.0
    dh /= 2.0

    if shape[::-1] != new_unpad:
        frame = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    return cv2.copyMakeBorder(
        frame,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )


def _preprocess_rknn(frame: np.ndarray, input_h: int, input_w: int, color: str) -> np.ndarray:
    image = _letterbox(frame, input_h, input_w)
    if color == "rgb":
        image = image[..., ::-1]
    image = np.ascontiguousarray(image.astype(np.uint8, copy=False)[None])
    return image


def _preprocess_onnx(frame: np.ndarray, input_h: int, input_w: int) -> np.ndarray:
    image = _letterbox(frame, input_h, input_w)
    image = image[..., ::-1].astype(np.float32)
    image /= 255.0
    image = np.transpose(image, (2, 0, 1))
    return np.ascontiguousarray(image[None])


def _flatten_outputs(raw_outputs: object) -> list[np.ndarray]:
    if raw_outputs is None:
        return []
    if isinstance(raw_outputs, np.ndarray):
        return [raw_outputs]
    if isinstance(raw_outputs, (list, tuple)):
        outputs: list[np.ndarray] = []
        for item in raw_outputs:
            if item is None:
                continue
            arr = np.asarray(item)
            if arr.size == 0:
                continue
            outputs.append(arr)
        return outputs
    arr = np.asarray(raw_outputs)
    return [arr] if arr.size else []


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float32, copy=False), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _attribute_maxima(arr: np.ndarray, limit: int = 12) -> list[float]:
    maxima: list[float] = []
    if arr.ndim == 3 and arr.shape[0] == 1:
        attrs = min(limit, arr.shape[1])
        for idx in range(attrs):
            maxima.append(round(float(arr[:, idx : idx + 1, :].max()), 4))
        return maxima
    if arr.ndim == 4 and arr.shape[0] == 1:
        if arr.shape[1] >= arr.shape[-1]:
            attrs = min(limit, arr.shape[1])
            for idx in range(attrs):
                maxima.append(round(float(arr[:, idx : idx + 1, :, :].max()), 4))
            return maxima
        attrs = min(limit, arr.shape[-1])
        for idx in range(attrs):
            maxima.append(round(float(arr[:, :, :, idx : idx + 1].max()), 4))
        return maxima
    return maxima


def _detect_head_summary(arr: np.ndarray) -> str:
    if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[1] >= 5:
        attrs = int(arr.shape[1])
        nc = max(0, attrs - 4)
        if nc <= 0:
            return f"flat_head attrs={{attrs}}"
        cls = arr[:, 4 : 4 + nc, :]
        return (
            f"flat_head attrs={{attrs}} nc={{nc}} "
            f"cls_max_raw={{float(cls.max()):.4f}} "
            f"cls_max_sigmoid={{float(_sigmoid(cls).max()):.4f}}"
        )

    if arr.ndim == 4 and arr.shape[0] == 1:
        channel_first = arr.shape[1] >= arr.shape[-1]
        channels = int(arr.shape[1] if channel_first else arr.shape[-1])
        if channels % 3 != 0:
            return f"4d_head channels={{channels}} (not divisible by 3)"
        attrs = channels // 3
        if attrs <= 5:
            return f"4d_head channels={{channels}} attrs={{attrs}}"
        nc = attrs - 5
        if channel_first:
            head = arr.reshape(1, 3, attrs, arr.shape[2], arr.shape[3])
        else:
            head = arr.reshape(1, arr.shape[1], arr.shape[2], 3, attrs).transpose(0, 3, 4, 1, 2)
        obj = head[:, :, 4:5, :, :]
        cls = head[:, :, 5:, :, :]
        return (
            f"yolo_head anchors=3 attrs={{attrs}} nc={{nc}} "
            f"obj_max_raw={{float(obj.max()):.4f}} "
            f"obj_max_sigmoid={{float(_sigmoid(obj).max()):.4f}} "
            f"cls_max_raw={{float(cls.max()):.4f}} "
            f"cls_max_sigmoid={{float(_sigmoid(cls).max()):.4f}}"
        )

    return "unclassified_head"


def _print_output_stats(prefix: str, outputs: list[np.ndarray]) -> None:
    if not outputs:
        print(prefix + " no outputs")
        return
    split_boxes_scores = (
        len(outputs) == 2
        and np.asarray(outputs[0]).ndim == 3
        and np.asarray(outputs[1]).ndim == 3
        and int(np.asarray(outputs[0]).shape[0]) == 1
        and int(np.asarray(outputs[1]).shape[0]) == 1
        and int(np.asarray(outputs[0]).shape[1]) == 4
        and int(np.asarray(outputs[0]).shape[2]) == int(np.asarray(outputs[1]).shape[2])
    )
    for index, output in enumerate(outputs):
        arr = np.asarray(output)
        summary = _detect_head_summary(arr)
        if split_boxes_scores and index == 0:
            summary = (
                f"split_boxes anchors={{int(arr.shape[2])}} "
                f"x_max={{float(arr[:, 0:1, :].max()):.4f}} "
                f"y_max={{float(arr[:, 1:2, :].max()):.4f}} "
                f"w_max={{float(arr[:, 2:3, :].max()):.4f}} "
                f"h_max={{float(arr[:, 3:4, :].max()):.4f}}"
            )
        elif split_boxes_scores and index == 1:
            summary = (
                f"split_scores nc={{int(arr.shape[1])}} "
                f"cls_max_raw={{float(arr.max()):.4f}} "
                f"cls_max_sigmoid={{float(_sigmoid(arr).max()):.4f}}"
            )
        print(
            prefix
            + f" output[{{index}}] shape={{tuple(arr.shape)}} dtype={{arr.dtype}} "
            + f"min={{float(arr.min()):.4f}} max={{float(arr.max()):.4f}} "
            + f"attr_max={{_attribute_maxima(arr)}} "
            + summary
        )


def _run_rknn(model_path: Path, tensor: np.ndarray) -> list[np.ndarray]:
    try:
        from rknnlite.api import RKNNLite
    except Exception as exc:
        raise SystemExit("rknnlite is required for RKNN smoke testing") from exc

    runtime = RKNNLite()
    if runtime.load_rknn(str(model_path)) != 0:
        raise SystemExit(f"Failed to load RKNN model: {{model_path}}")
    if runtime.init_runtime() != 0:
        raise SystemExit("Failed to initialize RKNN runtime")

    try:
        try:
            raw_outputs = runtime.inference(
                inputs=[tensor],
                data_type=["uint8"],
                data_format=["nhwc"],
            )
        except TypeError:
            try:
                raw_outputs = runtime.inference(
                    inputs=[tensor],
                    data_format=["nhwc"],
                )
            except TypeError:
                raw_outputs = runtime.inference(inputs=[tensor])
        return _flatten_outputs(raw_outputs)
    finally:
        try:
            runtime.release()
        except Exception:
            pass


def _run_onnx(model_path: Path, tensor: np.ndarray) -> list[np.ndarray]:
    try:
        import onnxruntime as ort
    except Exception:
        print("onnxruntime not installed; skipping ONNX comparison")
        return []

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    raw_outputs = session.run(None, {{input_name: tensor}})
    return _flatten_outputs(raw_outputs)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a standalone RKNNLite smoke test and optional ONNX comparison on one image."
    )
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument(
        "--rknn-model",
        default=str(DEFAULT_RKNN_MODEL),
        help="RKNN model path (default: bundle-local model.toolkit2.vendor.rknn)",
    )
    parser.add_argument(
        "--onnx-model",
        default=str(DEFAULT_ONNX_MODEL),
        help="Optional ONNX model path for comparison (default: bundle ONNX)",
    )
    parser.add_argument(
        "--input-size",
        default="auto",
        help="Input size as HxW, or 'auto' to infer from ONNX (default: auto)",
    )
    parser.add_argument(
        "--color",
        choices=["rgb", "bgr"],
        default="rgb",
        help="Color order fed into RKNN (default: rgb)",
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise SystemExit(f"Image not found: {{image_path}}")
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"Could not read image: {{image_path}}")

    onnx_path = Path(args.onnx_model).expanduser().resolve() if args.onnx_model else None
    input_h, input_w = _parse_input_size(args.input_size, onnx_path)
    print(
        f"smoke-test image={{image_path}} size={{frame.shape[1]}}x{{frame.shape[0]}} "
        f"input={{input_w}}x{{input_h}} color={{args.color}}"
    )

    rknn_model = Path(args.rknn_model).expanduser().resolve()
    if not rknn_model.exists():
        raise SystemExit(f"RKNN model not found: {{rknn_model}}")
    rknn_tensor = _preprocess_rknn(frame, input_h, input_w, args.color)
    print(
        f"rknn input shape={{tuple(rknn_tensor.shape)}} dtype={{rknn_tensor.dtype}} "
        f"min={{int(rknn_tensor.min())}} max={{int(rknn_tensor.max())}}"
    )
    rknn_outputs = _run_rknn(rknn_model, rknn_tensor)
    _print_output_stats("rknn", rknn_outputs)

    if onnx_path is not None and onnx_path.exists():
        onnx_tensor = _preprocess_onnx(frame, input_h, input_w)
        print(
            f"onnx input shape={{tuple(onnx_tensor.shape)}} dtype={{onnx_tensor.dtype}} "
            f"min={{float(onnx_tensor.min()):.4f}} max={{float(onnx_tensor.max()):.4f}}"
        )
        onnx_outputs = _run_onnx(onnx_path, onnx_tensor)
        _print_output_stats("onnx", onnx_outputs)
    else:
        print("onnx model not provided or not found; skipping ONNX comparison")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )
    one_shot_wrapper.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [--skip-build] /path/to/image [additional smoke_test_rknn.py args...]" >&2
  exit 64
}

SKIP_BUILD=0
IMAGE=""
EXTRA=()

while (($#)); do
  case "$1" in
    --skip-build)
      SKIP_BUILD=1
      ;;
    -h|--help)
      usage
      ;;
    *)
      if [[ -z "$IMAGE" ]]; then
        IMAGE="$1"
      else
        EXTRA+=("$1")
      fi
      ;;
  esac
  shift
done

if [[ -z "$IMAGE" ]]; then
  usage
fi

BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"

if (( SKIP_BUILD == 0 )); then
  python3 "$BUNDLE_DIR/convert_toolkit2_vendor.py"
fi

python3 "$BUNDLE_DIR/smoke_test_rknn.py" \
  --image "$IMAGE" \
  --rknn-model "$BUNDLE_DIR/model.toolkit2.vendor.rknn" \
  "${EXTRA[@]}"
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
    toolkit2_vendor_script.chmod(0o755)
    legacy_script.chmod(0o755)
    calibration_helper.chmod(0o755)
    smoke_test_script.chmod(0o755)
    one_shot_wrapper.chmod(0o755)

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
        "toolkit2_vendor_script": str(toolkit2_vendor_script),
        "legacy_script": str(legacy_script),
        "calibration_file": str(calibration_file),
        "calibration_helper": str(calibration_helper),
        "smoke_test_script": str(smoke_test_script),
        "one_shot_wrapper": str(one_shot_wrapper),
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

    need_ultralytics = any(t in target_list for t in ("onnx", "coreml", "tensorrt", "rknn"))
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
    rknn_onnx_artifact: Path | None = None
    if source_is_onnx:
        onnx_artifact = exports_dir / model_path.name
        shutil.copy2(model_path, onnx_artifact)
        rknn_onnx_artifact = onnx_artifact
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
            if rknn_onnx_artifact is None and source_is_pt:
                if yolo_model is None:
                    report["results"].append(
                        {
                            "target": "rknn",
                            "status": "skipped",
                            "message": "ultralytics model unavailable for RKNN-specific ONNX export",
                        }
                    )
                    continue
                try:
                    rknn_onnx_artifact = _export_rknn_wrapper_onnx(
                        yolo_model=yolo_model,
                        exports_dir=exports_dir,
                        imgsz=imgsz,
                    )
                    report["results"].append(
                        {
                            "target": "rknn-onnx",
                            "status": "ok",
                            "artifact": str(rknn_onnx_artifact),
                            "message": "Generated RKNN-specific ONNX export with split boxes/scores outputs",
                        }
                    )
                except Exception as exc:
                    report["results"].append(
                        {
                            "target": "rknn-onnx",
                            "status": "error",
                            "message": str(exc),
                        }
                    )

            if rknn_onnx_artifact is None:
                onnx_candidates = sorted(
                    path
                    for path in exports_dir.glob("*.onnx")
                    if path.name.lower().endswith(".rknn.onnx")
                )
                if onnx_candidates:
                    rknn_onnx_artifact = onnx_candidates[0]

            if rknn_onnx_artifact is None:
                if onnx_artifact is None:
                    onnx_candidates = sorted(exports_dir.glob("*.onnx"))
                    if onnx_candidates:
                        onnx_artifact = onnx_candidates[0]
                rknn_onnx_artifact = onnx_artifact

            if rknn_onnx_artifact is None:
                report["results"].append(
                    {
                        "target": "rknn",
                        "status": "skipped",
                        "message": "ONNX artifact required before RKNN conversion bundle generation",
                    }
                )
                continue

            bundle = _write_rknn_conversion_bundle(rknn_onnx_artifact, output_root)
            report["results"].append(
                {
                    "target": "rknn",
                    "status": "ok",
                    "artifact": str(output_root / "rknn"),
                    "onnx_artifact": str(rknn_onnx_artifact),
                    "bundle": bundle,
                    "message": "Generated RKNN conversion scripts, calibration template/helper, and chip-family notes",
                }
            )

    report_path = output_root / "reports" / "export_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
