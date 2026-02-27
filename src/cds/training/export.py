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

    toolkit2_output = rknn_dir / "model.toolkit2.rknn"
    legacy_output = rknn_dir / "model.legacy.rknn"

    toolkit2_script.write_text(
        f"""#!/usr/bin/env python3
from __future__ import annotations

# Toolkit2 conversion template. Run on an RKNN Toolkit2-capable host.
# Update target_platform and dataset calibration file before execution.

try:
    import pkg_resources  # noqa: F401
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing Python package 'setuptools' (provides pkg_resources). "
        "Install it in this environment: python3 -m pip install setuptools"
    ) from exc

from rknn.api import RKNN

ONNX_PATH = r\"{onnx_path}\"
OUTPUT_PATH = r\"{toolkit2_output}\"
TARGET_PLATFORM = "RK3588"
CALIBRATION_DATASET = "./calibration.txt"

rknn = RKNN(verbose=True)
rknn.config(target_platform=TARGET_PLATFORM)
assert rknn.load_onnx(model=ONNX_PATH) == 0
assert rknn.build(do_quantization=True, dataset=CALIBRATION_DATASET) == 0
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
    import pkg_resources  # noqa: F401
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing Python package 'setuptools' (provides pkg_resources). "
        "Install it in this environment: python3 -m pip install setuptools"
    ) from exc

from rknn.api import RKNN

ONNX_PATH = r\"{onnx_path}\"
OUTPUT_PATH = r\"{legacy_output}\"
TARGET_PLATFORM = "RK1808"
CALIBRATION_DATASET = "./calibration.txt"

rknn = RKNN(verbose=True)
rknn.config(target_platform=TARGET_PLATFORM)
assert rknn.load_onnx(model=ONNX_PATH) == 0
assert rknn.build(do_quantization=True, dataset=CALIBRATION_DATASET) == 0
assert rknn.export_rknn(OUTPUT_PATH) == 0
rknn.release()
print("Exported", OUTPUT_PATH)
""",
        encoding="utf-8",
    )

    toolkit2_script.chmod(0o755)
    legacy_script.chmod(0o755)

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
                    "message": "Generated RKNN conversion scripts and chip-family notes",
                }
            )

    report_path = output_root / "reports" / "export_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
