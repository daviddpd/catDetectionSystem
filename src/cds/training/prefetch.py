from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

from cds.training.constants import DEFAULT_BOOTSTRAP_MODEL, DEFAULT_TRAIN_MODEL


def _detect_engine() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if platform.system().lower() == "darwin" and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _recommended_train_model(engine: str) -> str:
    if engine == "cuda":
        return "yolov8s.pt"
    if engine == "mps":
        return "yolov8s.pt"
    return "yolov8n.pt"


def prefetch_recommended_models(output_dir: Path | None = None) -> dict[str, Any]:
    engine = _detect_engine()
    train_model_name = _recommended_train_model(engine)
    bootstrap_model_name = DEFAULT_BOOTSTRAP_MODEL

    payload: dict[str, Any] = {
        "detected_engine": engine,
        "recommended": {
            "train_model": train_model_name,
            "bootstrap_model": bootstrap_model_name,
        },
        "pulled": [],
        "warnings": [],
    }

    try:
        from ultralytics import YOLO, YOLOWorld
    except Exception:
        payload["warnings"].append(
            "ultralytics is not installed; cannot prefetch automatically. "
            "Install requirements and retry."
        )
        return payload

    try:
        train_model = YOLO(train_model_name)
        payload["pulled"].append(
            {
                "model": train_model_name,
                "type": "train-base",
                "resolved_path": str(getattr(train_model, "ckpt_path", train_model_name)),
            }
        )
    except Exception as exc:
        payload["warnings"].append(f"failed to pull {train_model_name}: {exc}")

    try:
        world_model = YOLOWorld(bootstrap_model_name)
        payload["pulled"].append(
            {
                "model": bootstrap_model_name,
                "type": "bootstrap-openvocab",
                "resolved_path": str(getattr(world_model, "ckpt_path", bootstrap_model_name)),
            }
        )
    except Exception as exc:
        payload["warnings"].append(f"failed to pull {bootstrap_model_name}: {exc}")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "prefetch_report.json"
        import json

        report_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        payload["report_path"] = str(report_path)

    return payload
