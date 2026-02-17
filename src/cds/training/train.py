from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cds.training.constants import DEFAULT_TRAIN_MODEL
from cds.training.export import export_model_artifacts
from cds.training.paths import new_run_id, prepare_artifact_dirs
from cds.utils.config_io import deep_merge, load_config_file


def _default_train_config() -> dict[str, Any]:
    return {
        "experiment": {
            "name": "communitycats-prod",
            "artifact_root": "artifacts/models",
        },
        "dataset": {
            "data_yaml": "dataset/data.yaml",
        },
        "model": {
            "base": DEFAULT_TRAIN_MODEL,
            "imgsz": 640,
        },
        "train": {
            "epochs": 50,
            "batch": 16,
            "device": "cpu",
            "amp": True,
            "workers": 4,
            "patience": 20,
        },
        "export": {
            "enabled": True,
            "targets": "onnx,coreml,tensorrt,rknn",
        },
        "gating": {
            "required_classes": [],
            "recall_thresholds": {},
        },
    }


def _validate_required_classes(cfg: dict[str, Any]) -> None:
    gating = cfg.get("gating", {})
    required_classes = [str(item).strip() for item in gating.get("required_classes", []) if str(item).strip()]
    if not required_classes:
        raise ValueError("train config must define non-empty gating.required_classes")

    recall_thresholds = gating.get("recall_thresholds", {}) or {}
    missing = [name for name in required_classes if name not in recall_thresholds]
    if missing:
        raise ValueError(
            "train config gating.recall_thresholds missing required classes: "
            + ", ".join(missing)
        )

    gating["required_classes"] = required_classes
    cfg["gating"] = gating


def load_train_config(config_path: str | None, cli_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = _default_train_config()
    file_cfg = load_config_file(config_path)
    deep_merge(config, file_cfg)
    if cli_overrides:
        deep_merge(config, cli_overrides)
    _validate_required_classes(config)
    return config


def run_finetune_training(
    repo_root: Path,
    config_path: str | None,
    cli_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = load_train_config(config_path, cli_overrides)

    artifact_root = Path(cfg["experiment"]["artifact_root"])
    if not artifact_root.is_absolute():
        artifact_root = (repo_root / artifact_root).resolve()
    run_id = new_run_id(cfg["experiment"].get("name", "train"))
    dirs = prepare_artifact_dirs(artifact_root, run_id)

    data_yaml = Path(cfg["dataset"]["data_yaml"])
    if not data_yaml.is_absolute():
        data_yaml = (repo_root / data_yaml).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Training requires ultralytics package") from exc

    model_base = cfg["model"]["base"]
    model = YOLO(model_base)

    train_args = {
        "data": str(data_yaml),
        "epochs": int(cfg["train"]["epochs"]),
        "imgsz": int(cfg["model"].get("imgsz", 640)),
        "batch": int(cfg["train"]["batch"]),
        "device": str(cfg["train"].get("device", "cpu")),
        "amp": bool(cfg["train"].get("amp", True)),
        "workers": int(cfg["train"].get("workers", 4)),
        "patience": int(cfg["train"].get("patience", 20)),
        "project": str(dirs["root"]),
        "name": "ultralytics_train",
    }

    results = model.train(**train_args)

    best = None
    if hasattr(results, "save_dir"):
        candidate = Path(str(results.save_dir)) / "weights" / "best.pt"
        if candidate.exists():
            best = candidate

    if best is None:
        fallback = dirs["root"] / "ultralytics_train" / "weights" / "best.pt"
        if fallback.exists():
            best = fallback

    if best is None:
        raise RuntimeError("Training finished but best.pt was not found")

    best_dest = dirs["checkpoints"] / "best.pt"
    best_dest.write_bytes(best.read_bytes())

    summary: dict[str, Any] = {
        "run_id": run_id,
        "artifact_root": str(dirs["root"]),
        "checkpoint": str(best_dest),
        "train_args": train_args,
    }

    if cfg.get("export", {}).get("enabled", True):
        export_report = export_model_artifacts(
            model_path=best_dest,
            output_root=dirs["root"],
            targets_csv=str(cfg.get("export", {}).get("targets", "all")),
            imgsz=int(cfg["model"].get("imgsz", 640)),
            half=bool(cfg["train"].get("amp", True)),
        )
        summary["export"] = export_report

    summary_path = dirs["reports"] / "train_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
