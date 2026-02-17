from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cds.utils.config_io import deep_merge, load_config_file


def _default_eval_config() -> dict[str, Any]:
    return {
        "model": {"path": "artifacts/models/latest/checkpoints/best.pt"},
        "dataset": {"data_yaml": "dataset/data.yaml", "split": "val"},
        "eval": {
            "device": "cpu",
            "imgsz": 640,
            "batch": 16,
            "conf": 0.25,
        },
        "gating": {
            "required_classes": [],
            "recall_thresholds": {},
        },
        "subgroups": {
            "enabled": False,
            "day_data_yaml": "",
            "night_data_yaml": "",
        },
        "latency": {
            "runtime_benchmark_file": "",
        },
    }


def _validate_required_classes(cfg: dict[str, Any]) -> None:
    gating = cfg.get("gating", {})
    required_classes = [str(item).strip() for item in gating.get("required_classes", []) if str(item).strip()]
    if not required_classes:
        raise ValueError("eval config must define non-empty gating.required_classes")

    recall_thresholds = gating.get("recall_thresholds", {}) or {}
    missing = [name for name in required_classes if name not in recall_thresholds]
    if missing:
        raise ValueError(
            "eval config gating.recall_thresholds missing required classes: "
            + ", ".join(missing)
        )

    gating["required_classes"] = required_classes
    cfg["gating"] = gating


def load_eval_config(config_path: str | None, cli_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = _default_eval_config()
    deep_merge(cfg, load_config_file(config_path))
    if cli_overrides:
        deep_merge(cfg, cli_overrides)
    _validate_required_classes(cfg)
    return cfg


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _extract_metrics(results: Any, required_classes: list[str]) -> dict[str, Any]:
    metrics = {
        "map50": 0.0,
        "map50_95": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "per_class": {},
        "confusion_matrix_path": None,
    }

    results_dict = getattr(results, "results_dict", {}) or {}
    metrics["map50"] = _safe_float(results_dict.get("metrics/mAP50(B)", 0.0))
    metrics["map50_95"] = _safe_float(results_dict.get("metrics/mAP50-95(B)", 0.0))
    metrics["precision"] = _safe_float(results_dict.get("metrics/precision(B)", 0.0))
    metrics["recall"] = _safe_float(results_dict.get("metrics/recall(B)", 0.0))

    names = getattr(results, "names", {}) or {}
    name_to_idx = {str(v): int(k) for k, v in names.items()} if isinstance(names, dict) else {}
    box = getattr(results, "box", None)
    per_class_precision = getattr(box, "p", []) if box is not None else []
    per_class_recall = getattr(box, "r", []) if box is not None else []
    per_class_map50 = getattr(box, "ap50", []) if box is not None else []

    for fallback_idx, class_name in enumerate(required_classes):
        idx = name_to_idx.get(class_name, fallback_idx)
        name_from_results = names.get(idx, class_name) if isinstance(names, dict) else class_name
        metrics["per_class"][class_name] = {
            "reported_name": str(name_from_results),
            "precision": _safe_float(per_class_precision[idx] if idx < len(per_class_precision) else 0.0),
            "recall": _safe_float(per_class_recall[idx] if idx < len(per_class_recall) else 0.0),
            "map50": _safe_float(per_class_map50[idx] if idx < len(per_class_map50) else 0.0),
        }

    save_dir = Path(str(getattr(results, "save_dir", "")))
    matrix = save_dir / "confusion_matrix.png"
    if matrix.exists():
        metrics["confusion_matrix_path"] = str(matrix)

    return metrics


def _production_gate(metrics: dict[str, Any], recall_thresholds: dict[str, float]) -> dict[str, Any]:
    failing: list[dict[str, Any]] = []
    per_class = metrics.get("per_class", {})
    for class_name, threshold in recall_thresholds.items():
        recall = _safe_float(per_class.get(class_name, {}).get("recall", 0.0))
        if recall < float(threshold):
            failing.append(
                {
                    "class": class_name,
                    "recall": recall,
                    "threshold": float(threshold),
                }
            )

    return {
        "is_production_candidate": len(failing) == 0,
        "failing_classes": failing,
    }


def run_evaluation(
    repo_root: Path,
    config_path: str | None,
    cli_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = load_eval_config(config_path, cli_overrides)

    model_path = Path(cfg["model"]["path"])
    if not model_path.is_absolute():
        model_path = (repo_root / model_path).resolve()

    data_yaml = Path(cfg["dataset"]["data_yaml"])
    if not data_yaml.is_absolute():
        data_yaml = (repo_root / data_yaml).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Evaluation requires ultralytics package") from exc

    model = YOLO(str(model_path))

    val_results = model.val(
        data=str(data_yaml),
        split=str(cfg["dataset"].get("split", "val")),
        imgsz=int(cfg["eval"].get("imgsz", 640)),
        batch=int(cfg["eval"].get("batch", 16)),
        device=str(cfg["eval"].get("device", "cpu")),
        conf=float(cfg["eval"].get("conf", 0.25)),
        plots=True,
        save_json=True,
    )

    required_classes = list(cfg["gating"].get("required_classes", []))
    recall_thresholds = {
        name: float(value)
        for name, value in cfg["gating"].get("recall_thresholds", {}).items()
    }

    metrics = _extract_metrics(val_results, required_classes)
    gate = _production_gate(metrics, recall_thresholds)

    subgroup_metrics: dict[str, Any] = {}
    subgroup_cfg = cfg.get("subgroups", {})
    if subgroup_cfg.get("enabled", False):
        for subgroup_name, key in (("day", "day_data_yaml"), ("night", "night_data_yaml")):
            subgroup_yaml = subgroup_cfg.get(key)
            if not subgroup_yaml:
                continue
            subgroup_path = Path(subgroup_yaml)
            if not subgroup_path.is_absolute():
                subgroup_path = (repo_root / subgroup_path).resolve()
            if not subgroup_path.exists():
                continue
            subgroup_result = model.val(
                data=str(subgroup_path),
                split=str(cfg["dataset"].get("split", "val")),
                imgsz=int(cfg["eval"].get("imgsz", 640)),
                batch=int(cfg["eval"].get("batch", 16)),
                device=str(cfg["eval"].get("device", "cpu")),
                conf=float(cfg["eval"].get("conf", 0.25)),
                plots=False,
                save_json=True,
            )
            subgroup_metrics[subgroup_name] = _extract_metrics(subgroup_result, required_classes)

    latency = {}
    latency_file = cfg.get("latency", {}).get("runtime_benchmark_file")
    if latency_file:
        p = Path(latency_file)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        if p.exists():
            try:
                latency = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                latency = {"note": f"Could not parse latency benchmark file: {p}"}

    report = {
        "model": str(model_path),
        "dataset": str(data_yaml),
        "metrics": metrics,
        "subgroup_metrics": subgroup_metrics,
        "latency": latency,
        "gating": gate,
    }

    save_dir = Path(str(getattr(val_results, "save_dir", repo_root / "artifacts" / "eval")))
    save_dir.mkdir(parents=True, exist_ok=True)
    report_path = save_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)

    return report
