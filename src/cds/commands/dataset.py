from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from cds.training.dataset.prepare import prepare_dataset_pipeline
from cds.training.dataset.validate import validate_yolo_dataset
from cds.utils.config_io import deep_merge, load_config_file, write_json


def _parse_classes(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _require_classes(classes: list[str], context: str) -> list[str]:
    filtered = [item for item in classes if item]
    if not filtered:
        raise ValueError(
            f"{context} requires classes via --classes or config dataset.classes"
        )
    return filtered


def run_dataset(args: Any, repo_root: Path) -> int:
    try:
        if args.dataset_command == "prepare":
            cfg = {
                "dataset": {
                    "output_root": "dataset",
                    "xml_root": "annotations/xml",
                    "image_root": None,
                    "classes": [],
                    "split_mode": "deterministic",
                    "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
                }
            }
            deep_merge(cfg, load_config_file(args.config))

            classes = _require_classes(
                _parse_classes(args.classes) or list(cfg["dataset"].get("classes", [])),
                "dataset prepare",
            )

            output_root = Path(args.output_root or cfg["dataset"]["output_root"])
            xml_root = Path(args.xml_root or cfg["dataset"]["xml_root"])
            image_root_value = args.image_root if args.image_root is not None else cfg["dataset"].get("image_root")
            image_root = Path(image_root_value) if image_root_value else None

            if not output_root.is_absolute():
                output_root = (repo_root / output_root).resolve()
            if not xml_root.is_absolute():
                xml_root = (repo_root / xml_root).resolve()
            if image_root is not None and not image_root.is_absolute():
                image_root = (repo_root / image_root).resolve()

            ratios = cfg["dataset"].get("split_ratios", {})
            summary = prepare_dataset_pipeline(
                output_root=output_root,
                xml_root=xml_root,
                image_root=image_root,
                class_names=classes,
                split_mode=str(args.split_mode or cfg["dataset"].get("split_mode", "deterministic")),
                train_ratio=float(ratios.get("train", 0.8)),
                val_ratio=float(ratios.get("val", 0.1)),
                test_ratio=float(ratios.get("test", 0.1)),
            )
            print(json.dumps(summary, ensure_ascii=True, indent=2))
            health_status = summary.get("health", {}).get("status", "fail")
            return 0 if health_status == "pass" else 1

        if args.dataset_command == "validate":
            cfg = {
                "dataset": {
                    "root": "dataset",
                    "classes": [],
                    "report": "dataset/reports/dataset_health.json",
                }
            }
            deep_merge(cfg, load_config_file(args.config))

            dataset_root = Path(args.dataset_root or cfg["dataset"]["root"])
            if not dataset_root.is_absolute():
                dataset_root = (repo_root / dataset_root).resolve()

            classes = _require_classes(
                _parse_classes(args.classes) or list(cfg["dataset"].get("classes", [])),
                "dataset validate",
            )
            report = validate_yolo_dataset(dataset_root, classes)

            report_path = Path(args.report or cfg["dataset"].get("report", dataset_root / "reports" / "dataset_health.json"))
            if not report_path.is_absolute():
                report_path = (repo_root / report_path).resolve()
            write_json(report_path, report)

            payload = {
                "report_path": str(report_path),
                "status": report["status"],
                "counts": report["counts"],
                "issue_count": len(report.get("issues", [])),
            }
            print(json.dumps(payload, ensure_ascii=True, indent=2))
            return 0 if report["status"] == "pass" else 1

        raise RuntimeError(f"Unsupported dataset command: {args.dataset_command}")
    except Exception as exc:
        print(f"dataset command failed: {exc}", file=sys.stderr)
        return 2
