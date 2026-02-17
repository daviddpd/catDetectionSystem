from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from cds.training.active_learning import merge_review_queue, queue_uncertain_events
from cds.training.bootstrap import run_bootstrap_openvocab
from cds.training.prefetch import prefetch_recommended_models
from cds.training.train import run_finetune_training


def _parse_class_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def run_train(args: Any, repo_root: Path) -> int:
    try:
        if args.train_command == "bootstrap-openvocab":
            artifact_root = Path(args.output_dir or "artifacts/models")
            if not artifact_root.is_absolute():
                artifact_root = (repo_root / artifact_root).resolve()

            summary = run_bootstrap_openvocab(
                source=args.source,
                classes_csv=args.classes,
                output_dir=artifact_root,
                model_name=args.model,
                conf=float(args.conf),
                imgsz=int(args.imgsz),
                max_frames=int(args.max_frames),
                materialize_non_image_frames=bool(args.materialize_non_image_frames),
            )
            print(json.dumps(summary, ensure_ascii=True, indent=2))
            return 0

        if args.train_command == "prefetch-models":
            report_dir = None
            if args.output_dir:
                report_dir = Path(args.output_dir)
                if not report_dir.is_absolute():
                    report_dir = (repo_root / report_dir).resolve()
            summary = prefetch_recommended_models(output_dir=report_dir)
            print(json.dumps(summary, ensure_ascii=True, indent=2))
            return 0

        if args.train_command == "active-learning":
            if args.active_command == "queue":
                queue_path = Path(args.output)
                if not queue_path.is_absolute():
                    queue_path = (repo_root / queue_path).resolve()
                events = Path(args.events)
                if not events.is_absolute():
                    events = (repo_root / events).resolve()
                truth_path = None
                if args.truth:
                    truth_path = Path(args.truth)
                    if not truth_path.is_absolute():
                        truth_path = (repo_root / truth_path).resolve()
                    else:
                        truth_path = truth_path.resolve()

                summary = queue_uncertain_events(
                    events_jsonl=events,
                    output_path=queue_path,
                    min_conf=float(args.min_conf),
                    max_conf=float(args.max_conf),
                    class_filter=_parse_class_list(args.class_filter),
                    truth_jsonl=truth_path,
                )
                print(json.dumps(summary, ensure_ascii=True, indent=2))
                return 0

            if args.active_command == "merge":
                queue = Path(args.queue)
                if not queue.is_absolute():
                    queue = (repo_root / queue).resolve()
                source_root = Path(args.source_images)
                if not source_root.is_absolute():
                    source_root = (repo_root / source_root).resolve()
                dataset_root = Path(args.dataset)
                if not dataset_root.is_absolute():
                    dataset_root = (repo_root / dataset_root).resolve()

                summary = merge_review_queue(
                    queue_jsonl=queue,
                    source_images_root=source_root,
                    dataset_root=dataset_root,
                    target_split=args.split,
                )
                print(json.dumps(summary, ensure_ascii=True, indent=2))
                return 0

            raise RuntimeError("Unknown active-learning command")

        overrides: dict[str, Any] = {
            "experiment": {
                "artifact_root": args.output_dir,
                "name": args.experiment_name,
            },
            "dataset": {
                "data_yaml": args.dataset,
            },
            "model": {
                "base": args.model,
                "imgsz": args.imgsz,
            },
            "train": {
                "epochs": args.epochs,
                "batch": args.batch,
                "device": args.device,
                "amp": (False if args.no_amp else None),
            },
            "export": {
                "enabled": (False if args.no_export else None),
                "targets": args.export_targets,
            },
        }

        # remove None leaves
        def prune(obj: Any) -> Any:
            if isinstance(obj, dict):
                cleaned = {k: prune(v) for k, v in obj.items() if v is not None}
                return {k: v for k, v in cleaned.items() if v not in ({}, [], None)}
            return obj

        summary = run_finetune_training(
            repo_root=repo_root,
            config_path=args.config,
            cli_overrides=prune(overrides),
        )
        print(json.dumps(summary, ensure_ascii=True, indent=2))
        return 0
    except Exception as exc:
        print(f"train command failed: {exc}", file=sys.stderr)
        return 2
