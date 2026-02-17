from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from cds.training.export import export_model_artifacts
from cds.utils.config_io import deep_merge, load_config_file


def run_export(args: Any, repo_root: Path) -> int:
    try:
        cfg = {
            "model": {"path": None},
            "export": {
                "output_dir": "artifacts/models/export-manual",
                "targets": "all",
                "imgsz": 640,
                "half": False,
            },
        }

        deep_merge(cfg, load_config_file(args.config))

        overrides = {
            "model": {"path": args.model},
            "export": {
                "output_dir": args.output_dir,
                "targets": args.targets,
                "imgsz": args.imgsz,
                "half": args.half,
            },
        }

        # prune Nones
        def prune(obj):
            if isinstance(obj, dict):
                result = {k: prune(v) for k, v in obj.items() if v is not None}
                return {k: v for k, v in result.items() if v not in ({}, None)}
            return obj

        deep_merge(cfg, prune(overrides))

        model_path = Path(cfg["model"]["path"])
        if not model_path.is_absolute():
            model_path = (repo_root / model_path).resolve()

        output_dir = Path(cfg["export"]["output_dir"])
        if not output_dir.is_absolute():
            output_dir = (repo_root / output_dir).resolve()

        report = export_model_artifacts(
            model_path=model_path,
            output_root=output_dir,
            targets_csv=str(cfg["export"].get("targets", "all")),
            imgsz=int(cfg["export"].get("imgsz", 640)),
            half=bool(cfg["export"].get("half", False)),
        )
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return 0
    except Exception as exc:
        print(f"export command failed: {exc}", file=sys.stderr)
        return 2
