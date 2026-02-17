from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from cds.training.evaluate import run_evaluation


def run_evaluate(args: Any, repo_root: Path) -> int:
    try:
        overrides: dict[str, Any] = {
            "model": {"path": args.model},
            "dataset": {"data_yaml": args.dataset, "split": args.split},
            "eval": {
                "device": args.device,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "conf": args.conf,
            },
        }

        def prune(obj):
            if isinstance(obj, dict):
                result = {k: prune(v) for k, v in obj.items() if v is not None}
                return {k: v for k, v in result.items() if v not in ({}, None)}
            return obj

        report = run_evaluation(
            repo_root=repo_root,
            config_path=args.config,
            cli_overrides=prune(overrides),
        )
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return 0
    except Exception as exc:
        print(f"evaluate command failed: {exc}", file=sys.stderr)
        return 2
