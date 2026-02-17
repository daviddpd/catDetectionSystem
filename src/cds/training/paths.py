from __future__ import annotations

from datetime import datetime
from pathlib import Path


def new_run_id(prefix: str) -> str:
    return f"{prefix}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"


def prepare_artifact_dirs(base_dir: Path, run_id: str) -> dict[str, Path]:
    root = base_dir / run_id
    dirs = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "exports": root / "exports",
        "reports": root / "reports",
        "bootstrap": root / "bootstrap",
        "active_learning": root / "active_learning",
        "rknn": root / "rknn",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
