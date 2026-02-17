from __future__ import annotations

from pathlib import Path
from typing import Any

from cds.commands.detect import run_detect


def run_infer(args: Any, repo_root: Path) -> int:
    # `infer` is a Stage 2 alias for the Stage 1 runtime detect flow.
    return run_detect(args, repo_root)
