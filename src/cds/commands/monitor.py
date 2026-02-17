from __future__ import annotations

import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cds.io.ingest import probe_decoder_path


def _load_avg() -> tuple[float, float, float] | None:
    try:
        return os.getloadavg()
    except Exception:
        return None


def run_monitor(args: Any, repo_root: Path) -> int:
    _ = repo_root
    interval = max(1.0, float(args.interval))
    count = int(args.count)

    iterations = 0
    while count <= 0 or iterations < count:
        decoder = probe_decoder_path()
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "host": platform.node(),
            "system": platform.system(),
            "machine": platform.machine(),
            "decoder": decoder.selected_decoder,
            "decoder_reason": decoder.reason,
            "loadavg": _load_avg(),
        }

        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print(
                f"[{payload['ts']}] decoder={payload['decoder']} "
                f"machine={payload['machine']} loadavg={payload['loadavg']}"
            )

        iterations += 1
        if count <= 0 or iterations < count:
            time.sleep(interval)

    return 0
