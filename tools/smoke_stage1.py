#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def run_case(name: str, cmd: list[str], timeout_seconds: int) -> int:
    print(f"[smoke] case={name} cmd={' '.join(shlex.quote(p) for p in cmd)}")
    try:
        completed = subprocess.run(cmd, timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
        print(f"[smoke] case={name} timeout after {timeout_seconds}s")
        return 124

    print(f"[smoke] case={name} rc={completed.returncode}")
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1 smoke tests")
    parser.add_argument("--rtsp-uri", required=True)
    parser.add_argument("--video-file", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--cds", default="./cds", help="Path to cds launcher")
    parser.add_argument("--timeout", type=int, default=20, help="Timeout per case in seconds")

    args = parser.parse_args()

    cds_path = Path(args.cds).resolve()
    if not cds_path.exists():
        print(f"cds launcher not found: {cds_path}", file=sys.stderr)
        return 2

    base = [
        str(cds_path),
        "detect",
        "--model-path",
        args.model_path,
        "--headless",
        "--no-event-stdout",
        "--log-level",
        "WARNING",
    ]

    cases = [
        ("rtsp", base + ["--uri", args.rtsp_uri]),
        ("video", base + ["--uri", args.video_file]),
        ("image_dir", base + ["--uri", args.image_dir]),
    ]

    rc = 0
    for name, cmd in cases:
        case_rc = run_case(name, cmd, args.timeout)
        if case_rc != 0 and rc == 0:
            rc = case_rc

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
