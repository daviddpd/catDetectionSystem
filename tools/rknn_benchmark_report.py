#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

STATS_RE = re.compile(
    r"stats fps_decode=(?P<fps_decode>[0-9.]+) "
    r"fps_in=(?P<fps_in>[0-9.]+) "
    r"fps_infer=(?P<fps_infer>[0-9.]+) "
    r"sampled_frames=(?P<sampled_frames>\d+) "
    r"dropped_frames=(?P<dropped_frames>\d+) "
    r"queue_depth=(?P<queue_depth>\d+) "
    r"frame_age_ms=(?P<frame_age_ms>[0-9.]+)"
)
EFFECTIVE_RE = re.compile(
    r"effective model input "
    r"height=(?P<height>\d+) width=(?P<width>\d+) "
    r"format=(?P<format>\S+) dtype=(?P<dtype>\S+) batch=(?P<batch>\S+) "
    r"scale=(?P<scale>\S+) color=(?P<color>\S+) backend=(?P<backend>\S+)"
)
PERF_TOKEN_RE = {
    "model_path": re.compile(r"\bmodel_path=(?P<value>\S+)"),
    "configured_imgsz": re.compile(r"\bconfigured_imgsz=(?P<value>\d+)"),
    "backend": re.compile(r"\bbackend=(?P<value>\S+)"),
    "ingest": re.compile(r"\bingest=(?P<value>\S+)"),
    "benchmark": re.compile(r"\bbenchmark=(?P<value>\S+)"),
}


@dataclass
class NumericSeries:
    values: list[float]

    def summary(self) -> dict[str, float]:
        if not self.values:
            return {}
        count = len(self.values)
        mean = sum(self.values) / count
        minimum = min(self.values)
        maximum = max(self.values)
        if count > 1:
            variance = sum((v - mean) ** 2 for v in self.values) / count
            stdev = math.sqrt(variance)
        else:
            stdev = 0.0
        return {
            "count": float(count),
            "mean": mean,
            "min": minimum,
            "max": maximum,
            "stdev": stdev,
            "latest": self.values[-1],
        }


def _parse_log(path: Path) -> dict:
    fps_decode: list[float] = []
    fps_infer: list[float] = []
    frame_age_ms: list[float] = []
    queue_depth: list[float] = []
    sampled_frames: list[float] = []
    dropped_frames: list[float] = []
    effective = None
    perf: dict[str, str] | None = None

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stats_match = STATS_RE.search(raw_line)
        if stats_match:
            fps_decode.append(float(stats_match.group("fps_decode")))
            fps_infer.append(float(stats_match.group("fps_infer")))
            frame_age_ms.append(float(stats_match.group("frame_age_ms")))
            queue_depth.append(float(stats_match.group("queue_depth")))
            sampled_frames.append(float(stats_match.group("sampled_frames")))
            dropped_frames.append(float(stats_match.group("dropped_frames")))
            continue

        effective_match = EFFECTIVE_RE.search(raw_line)
        if effective_match and effective is None:
            effective = effective_match.groupdict()
            continue

        if "perf config " in raw_line and perf is None:
            perf = {}
            for key, pattern in PERF_TOKEN_RE.items():
                match = pattern.search(raw_line)
                if match:
                    perf[key] = match.group("value")
            continue

    return {
        "log_path": str(path),
        "perf": perf or {},
        "effective_input": effective or {},
        "fps_decode": NumericSeries(fps_decode).summary(),
        "fps_infer": NumericSeries(fps_infer).summary(),
        "frame_age_ms": NumericSeries(frame_age_ms).summary(),
        "queue_depth": NumericSeries(queue_depth).summary(),
        "sampled_frames": NumericSeries(sampled_frames).summary(),
        "dropped_frames": NumericSeries(dropped_frames).summary(),
    }


def _compact(summary: dict) -> str:
    perf = summary.get("perf", {})
    effective = summary.get("effective_input", {})
    fps_decode = summary.get("fps_decode", {})
    fps_infer = summary.get("fps_infer", {})
    frame_age = summary.get("frame_age_ms", {})
    status = "ok"
    if perf and not fps_infer:
        status = "no-frames"
    elif not perf and not fps_infer:
        status = "startup-failed"

    return (
        f"log={summary.get('log_path', '')} "
        f"status={status} "
        f"ingest={perf.get('ingest', '?')} "
        f"configured_imgsz={perf.get('configured_imgsz', '?')} "
        f"effective={effective.get('width', '?')}x{effective.get('height', '?')} "
        f"{effective.get('dtype', '?')}/{effective.get('format', '?')} "
        f"fps_decode_avg={fps_decode.get('mean', float('nan')):.2f} "
        f"fps_infer_avg={fps_infer.get('mean', float('nan')):.2f} "
        f"frame_age_avg_ms={frame_age.get('mean', float('nan')):.1f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize CDS benchmark logs into the key FPS/latency metrics.")
    parser.add_argument("logs", nargs="+", help="One or more CDS log files to summarize")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Summary output format")
    args = parser.parse_args()

    summaries = [_parse_log(Path(item)) for item in args.logs]
    if args.format == "json":
        print(json.dumps(summaries, indent=2))
        return 0

    for summary in summaries:
        print(_compact(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
