from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class MetricsSnapshot:
    fps_decode: float
    fps_in: float
    fps_infer: float
    sampled_frames: int
    dropped_frames: int
    queue_depth: int
    frame_age_ms: float


class RuntimeMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start = time.monotonic()
        self._frames_decode = 0
        self._frames_in = 0
        self._frames_infer = 0
        self._sampled_frames = 0
        self._dropped_frames = 0
        self._queue_depth = 0
        self._frame_age_ms = 0.0

        self._prometheus_started = False
        self._prometheus_counters = None

    def enable_prometheus(self, host: str, port: int) -> bool:
        try:
            from prometheus_client import Counter, Gauge, start_http_server
        except ImportError:
            return False

        if self._prometheus_started:
            return True

        start_http_server(port, addr=host)
        self._prometheus_started = True
        self._prometheus_counters = {
            "frames_decode": Counter("cds_frames_decode_total", "Frames decoded/read from source"),
            "frames_in": Counter("cds_frames_in_total", "Frames ingested"),
            "frames_infer": Counter("cds_frames_infer_total", "Frames inferred"),
            "sampled": Counter("cds_frames_sampled_total", "Frames sampled out before inference"),
            "dropped": Counter("cds_frames_dropped_total", "Frames dropped"),
            "queue_depth": Gauge("cds_queue_depth", "Ingest to infer queue depth"),
            "frame_age_ms": Gauge("cds_frame_age_ms", "Age of rendered frame in milliseconds"),
        }
        return True

    def mark_decode(self) -> None:
        with self._lock:
            self._frames_decode += 1
            if self._prometheus_counters:
                self._prometheus_counters["frames_decode"].inc()

    def mark_ingest(self) -> None:
        with self._lock:
            self._frames_in += 1
            if self._prometheus_counters:
                self._prometheus_counters["frames_in"].inc()

    def mark_infer(self) -> None:
        with self._lock:
            self._frames_infer += 1
            if self._prometheus_counters:
                self._prometheus_counters["frames_infer"].inc()

    def add_sampled_out(self, count: int = 1) -> None:
        with self._lock:
            self._sampled_frames += count
            if self._prometheus_counters:
                self._prometheus_counters["sampled"].inc(count)

    def add_dropped(self, count: int = 1) -> None:
        with self._lock:
            self._dropped_frames += count
            if self._prometheus_counters:
                self._prometheus_counters["dropped"].inc(count)

    def set_queue_depth(self, depth: int) -> None:
        with self._lock:
            self._queue_depth = depth
            if self._prometheus_counters:
                self._prometheus_counters["queue_depth"].set(depth)

    def set_frame_age_ms(self, age_ms: float) -> None:
        with self._lock:
            self._frame_age_ms = max(0.0, float(age_ms))
            if self._prometheus_counters:
                self._prometheus_counters["frame_age_ms"].set(self._frame_age_ms)

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            elapsed = max(1e-6, time.monotonic() - self._start)
            return MetricsSnapshot(
                fps_decode=self._frames_decode / elapsed,
                fps_in=self._frames_in / elapsed,
                fps_infer=self._frames_infer / elapsed,
                sampled_frames=self._sampled_frames,
                dropped_frames=self._dropped_frames,
                queue_depth=self._queue_depth,
                frame_age_ms=self._frame_age_ms,
            )
