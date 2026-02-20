from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from cds.monitoring.metrics import RuntimeMetrics


@dataclass
class RuntimeIdentity:
    backend: str
    decoder: str


class PeriodicStatsLogger:
    def __init__(
        self,
        metrics: RuntimeMetrics,
        identity: RuntimeIdentity,
        interval_seconds: float = 5.0,
    ) -> None:
        self._metrics = metrics
        self._identity = identity
        self._interval_seconds = max(0.5, interval_seconds)
        self._next_emit = time.monotonic() + self._interval_seconds
        self._logger = logging.getLogger("cds.stats")

    def maybe_emit(self) -> None:
        now = time.monotonic()
        if now < self._next_emit:
            return

        snapshot = self._metrics.snapshot()
        self._logger.info(
            "stats fps_decode=%.2f fps_in=%.2f fps_infer=%.2f sampled_frames=%d dropped_frames=%d queue_depth=%d frame_age_ms=%.1f backend=%s decoder=%s",
            snapshot.fps_decode,
            snapshot.fps_in,
            snapshot.fps_infer,
            snapshot.sampled_frames,
            snapshot.dropped_frames,
            snapshot.queue_depth,
            snapshot.frame_age_ms,
            self._identity.backend,
            self._identity.decoder,
        )

        self._next_emit = now + self._interval_seconds
