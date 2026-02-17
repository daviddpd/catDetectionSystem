from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Detection:
    """Normalized detection output used across backends."""

    class_id: int
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    backend: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)


@dataclass
class FramePacket:
    """Frame packet moved through ingest/inference/output stages."""

    frame_id: int
    frame: Any
    source: str
    timestamp: datetime


@dataclass
class RuntimeStats:
    fps_in: float
    fps_infer: float
    dropped_frames: int
    queue_depth: int
    backend: str
    decoder: str
