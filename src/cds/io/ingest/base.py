from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from cds.types import FramePacket


class VideoIngest(ABC):
    @abstractmethod
    def open(self, uri: str, options: dict[str, Any] | None = None) -> None:
        """Open input source."""

    @abstractmethod
    def read_latest(self) -> FramePacket | None:
        """Read the newest available frame in a non-blocking-friendly way."""

    @abstractmethod
    def close(self) -> None:
        """Release source resources."""

    @abstractmethod
    def name(self) -> str:
        """Stable backend name."""

    def source_mode(self) -> str:
        """Source mode: live-stream, video-file, image-file, or directory."""
        return "live-stream"

    def nominal_fps(self) -> float | None:
        """Best-effort source FPS if available."""
        return None
