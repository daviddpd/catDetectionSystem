from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class OutputSink(ABC):
    @abstractmethod
    def open(self) -> None:
        """Allocate sink resources."""

    @abstractmethod
    def write(self, frame: Any, metadata: dict | None = None) -> bool:
        """Write frame payload. Return False if runtime should stop."""

    @abstractmethod
    def close(self) -> None:
        """Release sink resources."""

    @abstractmethod
    def name(self) -> str:
        """Stable sink name for logging."""
