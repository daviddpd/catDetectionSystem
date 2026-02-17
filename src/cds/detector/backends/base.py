from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from cds.detector.models.model_spec import ModelSpec
from cds.types import Detection


class DetectorBackend(ABC):
    @abstractmethod
    def load(self, model_spec: ModelSpec) -> None:
        """Load model artifacts and initialize backend."""

    @abstractmethod
    def infer(self, frame: Any) -> list[Detection]:
        """Run inference on a single frame."""

    @abstractmethod
    def warmup(self) -> None:
        """Run one-time warmup inference if supported."""

    @abstractmethod
    def name(self) -> str:
        """Return stable backend name for logging and metrics."""

    @abstractmethod
    def device_info(self) -> str:
        """Return selected device/accelerator detail string."""
