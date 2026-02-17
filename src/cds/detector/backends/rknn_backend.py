from __future__ import annotations

from typing import Any

from cds.detector.backends.base import DetectorBackend
from cds.detector.errors import BackendUnavailable, ModelLoadError
from cds.detector.models.model_spec import ModelSpec
from cds.types import Detection


class RKNNBackend(DetectorBackend):
    """Rockchip RKNN integration point.

    This backend is intentionally conservative for Stage 1: it provides runtime wiring
    and accelerator selection hooks while preserving CPU fallback behavior when RKNN
    runtime or post-processing pipelines are unavailable.
    """

    def __init__(self) -> None:
        self._runtime: Any | None = None
        self._model_spec: ModelSpec | None = None

    def load(self, model_spec: ModelSpec) -> None:
        try:
            from rknnlite.api import RKNNLite
        except ImportError as exc:
            raise BackendUnavailable(
                "RKNN backend requested but rknnlite is not installed."
            ) from exc

        if not model_spec.model_path:
            raise ModelLoadError("RKNN backend requires model.path pointing to .rknn file")

        if not str(model_spec.model_path).lower().endswith(".rknn"):
            raise ModelLoadError("RKNN backend expects .rknn model artifact")

        runtime = RKNNLite()
        if runtime.load_rknn(model_spec.model_path) != 0:
            raise ModelLoadError(f"Failed to load RKNN model: {model_spec.model_path}")
        if runtime.init_runtime() != 0:
            raise BackendUnavailable("Failed to initialize RKNN runtime")

        self._runtime = runtime
        self._model_spec = model_spec

    def infer(self, frame: Any) -> list[Detection]:
        if self._runtime is None:
            raise RuntimeError("Backend not loaded")

        # Stage 1 keeps RKNN as integration point with safe fallback behavior.
        # Post-processing is model-specific, so return empty detections unless a
        # dedicated decoder is added in a future stage.
        _ = frame
        return []

    def warmup(self) -> None:
        return

    def name(self) -> str:
        return "rknn"

    def device_info(self) -> str:
        return "npu"
