from __future__ import annotations

from typing import Any

import numpy as np

from cds.detector.backends.base import DetectorBackend
from cds.detector.errors import BackendUnavailable, ModelLoadError
from cds.detector.models.model_spec import ModelSpec
from cds.types import Detection


class UltralyticsBackend(DetectorBackend):
    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._model: Any | None = None
        self._model_spec: ModelSpec | None = None
        self._labels: list[str] = []

    def load(self, model_spec: ModelSpec) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise BackendUnavailable(
                "Ultralytics backend requested but `ultralytics` is not installed."
            ) from exc

        if not model_spec.model_path:
            raise ModelLoadError(
                "Ultralytics backend requires model.path (for example .pt/.onnx/.engine/.mlpackage)."
            )

        self._model = YOLO(model_spec.model_path)
        self._model_spec = model_spec
        self._labels = model_spec.read_labels()

    def infer(self, frame: Any) -> list[Detection]:
        if self._model is None or self._model_spec is None:
            raise RuntimeError("Backend not loaded")

        results = self._model.predict(
            source=frame,
            device=self._device,
            conf=self._model_spec.confidence,
            iou=self._model_spec.nms,
            imgsz=self._model_spec.imgsz,
            verbose=False,
        )
        if not results:
            return []

        detections: list[Detection] = []
        result = results[0]
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        for box in boxes:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            if cls_id in names:
                label = str(names[cls_id])
            elif 0 <= cls_id < len(self._labels):
                label = self._labels[cls_id]
            else:
                label = str(cls_id)

            if self._model_spec.class_filter and label not in self._model_spec.class_filter:
                continue

            detections.append(
                Detection(
                    class_id=cls_id,
                    label=label,
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    backend=self.name(),
                )
            )

        return detections

    def warmup(self) -> None:
        if self._model is None:
            return
        warm_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        self.infer(warm_frame)

    def name(self) -> str:
        return "ultralytics"

    def device_info(self) -> str:
        return self._device
