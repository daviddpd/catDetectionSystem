from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cds.detector.backends.base import DetectorBackend
from cds.detector.errors import ModelLoadError
from cds.detector.models.model_spec import ModelSpec
from cds.types import Detection


class OpenCVDarknetBackend(DetectorBackend):
    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._net: Any | None = None
        self._model_spec: ModelSpec | None = None
        self._labels: list[str] = []
        self._output_layers: list[str] = []

    def load(self, model_spec: ModelSpec) -> None:
        if not model_spec.cfg_path or not model_spec.weights_path:
            raise ModelLoadError(
                "OpenCV Darknet backend requires model.cfg_path and model.weights_path"
            )

        cfg_path = Path(model_spec.cfg_path)
        weights_path = Path(model_spec.weights_path)
        if not cfg_path.exists():
            raise ModelLoadError(f"Darknet cfg missing: {cfg_path}")
        if not weights_path.exists():
            raise ModelLoadError(f"Darknet weights missing: {weights_path}")

        self._net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
        if self._device == "cuda":
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        elif self._device == "opencl":
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
        else:
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        layer_names = self._net.getLayerNames()
        unconnected = self._net.getUnconnectedOutLayers()
        if len(unconnected) > 0 and isinstance(unconnected[0], (list, tuple, np.ndarray)):
            self._output_layers = [layer_names[idx[0] - 1] for idx in unconnected]
        else:
            self._output_layers = [layer_names[idx - 1] for idx in unconnected]

        self._model_spec = model_spec
        self._labels = model_spec.read_labels()

    def infer(self, frame: Any) -> list[Detection]:
        if self._net is None or self._model_spec is None:
            raise RuntimeError("Backend not loaded")

        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame,
            1 / 255.0,
            (self._model_spec.imgsz, self._model_spec.imgsz),
            swapRB=True,
            crop=False,
        )
        self._net.setInput(blob)
        outputs = self._net.forward(self._output_layers)

        class_ids: list[int] = []
        confidences: list[float] = []
        boxes: list[list[int]] = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence < self._model_spec.confidence:
                    continue

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                box_w = int(detection[2] * width)
                box_h = int(detection[3] * height)
                x = int(center_x - box_w / 2)
                y = int(center_y - box_h / 2)

                class_ids.append(class_id)
                confidences.append(confidence)
                boxes.append([x, y, box_w, box_h])

        idxs = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self._model_spec.confidence,
            self._model_spec.nms,
        )

        detections: list[Detection] = []
        if len(idxs) == 0:
            return detections

        for idx in np.array(idxs).flatten().tolist():
            x, y, w, h = boxes[idx]
            cls_id = class_ids[idx]
            label = self._labels[cls_id] if 0 <= cls_id < len(self._labels) else str(cls_id)

            if self._model_spec.class_filter and label not in self._model_spec.class_filter:
                continue

            detections.append(
                Detection(
                    class_id=cls_id,
                    label=label,
                    confidence=confidences[idx],
                    x1=x,
                    y1=y,
                    x2=x + w,
                    y2=y + h,
                    backend=self.name(),
                )
            )

        return detections

    def warmup(self) -> None:
        if self._net is None:
            return
        warm_frame = np.zeros((self._model_spec.imgsz, self._model_spec.imgsz, 3), dtype=np.uint8)
        self.infer(warm_frame)

    def name(self) -> str:
        return "opencv-darknet"

    def device_info(self) -> str:
        return self._device
