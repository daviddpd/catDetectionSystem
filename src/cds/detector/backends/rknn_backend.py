from __future__ import annotations

import logging
from typing import Any

import numpy as np

from cds.detector.backends.base import DetectorBackend
from cds.detector.errors import BackendUnavailable, ModelLoadError
from cds.detector.models.model_spec import ModelSpec
from cds.types import Detection


class RKNNBackend(DetectorBackend):
    """RKNN runtime backend for Ultralytics-exported detect models."""

    def __init__(self) -> None:
        self._logger = logging.getLogger("cds.detector.rknn")
        self._runtime: Any | None = None
        self._model_spec: ModelSpec | None = None
        self._labels: list[str] = []
        self._imgsz = 640
        self._non_max_suppression = None
        self._scale_boxes = None

    def load(self, model_spec: ModelSpec) -> None:
        try:
            from rknnlite.api import RKNNLite
        except ImportError as exc:
            raise BackendUnavailable(
                "RKNN backend requested but rknnlite is not installed."
            ) from exc

        self._non_max_suppression, self._scale_boxes = self._load_postprocess_helpers()

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
        self._labels = model_spec.read_labels()
        self._imgsz = max(32, int(model_spec.imgsz))

    def infer(self, frame: Any) -> list[Detection]:
        if self._runtime is None or self._model_spec is None:
            raise RuntimeError("Backend not loaded")

        input_tensor, input_hw = self._preprocess(frame)
        raw_outputs = self._run_rknn(input_tensor)
        return self._decode_outputs(
            raw_outputs=raw_outputs,
            frame_shape=frame.shape[:2],
            input_hw=input_hw,
        )

    def _load_postprocess_helpers(self):
        try:
            from ultralytics.utils.nms import non_max_suppression
            from ultralytics.utils.ops import scale_boxes
        except Exception as exc:
            raise BackendUnavailable(
                "RKNN backend requires ultralytics post-processing utilities "
                "(ultralytics + torch)."
            ) from exc
        return non_max_suppression, scale_boxes

    def _letterbox(self, frame: Any, size: int) -> np.ndarray:
        if frame is None:
            raise RuntimeError("Invalid frame")

        shape = frame.shape[:2]  # (h, w)
        new_shape = (size, size)
        gain = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (
            int(round(shape[1] * gain)),
            int(round(shape[0] * gain)),
        )
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        dw /= 2.0
        dh /= 2.0

        if shape[::-1] != new_unpad:
            import cv2

            frame = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR)

        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))

        import cv2

        return cv2.copyMakeBorder(
            frame,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

    def _preprocess(self, frame: Any) -> tuple[np.ndarray, tuple[int, int]]:
        image = self._letterbox(frame, self._imgsz)
        image = image[..., ::-1]  # BGR -> RGB
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.ascontiguousarray(image[None])
        return image, (int(image.shape[2]), int(image.shape[3]))

    def _run_rknn(self, input_tensor: np.ndarray) -> Any:
        assert self._runtime is not None
        try:
            return self._runtime.inference(
                inputs=[input_tensor],
                data_format=["nchw"],
            )
        except TypeError:
            return self._runtime.inference(inputs=[input_tensor])

    def _coerce_output_tensor(self, output: Any, expected_attrs: int | None) -> np.ndarray:
        arr = np.asarray(output)
        if arr.size == 0:
            raise ModelLoadError("RKNN backend received empty output tensor")

        arr = arr.astype(np.float32, copy=False)

        if arr.ndim == 1:
            raise ModelLoadError(
                f"Unsupported RKNN output rank=1 shape={tuple(arr.shape)}"
            )

        if arr.ndim == 2:
            if expected_attrs is not None and arr.shape[1] == expected_attrs:
                arr = arr.T
            elif expected_attrs is None and arr.shape[0] > arr.shape[1]:
                arr = arr.T
            return arr[None]

        if arr.ndim == 3:
            if arr.shape[0] == 1:
                batch = arr
            elif arr.shape[2] == 1:
                batch = np.transpose(arr, (2, 0, 1))
            elif arr.shape[1] == 1:
                batch = np.transpose(arr, (1, 0, 2))
            else:
                raise ModelLoadError(
                    f"Unsupported RKNN output shape={tuple(arr.shape)}"
                )
        elif arr.ndim == 4 and arr.shape[0] == 1:
            if expected_attrs is not None and arr.shape[1] == expected_attrs:
                batch = arr.reshape(1, arr.shape[1], -1)
            elif expected_attrs is not None and arr.shape[-1] == expected_attrs:
                batch = arr.reshape(-1, arr.shape[-1]).T[None]
            elif arr.shape[1] <= arr.shape[-1]:
                batch = arr.reshape(1, arr.shape[1], -1)
            else:
                batch = arr.reshape(1, -1, arr.shape[-1]).transpose(0, 2, 1)
        else:
            raise ModelLoadError(
                f"Unsupported RKNN output rank={arr.ndim} shape={tuple(arr.shape)}"
            )

        if expected_attrs is not None and batch.shape[1] != expected_attrs:
            if batch.shape[2] == expected_attrs:
                batch = batch.transpose(0, 2, 1)
            else:
                raise ModelLoadError(
                    "RKNN output tensor attribute dimension mismatch: "
                    f"shape={tuple(batch.shape)} expected_attrs={expected_attrs}"
                )

        return batch

    def _merge_outputs(self, raw_outputs: Any) -> np.ndarray:
        if isinstance(raw_outputs, np.ndarray):
            outputs = [raw_outputs]
        elif isinstance(raw_outputs, (list, tuple)):
            outputs = [item for item in raw_outputs if item is not None]
        else:
            outputs = [raw_outputs]

        if not outputs:
            raise ModelLoadError("RKNN backend returned no output tensors")

        expected_attrs = None
        if self._labels:
            expected_attrs = 4 + len(self._labels)

        batches = [self._coerce_output_tensor(item, expected_attrs) for item in outputs]
        merged = batches[0]
        for batch in batches[1:]:
            if batch.shape[0] != merged.shape[0]:
                raise ModelLoadError(
                    "RKNN output batch mismatch: "
                    f"{tuple(merged.shape)} vs {tuple(batch.shape)}"
                )
            if batch.shape[1] != merged.shape[1]:
                raise ModelLoadError(
                    "RKNN output channel mismatch: "
                    f"{tuple(merged.shape)} vs {tuple(batch.shape)}"
                )
            merged = np.concatenate((merged, batch), axis=2)
        return merged

    def _decode_outputs(
        self,
        *,
        raw_outputs: Any,
        frame_shape: tuple[int, int],
        input_hw: tuple[int, int],
    ) -> list[Detection]:
        if self._model_spec is None:
            raise RuntimeError("Backend not loaded")
        if self._non_max_suppression is None or self._scale_boxes is None:
            raise RuntimeError("RKNN postprocess helpers not loaded")

        merged = self._merge_outputs(raw_outputs)
        nc = max(1, merged.shape[1] - 4)

        try:
            import torch
        except Exception as exc:
            raise BackendUnavailable("RKNN backend requires torch for post-processing") from exc

        prediction = torch.from_numpy(merged)
        preds = self._non_max_suppression(
            prediction,
            conf_thres=self._model_spec.confidence,
            iou_thres=self._model_spec.nms,
            nc=nc,
        )
        if not preds or preds[0].numel() == 0:
            return []

        det_tensor = preds[0].clone()
        det_tensor[:, :4] = self._scale_boxes(input_hw, det_tensor[:, :4], frame_shape)

        detections: list[Detection] = []
        for row in det_tensor:
            x1, y1, x2, y2, confidence, cls = row[:6]
            cls_id = int(cls.item())
            label = self._label_for_class(cls_id)
            if self._model_spec.class_filter and label not in self._model_spec.class_filter:
                continue

            detections.append(
                Detection(
                    class_id=cls_id,
                    label=label,
                    confidence=float(confidence.item()),
                    x1=int(round(float(x1.item()))),
                    y1=int(round(float(y1.item()))),
                    x2=int(round(float(x2.item()))),
                    y2=int(round(float(y2.item()))),
                    backend=self.name(),
                )
            )
        return detections

    def _label_for_class(self, cls_id: int) -> str:
        if 0 <= cls_id < len(self._labels):
            return self._labels[cls_id]
        return str(cls_id)

    def warmup(self) -> None:
        return

    def name(self) -> str:
        return "rknn"

    def device_info(self) -> str:
        return "npu"
