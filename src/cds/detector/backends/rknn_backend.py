from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
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
        self._input_height = 640
        self._input_width = 640
        self._input_format = "nhwc"
        self._input_candidates: list[tuple[int, int, str]] = []
        self._non_max_suppression = None
        self._scale_boxes = None
        self._output_stats_logged = False
        self._confidence_hint_logged = False

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
        self._labels = self._resolve_labels(model_spec)
        self._imgsz = max(32, int(model_spec.imgsz))
        self._input_height = self._imgsz
        self._input_width = self._imgsz
        self._input_format = "nhwc"
        self._input_candidates = self._build_input_candidates(model_spec)

    def infer(self, frame: Any) -> list[Detection]:
        if self._runtime is None or self._model_spec is None:
            raise RuntimeError("Backend not loaded")

        errors: list[str] = []
        candidates = (
            [(self._input_height, self._input_width, self._input_format)]
            if len(self._input_candidates) <= 1
            else list(self._input_candidates)
        )
        for input_h, input_w, input_format in candidates:
            input_tensor, input_hw = self._preprocess(
                frame,
                input_h=input_h,
                input_w=input_w,
                input_format=input_format,
            )
            raw_outputs = self._run_rknn(input_tensor, input_format=input_format)
            if not self._raw_outputs_valid(raw_outputs):
                errors.append(
                    f"candidate={input_h}x{input_w}/{input_format} produced no usable outputs"
                )
                continue

            try:
                detections = self._decode_outputs(
                    raw_outputs=raw_outputs,
                    frame_shape=frame.shape[:2],
                    input_hw=input_hw,
                )
            except ModelLoadError as exc:
                errors.append(
                    f"candidate={input_h}x{input_w}/{input_format} decode failed: {exc}"
                )
                continue

            if (
                input_h != self._input_height
                or input_w != self._input_width
                or input_format != self._input_format
                or len(self._input_candidates) != 1
            ):
                self._input_height = input_h
                self._input_width = input_w
                self._input_format = input_format
                self._input_candidates = [(input_h, input_w, input_format)]
                self._logger.info(
                    "rknn input profile locked height=%d width=%d format=%s",
                    input_h,
                    input_w,
                    input_format,
                )
            return detections

        raise BackendUnavailable(
            "RKNN inference failed for all candidate input layouts. "
            + ("; ".join(errors) if errors else "No candidates evaluated.")
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

    def _letterbox(self, frame: Any, input_h: int, input_w: int) -> np.ndarray:
        if frame is None:
            raise RuntimeError("Invalid frame")

        shape = frame.shape[:2]  # (h, w)
        new_shape = (input_h, input_w)
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

    def _preprocess(
        self,
        frame: Any,
        *,
        input_h: int,
        input_w: int,
        input_format: str,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        image = self._letterbox(frame, input_h, input_w)
        image = image[..., ::-1]  # BGR -> RGB
        image = image.astype(np.float32) / 255.0
        if input_format == "nchw":
            image = np.transpose(image, (2, 0, 1))
            image = np.ascontiguousarray(image[None])
            return image, (int(image.shape[2]), int(image.shape[3]))
        if input_format != "nhwc":
            raise ModelLoadError(f"Unsupported RKNN input format: {input_format}")
        image = np.ascontiguousarray(image[None])
        return image, (int(image.shape[1]), int(image.shape[2]))

    def _run_rknn(self, input_tensor: np.ndarray, *, input_format: str) -> Any:
        assert self._runtime is not None
        try:
            return self._runtime.inference(
                inputs=[input_tensor],
                data_format=[input_format],
            )
        except TypeError:
            return self._runtime.inference(inputs=[input_tensor])

    @staticmethod
    def _raw_outputs_valid(raw_outputs: Any) -> bool:
        if raw_outputs is None:
            return False
        if isinstance(raw_outputs, np.ndarray):
            return raw_outputs.ndim > 0 and raw_outputs.size > 0
        if isinstance(raw_outputs, (list, tuple)):
            valid = False
            for item in raw_outputs:
                if item is None:
                    continue
                arr = np.asarray(item)
                if arr.ndim == 0 or arr.size == 0:
                    continue
                valid = True
            return valid
        arr = np.asarray(raw_outputs)
        return arr.ndim > 0 and arr.size > 0

    def _build_input_candidates(self, model_spec: ModelSpec) -> list[tuple[int, int, str]]:
        candidates: list[tuple[int, int, str]] = []

        def _add(h: int, w: int, fmt: str) -> None:
            item = (max(32, int(h)), max(32, int(w)), fmt.lower())
            if item not in candidates:
                candidates.append(item)

        sidecar_hw = self._infer_input_hw_from_sidecar_onnx(model_spec)
        if sidecar_hw is not None:
            h, w = sidecar_hw
            _add(h, w, "nhwc")
            _add(h, w, "nchw")

        _add(self._imgsz, self._imgsz, "nhwc")
        _add(self._imgsz, self._imgsz, "nchw")

        if self._imgsz != 640:
            _add(640, 640, "nhwc")
            _add(640, 640, "nchw")

        return candidates

    def _resolve_labels(self, model_spec: ModelSpec) -> list[str]:
        file_labels = model_spec.read_labels()
        sidecar_labels = self._infer_labels_from_sidecar_onnx(model_spec)
        if sidecar_labels:
            if file_labels and file_labels != sidecar_labels:
                self._logger.info(
                    "rknn labels overridden from paired onnx metadata file_labels=%d onnx_labels=%d",
                    len(file_labels),
                    len(sidecar_labels),
                )
            return sidecar_labels
        return file_labels

    def _infer_labels_from_sidecar_onnx(self, model_spec: ModelSpec) -> list[str]:
        onnx_path = self._find_sidecar_onnx_path(model_spec)
        if onnx_path is None:
            return []
        try:
            import onnx
        except Exception:
            return []

        try:
            model = onnx.load(str(onnx_path))
        except Exception:
            return []

        for prop in model.metadata_props:
            if prop.key != "names":
                continue
            try:
                parsed = ast.literal_eval(prop.value)
            except Exception:
                return []
            if isinstance(parsed, dict):
                items: list[tuple[int, str]] = []
                for key, value in parsed.items():
                    try:
                        idx = int(key)
                    except Exception:
                        continue
                    items.append((idx, str(value)))
                items.sort(key=lambda item: item[0])
                return [value for _, value in items]
            if isinstance(parsed, (list, tuple)):
                return [str(item) for item in parsed]
            return []
        return []

    def _infer_input_hw_from_sidecar_onnx(self, model_spec: ModelSpec) -> tuple[int, int] | None:
        onnx_path = self._find_sidecar_onnx_path(model_spec)
        if onnx_path is None:
            return None
        try:
            import onnx
        except Exception:
            return None

        try:
            model = onnx.load(str(onnx_path))
            if not model.graph.input:
                return None
            tensor_type = model.graph.input[0].type.tensor_type
            dims: list[int] = []
            for dim in tensor_type.shape.dim:
                if dim.dim_value:
                    dims.append(int(dim.dim_value))
                else:
                    dims.append(0)
            if len(dims) != 4:
                return None
            if dims[1] in {1, 3, 4} and dims[2] > 0 and dims[3] > 0:
                return dims[2], dims[3]
            if dims[3] in {1, 3, 4} and dims[1] > 0 and dims[2] > 0:
                return dims[1], dims[2]
        except Exception:
            return None
        return None

    def _find_sidecar_onnx_path(self, model_spec: ModelSpec) -> Path | None:
        if not model_spec.model_path:
            return None

        model_path = Path(model_spec.model_path).expanduser().resolve()
        candidates: list[Path] = []

        report_path = model_path.parent.parent / "reports" / "export_report.json"
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                for result in report.get("results", []):
                    if result.get("target") != "onnx":
                        continue
                    artifact = result.get("artifact")
                    if artifact:
                        candidates.append(Path(str(artifact)).expanduser())
            except Exception:
                pass

        exports_dir = model_path.parent.parent / "exports"
        if exports_dir.exists():
            preferred = [
                exports_dir / "best.onnx",
                exports_dir / "model.onnx",
            ]
            candidates.extend(preferred)
            candidates.extend(sorted(exports_dir.glob("*.onnx")))

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None

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

        try:
            batches = [self._coerce_output_tensor(item, expected_attrs) for item in outputs]
        except ModelLoadError as exc:
            if expected_attrs is None or "attribute dimension mismatch" not in str(exc):
                raise
            batches = [self._coerce_output_tensor(item, None) for item in outputs]
            inferred_attrs = batches[0].shape[1]
            inferred_classes = max(0, inferred_attrs - 4)
            self._logger.warning(
                "rknn output attribute count mismatches labels expected_attrs=%d inferred_attrs=%d labels=%d inferred_classes=%d",
                expected_attrs,
                inferred_attrs,
                len(self._labels),
                inferred_classes,
            )
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
        self._log_output_stats_once(merged, nc)

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
            self._log_confidence_hint(merged, nc)
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

    def _log_output_stats_once(self, merged: np.ndarray, nc: int) -> None:
        if self._output_stats_logged:
            return
        self._output_stats_logged = True
        if nc <= 0 or merged.shape[1] < 5:
            self._logger.info("rknn output stats shape=%s nc=%d", tuple(merged.shape), nc)
            return

        cls = merged[:, 4 : 4 + nc, :]
        flat = cls.reshape(-1)
        top = sorted((float(v) for v in flat), reverse=True)[:5]
        self._logger.info(
            "rknn output stats shape=%s nc=%d cls_min=%.4f cls_max=%.4f cls_top5=%s conf=%.3f nms=%.3f",
            tuple(merged.shape),
            nc,
            float(cls.min()),
            float(cls.max()),
            [round(v, 4) for v in top],
            self._model_spec.confidence if self._model_spec is not None else -1.0,
            self._model_spec.nms if self._model_spec is not None else -1.0,
        )

    def _log_confidence_hint(self, merged: np.ndarray, nc: int) -> None:
        if self._confidence_hint_logged or self._model_spec is None or nc <= 0:
            return
        cls = merged[:, 4 : 4 + nc, :]
        max_conf = float(cls.max())
        if max_conf < self._model_spec.confidence:
            self._confidence_hint_logged = True
            self._logger.warning(
                "rknn max class score %.4f is below confidence threshold %.4f; try lowering --confidence",
                max_conf,
                self._model_spec.confidence,
            )

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
