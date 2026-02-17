from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

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

        self._labels = model_spec.read_labels()
        model_path = self._prepare_model_path(model_spec)

        self._model = YOLO(model_path)
        self._model_spec = model_spec

    def infer(self, frame: Any) -> list[Detection]:
        if self._model is None or self._model_spec is None:
            raise RuntimeError("Backend not loaded")

        try:
            results = self._model.predict(
                **self._predict_kwargs(frame),
            )
        except KeyError as exc:
            model_path = (self._model_spec.model_path if self._model_spec else "") or ""
            if str(exc) == "'stride'" and model_path.lower().endswith((".mlmodel", ".mlpackage")):
                raise ModelLoadError(
                    "CoreML model is missing Ultralytics metadata (for example: stride/task/imgsz/names). "
                    "Use an Ultralytics-exported CoreML model, or run this model through CDS CoreML metadata patching."
                ) from exc
            raise
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

    def _predict_kwargs(self, frame: Any) -> dict[str, Any]:
        if self._model_spec is None:
            raise RuntimeError("Backend not loaded")

        kwargs: dict[str, Any] = {
            "source": frame,
            "conf": self._model_spec.confidence,
            "iou": self._model_spec.nms,
            "imgsz": self._model_spec.imgsz,
            "verbose": False,
        }
        model_path = (self._model_spec.model_path or "").lower()
        is_coreml_artifact = model_path.endswith((".mlmodel", ".mlpackage"))

        # Ultralytics CoreML artifacts do not accept `device="coreml"` in predict().
        # Passing no device lets Ultralytics dispatch correctly for exported CoreML models.
        if not (self._device == "coreml" and is_coreml_artifact):
            kwargs["device"] = self._device

        return kwargs

    def _prepare_model_path(self, model_spec: ModelSpec) -> str:
        model_path = str(model_spec.model_path)
        model_lower = model_path.lower()
        if self._device != "coreml":
            return model_path
        if not model_lower.endswith(".mlmodel"):
            return model_path

        try:
            import coremltools as ct
            from coremltools.models.utils import save_spec
        except Exception:
            return model_path

        mlmodel = ct.models.MLModel(model_path, skip_model_load=True)
        metadata = dict(getattr(mlmodel, "user_defined_metadata", {}) or {})
        required = {"stride", "task", "batch", "imgsz", "names"}
        if required.issubset(metadata):
            return model_path

        spec = mlmodel.get_spec()
        imgsz_h, imgsz_w = self._infer_coreml_input_size(spec, fallback=model_spec.imgsz)
        names = self._infer_coreml_names(spec, preferred=self._labels)
        names_map = {idx: name for idx, name in enumerate(names)}

        patched = dict(metadata)
        patched.setdefault("stride", "32")
        patched.setdefault("task", "detect")
        patched.setdefault("batch", "1")
        patched.setdefault("imgsz", str([imgsz_h, imgsz_w]))
        patched.setdefault("names", str(names_map))
        patched.setdefault("channels", "3")

        for key, value in patched.items():
            spec.description.metadata.userDefined[str(key)] = str(value)

        cache_dir = Path(tempfile.gettempdir()) / "cds-coreml-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        patched_path = cache_dir / f"{Path(model_path).stem}.cds_patched.mlmodel"
        save_spec(spec, str(patched_path))
        return str(patched_path)

    @staticmethod
    def _infer_coreml_input_size(spec: Any, fallback: int = 640) -> tuple[int, int]:
        try:
            for feature in spec.description.input:
                if feature.type.HasField("imageType"):
                    height = int(feature.type.imageType.height)
                    width = int(feature.type.imageType.width)
                    if height > 0 and width > 0:
                        return height, width
        except Exception:
            pass
        fallback_size = max(32, int(fallback))
        return fallback_size, fallback_size

    @staticmethod
    def _infer_coreml_names(spec: Any, preferred: list[str]) -> list[str]:
        if preferred:
            return preferred

        try:
            model_type = spec.WhichOneof("Type")
            if model_type == "pipeline":
                for child in spec.pipeline.models:
                    if child.WhichOneof("Type") != "nonMaximumSuppression":
                        continue
                    nms = child.nonMaximumSuppression
                    label_kind = nms.WhichOneof("ClassLabels")
                    if label_kind == "stringClassLabels":
                        labels = [str(v) for v in nms.stringClassLabels.vector]
                        if labels:
                            return labels
                    if label_kind == "int64ClassLabels":
                        labels = [str(v) for v in nms.int64ClassLabels.vector]
                        if labels:
                            return labels
        except Exception:
            pass

        # Fallback when class labels are unavailable from model metadata/spec.
        return [f"class{i}" for i in range(80)]

    def warmup(self) -> None:
        if self._model is None or self._model_spec is None:
            return
        try:
            import numpy as np
        except Exception:
            return
        warm_size = max(32, int(self._model_spec.imgsz))
        warm_frame = np.zeros((warm_size, warm_size, 3), dtype=np.uint8)
        self.infer(warm_frame)

    def name(self) -> str:
        return "ultralytics"

    def device_info(self) -> str:
        return self._device
