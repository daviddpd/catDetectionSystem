from __future__ import annotations

import unittest

from cds.detector.backends.ultralytics_backend import UltralyticsBackend
from cds.detector.errors import ModelLoadError
from cds.detector.models.model_spec import ModelSpec


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return []


class _KeyErrorModel:
    def predict(self, **kwargs):
        _ = kwargs
        raise KeyError("stride")


class UltralyticsBackendTests(unittest.TestCase):
    def test_coreml_artifact_omits_device_argument(self) -> None:
        backend = UltralyticsBackend(device="coreml")
        fake = _FakeModel()
        backend._model = fake
        backend._model_spec = ModelSpec(
            name="coreml-test",
            model_path="YOLOv3.mlmodel",
            confidence=0.3,
            nms=0.4,
            imgsz=416,
        )

        detections = backend.infer(object())

        self.assertEqual(detections, [])
        self.assertEqual(len(fake.calls), 1)
        call = fake.calls[0]
        self.assertNotIn("device", call)
        self.assertEqual(call["imgsz"], 416)

    def test_non_coreml_artifact_keeps_device_argument(self) -> None:
        backend = UltralyticsBackend(device="cpu")
        fake = _FakeModel()
        backend._model = fake
        backend._model_spec = ModelSpec(
            name="onnx-test",
            model_path="model.onnx",
            confidence=0.3,
            nms=0.4,
            imgsz=640,
        )

        detections = backend.infer(object())

        self.assertEqual(detections, [])
        self.assertEqual(len(fake.calls), 1)
        call = fake.calls[0]
        self.assertEqual(call.get("device"), "cpu")
        self.assertEqual(call["imgsz"], 640)

    def test_coreml_stride_keyerror_is_wrapped(self) -> None:
        backend = UltralyticsBackend(device="coreml")
        backend._model = _KeyErrorModel()
        backend._model_spec = ModelSpec(
            name="coreml-test",
            model_path="YOLOv3.mlmodel",
            confidence=0.3,
            nms=0.4,
            imgsz=416,
        )

        with self.assertRaises(ModelLoadError):
            backend.infer(object())


if __name__ == "__main__":
    unittest.main()
