from __future__ import annotations

import unittest

import numpy as np

from cds.detector.backends.rknn_backend import RKNNBackend
from cds.detector.models.model_spec import ModelSpec


class RKNNBackendTests(unittest.TestCase):
    def _backend(self) -> RKNNBackend:
        backend = RKNNBackend()
        non_max_suppression, scale_boxes = backend._load_postprocess_helpers()
        backend._non_max_suppression = non_max_suppression
        backend._scale_boxes = scale_boxes
        backend._model_spec = ModelSpec(
            name="rknn-test",
            model_path="model.rknn",
            confidence=0.25,
            nms=0.5,
            imgsz=100,
        )
        backend._labels = ["cat", "dog"]
        backend._imgsz = 100
        return backend

    def test_decode_outputs_produces_detection(self) -> None:
        backend = self._backend()
        raw = np.array(
            [
                [
                    [50.0],  # x center
                    [60.0],  # y center
                    [20.0],  # width
                    [10.0],  # height
                    [0.90],  # cat score
                    [0.10],  # dog score
                ]
            ],
            dtype=np.float32,
        )

        detections = backend._decode_outputs(
            raw_outputs=[raw],
            frame_shape=(100, 100),
            input_hw=(100, 100),
        )

        self.assertEqual(len(detections), 1)
        det = detections[0]
        self.assertEqual(det.class_id, 0)
        self.assertEqual(det.label, "cat")
        self.assertAlmostEqual(det.confidence, 0.90, places=5)
        self.assertEqual((det.x1, det.y1, det.x2, det.y2), (40, 55, 60, 65))

    def test_merge_outputs_concatenates_split_heads(self) -> None:
        backend = self._backend()
        raw_a = np.array(
            [[[50.0], [60.0], [20.0], [10.0], [0.90], [0.10]]],
            dtype=np.float32,
        )
        raw_b = np.array(
            [[[20.0], [30.0], [8.0], [12.0], [0.20], [0.80]]],
            dtype=np.float32,
        )

        merged = backend._merge_outputs([raw_a, raw_b])

        self.assertEqual(merged.shape, (1, 6, 2))
        self.assertAlmostEqual(float(merged[0, 4, 0]), 0.90, places=5)
        self.assertAlmostEqual(float(merged[0, 5, 1]), 0.80, places=5)


if __name__ == "__main__":
    unittest.main()
