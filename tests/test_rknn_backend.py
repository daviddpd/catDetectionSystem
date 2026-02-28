from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from cds.detector.errors import BackendUnavailable
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

        detections = backend._decode_merged(
            merged=backend._merge_outputs([raw]),
            nc=2,
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

    def test_merge_outputs_falls_back_when_label_count_mismatches(self) -> None:
        backend = self._backend()
        backend._labels = [
            "opossum",
            "skunk",
            "raccoon",
            "cat-olive",
            "cat-bean",
            "cat-domino",
            "cat",
            "dog",
        ]
        raw = np.zeros((1, 9, 8400), dtype=np.float32)

        merged = backend._merge_outputs([raw])

        self.assertEqual(merged.shape, (1, 9, 8400))

    def test_infer_falls_back_and_locks_working_input_profile(self) -> None:
        backend = RKNNBackend()
        backend._runtime = object()
        backend._model_spec = ModelSpec(
            name="rknn-test",
            model_path="model.rknn",
            confidence=0.25,
            nms=0.5,
            imgsz=416,
        )
        backend._labels = ["cat", "dog"]
        backend._input_height = 416
        backend._input_width = 416
        backend._input_format = "nhwc"
        backend._normalize_input = True
        backend._swap_rb = True
        backend._input_dtype = "float32"
        backend._input_batched = True
        backend._input_candidates = [
            (416, 416, "nhwc", True, True, "float32", True),
            (640, 640, "nhwc", False, False, "uint8", False),
        ]

        calls: list[tuple[int, int, str, bool, bool, str, bool]] = []

        def fake_preprocess(
            frame: Any,
            *,
            input_h: int,
            input_w: int,
            input_format: str,
            normalize_input: bool,
            swap_rb: bool,
            input_dtype: str,
            input_batched: bool,
        ) -> tuple[np.ndarray, tuple[int, int]]:
            _ = frame
            calls.append(
                (
                    input_h,
                    input_w,
                    input_format,
                    normalize_input,
                    swap_rb,
                    input_dtype,
                    input_batched,
                )
            )
            dtype = np.uint8 if input_dtype == "uint8" else np.float32
            shape = (input_h, input_w, 3) if not input_batched else (1, input_h, input_w, 3)
            return np.zeros(shape, dtype=dtype), (input_h, input_w)

        def fake_run(input_tensor: np.ndarray, *, input_format: str, input_dtype: str) -> Any:
            _ = input_tensor
            _ = input_dtype
            if input_format != "nhwc":
                return None
            if calls[-1][0] == 416:
                return None
            return np.array([[[50.0], [60.0], [20.0], [10.0], [0.90], [0.10]]], dtype=np.float32)

        backend._preprocess = fake_preprocess  # type: ignore[method-assign]
        backend._run_rknn = fake_run  # type: ignore[method-assign]
        backend._decode_merged = (  # type: ignore[method-assign]
            lambda *, merged, nc, frame_shape, input_hw: ["ok"]
        )

        detections = backend.infer(np.zeros((100, 100, 3), dtype=np.uint8))

        self.assertEqual(len(detections), 1)
        self.assertEqual(
            calls[:2],
            [
                (416, 416, "nhwc", True, True, "float32", True),
                (640, 640, "nhwc", False, False, "uint8", False),
            ],
        )
        self.assertEqual(backend._input_candidates, [(640, 640, "nhwc", False, False, "uint8", False)])
        self.assertEqual(
            (
                backend._input_height,
                backend._input_width,
                backend._input_format,
                backend._normalize_input,
                backend._swap_rb,
                backend._input_dtype,
                backend._input_batched,
            ),
            (640, 640, "nhwc", False, False, "uint8", False),
        )

    def test_infer_raises_when_all_input_profiles_fail(self) -> None:
        backend = RKNNBackend()
        backend._runtime = object()
        backend._model_spec = ModelSpec(
            name="rknn-test",
            model_path="model.rknn",
            confidence=0.25,
            nms=0.5,
            imgsz=416,
        )
        backend._input_height = 416
        backend._input_width = 416
        backend._input_format = "nhwc"
        backend._normalize_input = True
        backend._swap_rb = True
        backend._input_dtype = "float32"
        backend._input_batched = True
        backend._input_candidates = [(416, 416, "nhwc", True, True, "float32", True)]

        backend._preprocess = (  # type: ignore[method-assign]
            lambda frame, *, input_h, input_w, input_format, normalize_input, swap_rb, input_dtype, input_batched: (
                np.zeros((1, input_h, input_w, 3), dtype=np.float32),
                (input_h, input_w),
            )
        )
        backend._run_rknn = lambda input_tensor, *, input_format, input_dtype: None  # type: ignore[method-assign]

        with self.assertRaises(BackendUnavailable):
            backend.infer(np.zeros((10, 10, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
