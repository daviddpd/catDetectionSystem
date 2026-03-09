from __future__ import annotations

import unittest

from cds.training.train import _resolve_training_model_source


class TrainModelSourceTests(unittest.TestCase):
    def test_finetune_uses_base_model(self) -> None:
        source, from_scratch = _resolve_training_model_source(
            {"base": "yolov8s.pt", "from_scratch": False}
        )
        self.assertEqual(source, "yolov8s.pt")
        self.assertFalse(from_scratch)

    def test_from_scratch_uses_arch_override(self) -> None:
        source, from_scratch = _resolve_training_model_source(
            {"base": "yolov8s.pt", "from_scratch": True, "arch": "custom.yaml"}
        )
        self.assertEqual(source, "custom.yaml")
        self.assertTrue(from_scratch)

    def test_from_scratch_derives_yaml_from_pt_base(self) -> None:
        source, from_scratch = _resolve_training_model_source(
            {"base": "yolov8s.pt", "from_scratch": True}
        )
        self.assertEqual(source, "yolov8s.yaml")
        self.assertTrue(from_scratch)

    def test_from_scratch_accepts_yaml_base(self) -> None:
        source, from_scratch = _resolve_training_model_source(
            {"base": "yolov8s.yaml", "from_scratch": True}
        )
        self.assertEqual(source, "yolov8s.yaml")
        self.assertTrue(from_scratch)

    def test_from_scratch_requires_resolvable_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "from-scratch mode requires"):
            _resolve_training_model_source({"base": "weights.bin", "from_scratch": True})


if __name__ == "__main__":
    unittest.main()
