from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cds.training.dataset.validate import validate_yolo_dataset


class DatasetValidateTests(unittest.TestCase):
    def test_reports_out_of_bounds_and_unknown_class(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "images" / "train").mkdir(parents=True, exist_ok=True)
            (root / "labels" / "train").mkdir(parents=True, exist_ok=True)

            image_path = root / "images" / "train" / "a.jpg"
            label_path = root / "labels" / "train" / "a.txt"
            image_path.write_bytes(b"jpg")
            label_path.write_text("99 1.5 0.5 1.2 0.2\n", encoding="utf-8")

            report = validate_yolo_dataset(root, ["cat", "dog"])

            self.assertEqual(report["status"], "fail")
            self.assertGreaterEqual(report["counts"]["unknown_class_ids"], 1)
            self.assertGreaterEqual(report["counts"]["out_of_bounds_boxes"], 1)

    def test_empty_dataset_fails_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report = validate_yolo_dataset(root, ["cat", "dog"])
            self.assertEqual(report["status"], "fail")
            self.assertTrue(any(issue["code"] == "no_images" for issue in report["issues"]))


if __name__ == "__main__":
    unittest.main()
