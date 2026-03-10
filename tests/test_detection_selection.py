from __future__ import annotations

import unittest

from cds.pipeline.detection_selection import select_primary_detection
from cds.types import Detection


def _det(confidence: float, x1: int, y1: int, x2: int, y2: int) -> Detection:
    det = Detection(
        class_id=0,
        label="cat",
        confidence=confidence,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )
    det.extra["area_pixels"] = det.width * det.height
    return det


class DetectionSelectionTests(unittest.TestCase):
    def test_empty_list_returns_empty(self) -> None:
        self.assertEqual(select_primary_detection([]), [])

    def test_single_item_returns_same_list(self) -> None:
        item = _det(0.5, 0, 0, 10, 10)
        items = [item]
        self.assertIs(select_primary_detection(items), items)

    def test_selects_highest_confidence(self) -> None:
        detections = [
            _det(0.5, 0, 0, 10, 10),
            _det(0.9, 0, 0, 8, 8),
            _det(0.7, 0, 0, 20, 20),
        ]
        selected = select_primary_detection(detections)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].confidence, 0.9)

    def test_breaks_confidence_tie_with_area(self) -> None:
        small = _det(0.8, 0, 0, 10, 10)
        large = _det(0.8, 0, 0, 20, 20)
        selected = select_primary_detection([small, large])
        self.assertEqual(len(selected), 1)
        self.assertIs(selected[0], large)


if __name__ == "__main__":
    unittest.main()
