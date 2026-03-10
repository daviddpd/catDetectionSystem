from __future__ import annotations

import unittest

from cds.utils.window_layout import compute_side_by_side_rects, compute_single_rect


class WindowLayoutTests(unittest.TestCase):
    def test_side_by_side_rects_are_non_overlapping(self) -> None:
        left, right = compute_side_by_side_rects(1920, 1080)
        self.assertLess(left.x + left.width, right.x + 1)
        self.assertEqual(left.y, right.y)
        self.assertEqual(left.height, right.height)
        self.assertEqual(left.width, right.width)

    def test_rects_fit_screen_with_margins(self) -> None:
        left, right = compute_side_by_side_rects(1366, 768, margin=20, gap=12)
        self.assertGreaterEqual(left.x, 20)
        self.assertGreaterEqual(left.y, 20)
        self.assertLessEqual(right.x + right.width, 1366 - 20)
        self.assertLessEqual(left.y + left.height, 768)

    def test_small_screen_uses_safe_minimums(self) -> None:
        left, right = compute_side_by_side_rects(640, 480)
        self.assertGreaterEqual(left.width, 480)
        self.assertGreaterEqual(left.height, 360)
        self.assertGreaterEqual(right.width, 480)
        self.assertGreaterEqual(right.height, 360)

    def test_right_scale_reduces_detections_window_size(self) -> None:
        left, right = compute_side_by_side_rects(1920, 1080, right_scale=0.25)
        self.assertEqual(left.width, 928)
        self.assertEqual(left.height, 928)
        self.assertEqual(right.width, 240)
        self.assertEqual(right.height, 232)

    def test_single_rect_matches_left_panel_defaults(self) -> None:
        rect = compute_single_rect(1920, 1080)
        self.assertEqual(rect.x, 24)
        self.assertEqual(rect.y, 24)
        self.assertEqual(rect.width, 928)
        self.assertEqual(rect.height, 928)


if __name__ == "__main__":
    unittest.main()
