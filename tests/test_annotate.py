from __future__ import annotations

import unittest

from cds.pipeline.annotate import _color_for_label


class AnnotateTests(unittest.TestCase):
    def test_label_color_is_deterministic(self) -> None:
        self.assertEqual(_color_for_label("cat"), _color_for_label("cat"))

    def test_different_labels_get_different_colors(self) -> None:
        self.assertNotEqual(_color_for_label("cat"), _color_for_label("dog"))


if __name__ == "__main__":
    unittest.main()
