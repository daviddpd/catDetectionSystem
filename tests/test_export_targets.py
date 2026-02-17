from __future__ import annotations

import unittest

from cds.training.export import available_export_targets


class ExportTargetsTests(unittest.TestCase):
    def test_export_targets_shape(self) -> None:
        targets = available_export_targets()
        for name in ["pytorch", "onnx", "coreml", "tensorrt", "rknn"]:
            self.assertIn(name, targets)
            self.assertIn("supported", targets[name])
            self.assertIn("reason", targets[name])


if __name__ == "__main__":
    unittest.main()
