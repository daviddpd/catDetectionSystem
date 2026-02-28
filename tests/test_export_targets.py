from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cds.training.export import _write_rknn_conversion_bundle, available_export_targets


class ExportTargetsTests(unittest.TestCase):
    def test_export_targets_shape(self) -> None:
        targets = available_export_targets()
        for name in ["pytorch", "onnx", "coreml", "tensorrt", "rknn"]:
            self.assertIn(name, targets)
            self.assertIn("supported", targets[name])
            self.assertIn("reason", targets[name])

    def test_rknn_bundle_scripts_include_setuptools_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = _write_rknn_conversion_bundle(
                onnx_path=Path("/tmp/model.onnx"),
                output_root=Path(tmpdir),
            )
            toolkit2_text = Path(bundle["toolkit2_script"]).read_text(encoding="utf-8")
            legacy_text = Path(bundle["legacy_script"]).read_text(encoding="utf-8")

            self.assertIn("from rknn.api import RKNN", toolkit2_text)
            self.assertIn("python3 -m pip install 'setuptools<82'", toolkit2_text)
            self.assertIn('Path(__file__).with_name("calibration.txt")', toolkit2_text)
            self.assertIn("DO_QUANTIZATION = True", toolkit2_text)
            self.assertIn("MEAN_VALUES = [[0, 0, 0]]", toolkit2_text)
            self.assertIn("STD_VALUES = [[255, 255, 255]]", toolkit2_text)
            self.assertIn("mean_values=MEAN_VALUES", toolkit2_text)
            self.assertIn("std_values=STD_VALUES", toolkit2_text)
            self.assertIn("from rknn.api import RKNN", legacy_text)
            self.assertIn("python3 -m pip install 'setuptools<82'", legacy_text)
            self.assertIn('Path(__file__).with_name("calibration.txt")', legacy_text)
            self.assertIn("DO_QUANTIZATION = True", legacy_text)
            self.assertIn("MEAN_VALUES = [[0, 0, 0]]", legacy_text)
            self.assertIn("STD_VALUES = [[255, 255, 255]]", legacy_text)
            self.assertIn("mean_values=MEAN_VALUES", legacy_text)
            self.assertIn("std_values=STD_VALUES", legacy_text)

            helper_text = Path(bundle["calibration_helper"]).read_text(encoding="utf-8")
            calibration_text = Path(bundle["calibration_file"]).read_text(encoding="utf-8")
            self.assertIn("Build an RKNN calibration.txt file", helper_text)
            self.assertIn("One absolute image path per line", calibration_text)


if __name__ == "__main__":
    unittest.main()
