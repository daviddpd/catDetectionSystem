from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cds.training.export import _stage_export_artifact


class ExportArtifactsTests(unittest.TestCase):
    def test_stage_export_artifact_copies_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            exports_dir = root / "exports"
            exports_dir.mkdir(parents=True, exist_ok=True)
            source = root / "model.onnx"
            source.write_text("onnx", encoding="utf-8")

            staged = _stage_export_artifact(source, exports_dir)

            self.assertEqual(staged, exports_dir / "model.onnx")
            self.assertTrue(staged.exists())
            self.assertTrue(staged.is_file())
            self.assertEqual(staged.read_text(encoding="utf-8"), "onnx")

    def test_stage_export_artifact_copies_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            exports_dir = root / "exports"
            exports_dir.mkdir(parents=True, exist_ok=True)

            source_dir = root / "best.mlpackage"
            source_dir.mkdir(parents=True, exist_ok=True)
            payload = source_dir / "weights.bin"
            payload.write_text("coreml", encoding="utf-8")

            staged = _stage_export_artifact(source_dir, exports_dir)

            self.assertEqual(staged, exports_dir / "best.mlpackage")
            self.assertTrue(staged.exists())
            self.assertTrue(staged.is_dir())
            self.assertEqual((staged / "weights.bin").read_text(encoding="utf-8"), "coreml")


if __name__ == "__main__":
    unittest.main()
