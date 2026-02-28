from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cds.detector.models.model_spec import ModelSpec


class ModelSpecTests(unittest.TestCase):
    def test_read_labels_ignores_comments_and_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            labels_path = Path(tmpdir) / "labels.txt"
            labels_path.write_text(
                "\n# comment\ncat\n dog \n#another\n\nraccoon\n",
                encoding="utf-8",
            )
            spec = ModelSpec(
                name="labels-test",
                labels_path=str(labels_path),
            )

            labels = spec.read_labels()

            self.assertEqual(labels, ["cat", "dog", "raccoon"])


if __name__ == "__main__":
    unittest.main()
