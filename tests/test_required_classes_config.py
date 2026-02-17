from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cds.training.evaluate import load_eval_config
from cds.training.train import load_train_config


class RequiredClassesConfigTests(unittest.TestCase):
    def test_train_config_requires_required_classes(self) -> None:
        with self.assertRaisesRegex(ValueError, "gating.required_classes"):
            load_train_config(config_path=None)

    def test_eval_config_requires_required_classes(self) -> None:
        with self.assertRaisesRegex(ValueError, "gating.required_classes"):
            load_eval_config(config_path=None)

    def test_train_config_accepts_required_classes_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "train.json"
            config_path.write_text(
                """{
  "gating": {
    "required_classes": ["cat", "dog"],
    "recall_thresholds": {
      "cat": 0.6,
      "dog": 0.6
    }
  }
}""".strip(),
                encoding="utf-8",
            )
            cfg = load_train_config(config_path=str(config_path))
            self.assertEqual(cfg["gating"]["required_classes"], ["cat", "dog"])

    def test_eval_config_accepts_required_classes_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "eval.json"
            config_path.write_text(
                """{
  "gating": {
    "required_classes": ["cat", "dog"],
    "recall_thresholds": {
      "cat": 0.6,
      "dog": 0.6
    }
  }
}""".strip(),
                encoding="utf-8",
            )
            cfg = load_eval_config(config_path=str(config_path))
            self.assertEqual(cfg["gating"]["required_classes"], ["cat", "dog"])


if __name__ == "__main__":
    unittest.main()
