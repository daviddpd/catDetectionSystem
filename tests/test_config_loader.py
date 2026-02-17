from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cds.config.loader import load_runtime_config


class ConfigLoaderTests(unittest.TestCase):
    def test_headless_forces_remote_off_and_event_stdout_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_file = root / "cds.json"
            config_file.write_text(
                """
{
  "output": {
    "headless": true,
    "remote_enabled": true
  },
  "monitoring": {
    "event_stdout": true
  }
}
""".strip(),
                encoding="utf-8",
            )

            config = load_runtime_config(repo_root=root, config_path=str(config_file))

            self.assertTrue(config.output.headless)
            self.assertFalse(config.output.remote_enabled)
            self.assertFalse(config.monitoring.event_stdout)


if __name__ == "__main__":
    unittest.main()
