from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cds.training.bootstrap import _expand_sources


class BootstrapSourceScanTests(unittest.TestCase):
    def test_recursive_media_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a").mkdir(parents=True, exist_ok=True)
            (root / "b" / "nested").mkdir(parents=True, exist_ok=True)

            keep_1 = root / "a" / "one.jpg"
            keep_2 = root / "b" / "nested" / "two.mp4"
            skip = root / "a" / "notes.txt"

            keep_1.write_bytes(b"x")
            keep_2.write_bytes(b"x")
            skip.write_text("ignore", encoding="utf-8")

            results = _expand_sources(str(root))

            self.assertIn(str(keep_1.resolve()), results)
            self.assertIn(str(keep_2.resolve()), results)
            self.assertNotIn(str(skip.resolve()), results)


if __name__ == "__main__":
    unittest.main()
