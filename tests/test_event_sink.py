from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cds.io.output.events import JsonEventSink


class JsonEventSinkTests(unittest.TestCase):
    def test_enabled_false_when_no_stdout_and_no_file(self) -> None:
        sink = JsonEventSink(stdout_enabled=False, file_path=None)
        try:
            self.assertFalse(sink.enabled())
        finally:
            sink.close()

    def test_enabled_true_when_stdout_enabled(self) -> None:
        sink = JsonEventSink(stdout_enabled=True, file_path=None)
        try:
            self.assertTrue(sink.enabled())
        finally:
            sink.close()

    def test_enabled_true_when_file_is_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "events.jsonl"
            sink = JsonEventSink(stdout_enabled=False, file_path=str(output_path))
            try:
                self.assertTrue(sink.enabled())
            finally:
                sink.close()


if __name__ == "__main__":
    unittest.main()
