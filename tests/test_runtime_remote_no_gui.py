from __future__ import annotations

import sys
import types
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

sys.modules.setdefault("cv2", types.ModuleType("cv2"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))

from cds.config.models import RuntimeConfig
from cds.io.ingest.decoder_probe import DecoderProbeResult
from cds.pipeline.runtime import DetectionRuntime
from cds.types import FramePacket


class _FakeFrame:
    shape = (32, 32, 3)


class _FakeBackend:
    def warmup(self) -> None:
        return

    def infer(self, frame) -> list:
        return []

    def name(self) -> str:
        return "fake"

    def device_info(self) -> str:
        return "test-device"


class _FakeIngest:
    def __init__(self) -> None:
        self._packets = [
            FramePacket(
                frame_id=1,
                frame=_FakeFrame(),
                source="/tmp/test.mp4",
                timestamp=datetime.now(),
            )
        ]

    def name(self) -> str:
        return "fake-ingest"

    def open(self, uri: str, options: dict | None = None) -> None:
        return

    def read_latest(self):
        if self._packets:
            return self._packets.pop(0)
        return None

    def source_mode(self) -> str:
        return "video-file"

    def nominal_fps(self) -> float:
        return 30.0

    def close(self) -> None:
        return


class _FakeMjpegSink:
    instances: list["_FakeMjpegSink"] = []

    def __init__(self, host: str, port: int, path: str) -> None:
        self.host = host
        self.port = port
        self.path = path
        self.open_called = False
        self.write_calls = 0
        self.close_called = False
        self.endpoint_url = f"http://{host}:{port}{path}"
        self.__class__.instances.append(self)

    def open(self) -> None:
        self.open_called = True

    def write(self, frame) -> bool:
        self.write_calls += 1
        return True

    def close(self) -> None:
        self.close_called = True


class _FakeStatsLogger:
    def __init__(self, *args, **kwargs) -> None:
        return

    def maybe_emit(self) -> None:
        return


class _FailDisplaySink:
    def __init__(self, *args, **kwargs) -> None:
        raise AssertionError("DisplaySink should not be constructed when no GUI is available")


class RemoteNoGuiRuntimeTests(unittest.TestCase):
    def test_remote_mjpeg_runs_without_local_display_when_gui_unavailable(self) -> None:
        config = RuntimeConfig()
        config.ingest.uri = "/tmp/test.mp4"
        config.output.headless = False
        config.output.remote_enabled = True
        config.output.remote_host = "127.0.0.1"
        config.output.remote_port = 9090
        config.output.remote_path = "/stream.mjpg"
        config.output.detections_window_enabled = True
        config.monitoring.event_stdout = False
        config.triggers.audio.enabled = False
        config.triggers.hooks.enabled = False

        _FakeMjpegSink.instances.clear()

        with patch(
            "cds.pipeline.runtime.probe_decoder_path",
            return_value=DecoderProbeResult(
                selected_decoder="software",
                reason="test",
                available=["software"],
            ),
        ), patch(
            "cds.pipeline.runtime.select_backend",
            return_value=SimpleNamespace(
                backend=_FakeBackend(),
                reason="test backend",
            ),
        ), patch(
            "cds.pipeline.runtime.select_ingest_backend",
            return_value=(_FakeIngest(), "test ingest"),
        ), patch(
            "cds.pipeline.runtime.MjpegSink",
            _FakeMjpegSink,
        ), patch(
            "cds.pipeline.runtime.DisplaySink",
            _FailDisplaySink,
        ), patch(
            "cds.pipeline.runtime.PeriodicStatsLogger",
            _FakeStatsLogger,
        ), patch(
            "cds.pipeline.runtime.draw_overlays",
            lambda *args, **kwargs: None,
        ), patch(
            "cds.pipeline.runtime._local_gui_available",
            return_value=False,
        ):
            runtime = DetectionRuntime(repo_root=SimpleNamespace(), config=config)
            exit_code = runtime.run()

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(_FakeMjpegSink.instances), 1)
        sink = _FakeMjpegSink.instances[0]
        self.assertTrue(sink.open_called)
        self.assertGreaterEqual(sink.write_calls, 1)
        self.assertTrue(sink.close_called)


if __name__ == "__main__":
    unittest.main()
