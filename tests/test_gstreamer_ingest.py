from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from cds.io.ingest.gstreamer_ingest import (
    GStreamerIngest,
    _normalize_appsink_pipeline,
    _sample_to_bgr_frame,
)


class _FakeStructure:
    def __init__(self, *, width: int, height: int, pixel_format: str, framerate=None) -> None:
        self._values = {
            "width": width,
            "height": height,
            "format": pixel_format,
        }
        if framerate is not None:
            self._values["framerate"] = framerate

    def get_value(self, key: str):
        return self._values[key]

    def has_field(self, key: str) -> bool:
        return key in self._values


class _FakeCaps:
    def __init__(self, structure: _FakeStructure) -> None:
        self._structure = structure

    def get_size(self) -> int:
        return 1

    def get_structure(self, index: int) -> _FakeStructure:
        assert index == 0
        return self._structure


class _FakeMapInfo:
    def __init__(self, data: bytes) -> None:
        self.data = data


class _FakeBuffer:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.unmapped = False

    def map(self, _flags):
        return True, _FakeMapInfo(self._payload)

    def unmap(self, _map_info) -> None:
        self.unmapped = True


class _FakeSample:
    def __init__(self, caps: _FakeCaps, buffer: _FakeBuffer) -> None:
        self._caps = caps
        self._buffer = buffer

    def get_caps(self) -> _FakeCaps:
        return self._caps

    def get_buffer(self) -> _FakeBuffer:
        return self._buffer


class _FakeAppSink:
    def set_property(self, _name: str, _value) -> None:
        return None


class _FakeBus:
    def timed_pop_filtered(self, _timeout: int, _mask):
        return None


class _FakePipeline:
    def __init__(self, sink_name: str) -> None:
        self._appsink = _FakeAppSink()
        self._sink_name = sink_name
        self._bus = _FakeBus()
        self.states: list[object] = []

    def get_by_name(self, name: str):
        if name == self._sink_name:
            return self._appsink
        return None

    def get_bus(self) -> _FakeBus:
        return self._bus

    def set_state(self, state):
        self.states.append(state)
        return "ok"


class _FakeGst:
    SECOND = 1_000_000_000
    State = SimpleNamespace(PLAYING="playing", NULL="null")
    StateChangeReturn = SimpleNamespace(FAILURE="failure")
    MessageType = SimpleNamespace(ERROR=1, EOS=2)
    MapFlags = SimpleNamespace(READ=1)

    def __init__(self, sink_name: str = "cdsappsink") -> None:
        self._sink_name = sink_name

    def parse_launch(self, _pipeline: str) -> _FakePipeline:
        return _FakePipeline(self._sink_name)


class GStreamerIngestTests(unittest.TestCase):
    def test_normalize_pipeline_injects_default_appsink_name(self) -> None:
        pipeline, sink_name = _normalize_appsink_pipeline(
            "filesrc location=/tmp/test.mp4 ! qtdemux ! appsink drop=true"
        )

        self.assertEqual(sink_name, "cdsappsink")
        self.assertIn("appsink drop=true name=cdsappsink", pipeline)

    def test_normalize_pipeline_preserves_existing_appsink_name(self) -> None:
        pipeline, sink_name = _normalize_appsink_pipeline(
            "filesrc location=/tmp/test.mp4 ! appsink name=mysink drop=true"
        )

        self.assertEqual(sink_name, "mysink")
        self.assertEqual(pipeline, "filesrc location=/tmp/test.mp4 ! appsink name=mysink drop=true")

    @patch("cds.io.ingest.gstreamer_ingest._load_gst_bindings", return_value=_FakeGst())
    def test_sample_to_bgr_frame_handles_tightly_packed_buffer(self, _mock_gst) -> None:
        caps = _FakeCaps(_FakeStructure(width=2, height=2, pixel_format="BGR"))
        payload = bytes(range(12))
        buffer = _FakeBuffer(payload)
        sample = _FakeSample(caps, buffer)

        frame, fps = _sample_to_bgr_frame(sample)

        self.assertEqual(frame.shape, (2, 2, 3))
        self.assertTrue(buffer.unmapped)
        self.assertIsNone(fps)
        self.assertEqual(int(frame[1, 1, 2]), 11)

    @patch("cds.io.ingest.gstreamer_ingest._load_gst_bindings", return_value=_FakeGst())
    def test_sample_to_bgr_frame_handles_padded_rows(self, _mock_gst) -> None:
        caps = _FakeCaps(_FakeStructure(width=2, height=2, pixel_format="BGR"))
        row0 = bytes([1, 2, 3, 4, 5, 6, 99, 99])
        row1 = bytes([7, 8, 9, 10, 11, 12, 88, 88])
        buffer = _FakeBuffer(row0 + row1)
        sample = _FakeSample(caps, buffer)

        frame, _fps = _sample_to_bgr_frame(sample)

        self.assertEqual(frame.shape, (2, 2, 3))
        self.assertEqual(frame[0, 0].tolist(), [1, 2, 3])
        self.assertEqual(frame[1, 1].tolist(), [10, 11, 12])

    def test_open_requires_pipeline(self) -> None:
        ingest = GStreamerIngest()

        with self.assertRaises(RuntimeError):
            ingest.open("/tmp/test.mp4", {})

    @patch("cds.io.ingest.gstreamer_ingest._load_gst_bindings", return_value=_FakeGst())
    def test_open_prefetches_first_frame_for_file_sources(self, _mock_gst) -> None:
        ingest = GStreamerIngest()
        prefetched = np.zeros((8, 8, 3), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix=".mp4") as handle:
            with patch.object(ingest, "_pull_frame", return_value=prefetched):
                ingest.open(
                    handle.name,
                    {"gstreamer_pipeline": "filesrc location=/tmp/test.mp4 ! appsink drop=true"},
                )

            self.assertEqual(ingest.source_mode(), "video-file")
            packet = ingest.read_latest()
            self.assertIsNotNone(packet)
            assert packet is not None
            self.assertEqual(packet.frame.shape, (8, 8, 3))
            self.assertEqual(packet.source, handle.name)


if __name__ == "__main__":
    unittest.main()
