from __future__ import annotations

import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from cds.io.ingest.gstreamer_ingest import GStreamerIngest


class GStreamerIngestTests(unittest.TestCase):
    @patch("cds.io.ingest.gstreamer_ingest.cv2.VideoCapture")
    def test_pipeline_file_uri_preserves_file_source_mode(self, mock_capture_cls: MagicMock) -> None:
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture.get.return_value = 30.0
        mock_capture.read.return_value = (True, np.zeros((8, 8, 3), dtype=np.uint8))
        mock_capture_cls.return_value = mock_capture

        with tempfile.NamedTemporaryFile(suffix=".mp4") as handle:
            ingest = GStreamerIngest()
            ingest.open(
                handle.name,
                {"gstreamer_pipeline": "filesrc location=/tmp/test.mp4 ! fakesink"},
            )

            self.assertEqual(ingest.source_mode(), "video-file")
            self.assertEqual(ingest.nominal_fps(), 30.0)
            mock_capture_cls.assert_called_once_with(
                "filesrc location=/tmp/test.mp4 ! fakesink",
                cv2.CAP_GSTREAMER,
            )
            packet = ingest.read_latest()
            self.assertIsNotNone(packet)
            self.assertEqual(packet.frame_id, 1)
            self.assertEqual(packet.frame.shape, (8, 8, 3))

    @patch("cds.io.ingest.gstreamer_ingest.cv2.VideoCapture")
    def test_pipeline_open_raises_when_opencv_cannot_open_capture(
        self,
        mock_capture_cls: MagicMock,
    ) -> None:
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = False
        mock_capture_cls.return_value = mock_capture

        with tempfile.NamedTemporaryFile(suffix=".mp4") as handle:
            ingest = GStreamerIngest()
            with self.assertRaises(RuntimeError):
                ingest.open(
                    handle.name,
                    {"gstreamer_pipeline": "filesrc location=/tmp/test.mp4 ! fakesink"},
                )

        mock_capture.release.assert_called()

    @patch("cds.io.ingest.gstreamer_ingest.cv2.VideoCapture")
    def test_pipeline_open_raises_when_first_frame_cannot_be_read(
        self,
        mock_capture_cls: MagicMock,
    ) -> None:
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture.get.return_value = 30.0
        mock_capture.read.return_value = (False, None)
        mock_capture_cls.return_value = mock_capture

        with tempfile.NamedTemporaryFile(suffix=".mp4") as handle:
            ingest = GStreamerIngest()
            with self.assertRaises(RuntimeError):
                ingest.open(
                    handle.name,
                    {"gstreamer_pipeline": "filesrc location=/tmp/test.mp4 ! fakesink"},
                )

        mock_capture.release.assert_called()


if __name__ == "__main__":
    unittest.main()
