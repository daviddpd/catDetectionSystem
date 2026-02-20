from __future__ import annotations

from typing import Any

import cv2

from cds.io.ingest.opencv_ingest import OpenCVIngest
from cds.types import FramePacket


class GStreamerIngest(OpenCVIngest):
    def __init__(self) -> None:
        super().__init__()
        self._pipeline: str | None = None

    def open(self, uri: str, options: dict[str, Any] | None = None) -> None:
        self._pipeline = None
        options = options or {}
        pipeline = options.get("gstreamer_pipeline")
        if isinstance(pipeline, str) and pipeline.strip():
            self._pipeline = pipeline
            self._uri = pipeline
            self._entries = []
            self._entry_index = 0
            self._current_image_pending = False
            self._source_mode = "live-stream"
            self._nominal_fps = None
            self._release_capture()
            self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            return

        super().open(uri, options)

    def read_latest(self) -> FramePacket | None:
        return super().read_latest()

    def name(self) -> str:
        return "gstreamer"
