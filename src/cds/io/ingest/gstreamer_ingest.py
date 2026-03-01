from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from cds.io.ingest.opencv_ingest import OpenCVIngest
from cds.types import FramePacket


class GStreamerIngest(OpenCVIngest):
    def __init__(self) -> None:
        super().__init__()
        self._pipeline: str | None = None
        self._prefetched_frame: Any | None = None

    def open(self, uri: str, options: dict[str, Any] | None = None) -> None:
        self._pipeline = None
        self._prefetched_frame = None
        options = options or {}
        pipeline = options.get("gstreamer_pipeline")
        if isinstance(pipeline, str) and pipeline.strip():
            self._pipeline = pipeline
            self._uri = pipeline
            self._entries = []
            self._entry_index = 0
            self._current_image_pending = False
            path = Path(uri)
            self._source_mode = "video-file" if path.is_file() else "live-stream"
            self._nominal_fps = None
            self._release_capture()
            self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not self._cap.isOpened():
                self._release_capture()
                raise RuntimeError(
                    "Failed to open GStreamer pipeline with OpenCV CAP_GSTREAMER. "
                    "Verify OpenCV was built with GStreamer support and that the "
                    "pipeline ends with an appsink OpenCV can consume."
                )
            try:
                fps = float(self._cap.get(cv2.CAP_PROP_FPS))
                self._nominal_fps = fps if fps > 0 else None
            except Exception:
                self._nominal_fps = None
            if self._source_mode == "video-file":
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    self._release_capture()
                    raise RuntimeError(
                        "GStreamer pipeline opened, but OpenCV CAP_GSTREAMER could not "
                        "read the first frame. The pipeline may work in gst-launch "
                        "while still being incompatible with OpenCV appsink capture."
                    )
                self._prefetched_frame = frame
            return

        super().open(uri, options)

    def read_latest(self) -> FramePacket | None:
        if self._prefetched_frame is not None:
            frame = self._prefetched_frame
            self._prefetched_frame = None
            self._frame_id += 1
            return FramePacket(
                frame_id=self._frame_id,
                frame=frame,
                source=self._uri,
                timestamp=datetime.now(),
            )
        return super().read_latest()

    def name(self) -> str:
        return "gstreamer"
