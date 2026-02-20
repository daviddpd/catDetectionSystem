from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from cds.io.ingest.base import VideoIngest
from cds.types import FramePacket

_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm"}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class OpenCVIngest(VideoIngest):
    def __init__(self) -> None:
        self._uri = ""
        self._options: dict[str, Any] = {}
        self._frame_id = 0
        self._entries: list[Path] = []
        self._entry_index = 0
        self._current_image_pending = False
        self._cap: cv2.VideoCapture | None = None
        self._source_mode = "live-stream"
        self._nominal_fps: float | None = None

    def open(self, uri: str, options: dict[str, Any] | None = None) -> None:
        self._uri = uri
        self._options = options or {}
        self._frame_id = 0
        self._entries = []
        self._entry_index = 0
        self._current_image_pending = False
        self._release_capture()
        self._nominal_fps = None

        path = Path(uri)
        if path.is_dir():
            self._source_mode = "directory"
            self._entries = sorted(
                [
                    p
                    for p in path.rglob("*")
                    if p.is_file() and p.suffix.lower() in (_VIDEO_EXTENSIONS | _IMAGE_EXTENSIONS)
                ]
            )
            return

        if path.is_file():
            if path.suffix.lower() in _IMAGE_EXTENSIONS:
                self._source_mode = "image-file"
            else:
                self._source_mode = "video-file"
            self._entries = [path]
            return

        # Treat as stream URL (rtsp/http/etc.)
        self._source_mode = "live-stream"
        self._cap = self._open_capture(uri)

    def _open_capture(self, uri: str) -> cv2.VideoCapture:
        if uri.startswith("rtsp://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|reorder_queue_size;0|fflags;nobuffer"
            )
        cap = cv2.VideoCapture(uri, cv2.CAP_FFMPEG)
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            self._nominal_fps = fps if fps > 0 else None
        except Exception:
            self._nominal_fps = None
        return cap

    def _release_capture(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _open_next_entry(self) -> bool:
        self._release_capture()
        if self._entry_index >= len(self._entries):
            return False

        entry = self._entries[self._entry_index]
        if entry.suffix.lower() in _IMAGE_EXTENSIONS:
            self._current_image_pending = True
            self._nominal_fps = None
            return True

        self._cap = self._open_capture(str(entry))
        self._current_image_pending = False
        return True

    def _next_frame_from_entries(self) -> FramePacket | None:
        while True:
            if self._entry_index >= len(self._entries):
                return None

            entry = self._entries[self._entry_index]
            if self._cap is None and not self._current_image_pending:
                if not self._open_next_entry():
                    return None

            if self._current_image_pending:
                frame = cv2.imread(str(entry))
                self._current_image_pending = False
                self._entry_index += 1
                if frame is None:
                    continue
                self._frame_id += 1
                return FramePacket(
                    frame_id=self._frame_id,
                    frame=frame,
                    source=str(entry),
                    timestamp=datetime.now(),
                )

            if self._cap is None:
                self._entry_index += 1
                continue

            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._entry_index += 1
                self._release_capture()
                continue

            self._frame_id += 1
            return FramePacket(
                frame_id=self._frame_id,
                frame=frame,
                source=str(entry),
                timestamp=datetime.now(),
            )

    def read_latest(self) -> FramePacket | None:
        if self._entries:
            return self._next_frame_from_entries()

        if self._cap is None:
            return None

        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None

        self._frame_id += 1
        return FramePacket(
            frame_id=self._frame_id,
            frame=frame,
            source=self._uri,
            timestamp=datetime.now(),
        )

    def close(self) -> None:
        self._release_capture()

    def name(self) -> str:
        return "opencv"

    def source_mode(self) -> str:
        return self._source_mode

    def nominal_fps(self) -> float | None:
        return self._nominal_fps
