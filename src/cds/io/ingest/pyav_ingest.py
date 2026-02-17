from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from cds.io.ingest.base import VideoIngest
from cds.types import FramePacket

_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm"}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class PyAVIngest(VideoIngest):
    def __init__(self) -> None:
        self._av = None
        self._uri = ""
        self._options: dict[str, Any] = {}
        self._frame_id = 0
        self._entries: list[Path] = []
        self._entry_index = 0
        self._current_image_pending = False
        self._container = None
        self._frame_iter = None

    def _require_av(self) -> Any:
        if self._av is not None:
            return self._av
        try:
            import av
        except ImportError as exc:
            raise RuntimeError(
                "PyAV ingest selected but `av` package is not installed."
            ) from exc
        self._av = av
        return av

    def open(self, uri: str, options: dict[str, Any] | None = None) -> None:
        self._uri = uri
        options = options or {}
        nested_options = options.get("pyav_options")
        if isinstance(nested_options, dict):
            self._options = {str(k): str(v) for k, v in nested_options.items()}
        else:
            self._options = {str(k): str(v) for k, v in options.items()}
        self._frame_id = 0
        self._entries = []
        self._entry_index = 0
        self._current_image_pending = False
        self._close_container()

        path = Path(uri)
        if path.is_dir():
            self._entries = sorted(
                [
                    p
                    for p in path.rglob("*")
                    if p.is_file() and p.suffix.lower() in (_VIDEO_EXTENSIONS | _IMAGE_EXTENSIONS)
                ]
            )
            return

        if path.is_file():
            self._entries = [path]
            return

        av = self._require_av()
        self._container = av.open(uri, options=self._options)
        self._frame_iter = self._container.decode(video=0)

    def _open_next_entry(self) -> bool:
        self._close_container()
        if self._entry_index >= len(self._entries):
            return False

        entry = self._entries[self._entry_index]
        if entry.suffix.lower() in _IMAGE_EXTENSIONS:
            self._current_image_pending = True
            return True

        av = self._require_av()
        self._container = av.open(str(entry), options=self._options)
        self._frame_iter = self._container.decode(video=0)
        self._current_image_pending = False
        return True

    def _close_container(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None
            self._frame_iter = None

    def _read_from_entries(self) -> FramePacket | None:
        while True:
            if self._entry_index >= len(self._entries):
                return None

            entry = self._entries[self._entry_index]
            if self._container is None and not self._current_image_pending:
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

            if self._frame_iter is None:
                self._entry_index += 1
                continue

            try:
                av_frame = next(self._frame_iter)
            except StopIteration:
                self._entry_index += 1
                self._close_container()
                continue

            frame = av_frame.to_ndarray(format="bgr24")
            self._frame_id += 1
            return FramePacket(
                frame_id=self._frame_id,
                frame=frame,
                source=str(entry),
                timestamp=datetime.now(),
            )

    def read_latest(self) -> FramePacket | None:
        if self._entries:
            return self._read_from_entries()

        if self._frame_iter is None:
            return None

        try:
            av_frame = next(self._frame_iter)
        except StopIteration:
            return None

        frame = av_frame.to_ndarray(format="bgr24")
        self._frame_id += 1
        return FramePacket(
            frame_id=self._frame_id,
            frame=frame,
            source=self._uri,
            timestamp=datetime.now(),
        )

    def close(self) -> None:
        self._close_container()

    def name(self) -> str:
        return "pyav"
