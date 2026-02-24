from __future__ import annotations

from typing import Any

import cv2

from cds.io.output.base import OutputSink


class DisplaySink(OutputSink):
    def __init__(self, window_name: str = "catDetectionSystem") -> None:
        self._window_name = window_name
        self._open = False
        self._last_key = -1

    def open(self) -> None:
        if self._open:
            return
        cv2.namedWindow(
            self._window_name,
            cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED,
        )
        self._open = True

    def write(self, frame: Any, metadata: dict | None = None) -> bool:
        if not self._open:
            self.open()
        cv2.imshow(self._window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        self._last_key = int(key)
        if key in {27, ord("q")}:
            return False
        return True

    def consume_key(self) -> int:
        key = self._last_key
        self._last_key = -1
        return key

    def close(self) -> None:
        if self._open:
            cv2.destroyWindow(self._window_name)
            self._open = False
        self._last_key = -1

    def name(self) -> str:
        return "display"
