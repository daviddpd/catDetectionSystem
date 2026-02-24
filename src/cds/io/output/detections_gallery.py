from __future__ import annotations

import math
import time
from typing import Any

import cv2
import numpy as np

from cds.pipeline.annotate import _color_for_label
from cds.pipeline.detection_capture import DetectionSnapshot


class DetectionsGallerySink:
    def __init__(
        self,
        window_name: str = "cds-detections",
        slots: int = 6,
        scale: float = 0.5,
    ) -> None:
        self._window_name = window_name
        self._slots = max(1, int(slots))
        self._scale = max(0.1, float(scale))
        self._open = False
        self._page = 0
        self._last_render_at = 0.0
        self._render_interval_s = 0.1

    def open(self) -> None:
        if self._open:
            return
        cv2.namedWindow(
            self._window_name,
            cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED,
        )
        self._open = True

    def handle_key(self, key: int) -> None:
        if key < 0:
            return
        if key in {ord("["), ord(","), ord("j")}:
            self._page += 1
        elif key in {ord("]"), ord("."), ord("k")}:
            self._page = max(0, self._page - 1)
        elif key in {ord("0"), ord("g")}:
            self._page = 0

    def _draw_detection_overlays(self, frame: Any, detections: list[Any]) -> Any:
        canvas = frame.copy()
        for det in detections:
            color = _color_for_label(str(det.label))
            cv2.rectangle(canvas, (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2)), color, 2)
            area_percent = float(getattr(det, "extra", {}).get("area_percent", 0.0))
            label = f"{det.label} {det.confidence:.2f} {area_percent:.2f}%"
            cv2.putText(
                canvas,
                label,
                (int(det.x1), max(18, int(det.y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
        return canvas

    def _select_page(self, snapshots: list[DetectionSnapshot]) -> list[DetectionSnapshot]:
        if not snapshots:
            self._page = 0
            return []
        total = len(snapshots)
        page_size = self._slots
        max_page = max(0, math.ceil(total / page_size) - 1)
        self._page = min(self._page, max_page)
        end = total - (self._page * page_size)
        start = max(0, end - page_size)
        return snapshots[start:end]

    def _tile(self, tiles: list[Any], total_count: int) -> Any:
        if not tiles:
            canvas = np.zeros((220, 640, 3), dtype=np.uint8)
            cv2.putText(canvas, "cds-detections: no captured detections yet", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Keys: [ older   ] newer   0 reset", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
            return canvas

        resized: list[Any] = []
        cell_w = 0
        cell_h = 0
        for img in tiles:
            scaled = cv2.resize(
                img,
                None,
                fx=self._scale,
                fy=self._scale,
                interpolation=cv2.INTER_AREA,
            )
            resized.append(scaled)
            h, w = scaled.shape[:2]
            cell_h = max(cell_h, h)
            cell_w = max(cell_w, w)

        cols = min(3, max(1, self._slots))
        rows = max(1, math.ceil(self._slots / cols))
        pad = 8
        header_h = 44
        canvas_h = header_h + rows * cell_h + (rows + 1) * pad
        canvas_w = cols * cell_w + (cols + 1) * pad
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        for idx, img in enumerate(resized):
            row = idx // cols
            col = idx % cols
            y = header_h + pad + row * (cell_h + pad)
            x = pad + col * (cell_w + pad)
            h, w = img.shape[:2]
            canvas[y : y + h, x : x + w] = img

        cv2.putText(
            canvas,
            f"cds-detections total={total_count} page={self._page} slots={self._slots}  keys:[ ] 0",
            (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        return canvas

    def render(self, snapshots: list[DetectionSnapshot], key: int = -1) -> None:
        if not self._open:
            self.open()
        self.handle_key(key)
        now = time.monotonic()
        if now - self._last_render_at < self._render_interval_s:
            return
        self._last_render_at = now

        page = self._select_page(snapshots)
        tiles: list[Any] = []
        for snapshot in page:
            annotated = self._draw_detection_overlays(snapshot.frame, snapshot.detections)
            tiles.append(annotated)
        canvas = self._tile(tiles, total_count=len(snapshots))
        cv2.imshow(self._window_name, canvas)

    def close(self) -> None:
        if self._open:
            cv2.destroyWindow(self._window_name)
            self._open = False

    def name(self) -> str:
        return "detections-gallery"
