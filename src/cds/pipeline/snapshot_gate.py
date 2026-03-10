from __future__ import annotations

from dataclasses import dataclass

from cds.types import Detection


@dataclass
class WindowSnapshotGate:
    on_frames: int
    off_frames: int
    min_area_pixels: int
    min_area_percent: float
    _on_count: int = 0
    _off_count: int = 0
    _active: bool = False

    def _qualifies(self, detection: Detection) -> bool:
        area_pixels = int(detection.extra.get("area_pixels", detection.width * detection.height))
        area_percent = float(detection.extra.get("area_percent", 0.0))
        if self.min_area_pixels > 0 and area_pixels < self.min_area_pixels:
            return False
        if self.min_area_percent > 0.0 and area_percent < self.min_area_percent:
            return False
        return True

    def observe(self, detections: list[Detection]) -> tuple[list[Detection], bool]:
        matching = [det for det in detections if self._qualifies(det)]
        activated: list[Detection] = []

        if matching:
            self._on_count += 1
            self._off_count = 0
            if not self._active and self._on_count >= self.on_frames:
                self._active = True
                activated = matching
        else:
            if self._active:
                self._off_count += 1
                if self._off_count >= self.off_frames:
                    self._active = False
                    self._on_count = 0
                    self._off_count = 0
            else:
                self._on_count = 0
                self._off_count = 0

        return activated, self._active
