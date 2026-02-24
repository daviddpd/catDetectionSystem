from __future__ import annotations

import copy
import logging
import queue
import random
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import cv2

from cds.types import Detection, FramePacket


@dataclass
class DetectionSnapshot:
    frame_id: int
    source: str
    timestamp: datetime
    frame: Any
    detections: list[Detection]
    activated_labels: list[str]


@dataclass
class _ExportCandidate:
    frame_id: int
    source: str
    timestamp: datetime
    frame: Any
    detections: list[Detection]
    reason: str


class DetectionCaptureManager:
    def __init__(
        self,
        *,
        buffer_frames: int,
        export_enabled: bool,
        export_dir: str | None,
        export_sample_percent: float,
        confidence_low: float,
        confidence_min: float | None,
    ) -> None:
        self._logger = logging.getLogger("cds.capture")
        self._buffer_frames = max(1, int(buffer_frames))
        self._snapshots: deque[DetectionSnapshot] = deque(maxlen=self._buffer_frames)
        self._snapshots_lock = threading.Lock()
        self._queue: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=128)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._export_enabled = bool(export_enabled and export_dir)
        self._export_dir = Path(export_dir).expanduser().resolve() if export_dir else None
        self._export_sample_percent = max(0.0, min(100.0, float(export_sample_percent)))
        self._confidence_low = float(confidence_low)
        self._confidence_min = (
            float(confidence_min) if confidence_min is not None else None
        )
        self._rng = random.Random()
        self._export_serial = 0
        self._export_state = "idle"  # idle|band|high
        self._last_band_candidate: _ExportCandidate | None = None

        if self._export_enabled and (
            self._confidence_min is None or self._confidence_min <= self._confidence_low
        ):
            self._logger.warning(
                "export_frames disabled because confidence_min (%s) must be > confidence (%s)",
                self._confidence_min,
                self._confidence_low,
            )
            self._export_enabled = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._worker_loop, name="cds-capture", daemon=True)
        self._thread.start()

    def enabled(self) -> bool:
        return True

    def export_enabled(self) -> bool:
        return self._export_enabled

    def _clone_detections(self, detections: list[Detection]) -> list[Detection]:
        return [copy.deepcopy(det) for det in detections]

    def _submit_task(self, kind: str, payload: object) -> None:
        try:
            self._queue.put_nowait((kind, payload))
        except queue.Full:
            self._logger.warning("capture queue full, dropping task kind=%s", kind)

    def submit_activation_snapshot(
        self,
        packet: FramePacket,
        frame: Any,
        detections: list[Detection],
        activated_detections: list[Detection],
    ) -> None:
        activated_labels = sorted({det.label for det in activated_detections})
        if not activated_labels:
            return
        snapshot = DetectionSnapshot(
            frame_id=packet.frame_id,
            source=packet.source,
            timestamp=packet.timestamp,
            frame=frame.copy(),
            detections=self._clone_detections(detections),
            activated_labels=activated_labels,
        )
        self._submit_task("snapshot", snapshot)

    def observe_benchmark_export(
        self,
        packet: FramePacket,
        frame: Any,
        detections: list[Detection],
    ) -> None:
        if not self._export_enabled:
            return
        if self._confidence_min is None:
            return

        band_dets = [
            det
            for det in detections
            if self._confidence_low <= det.confidence < self._confidence_min
        ]
        high_dets = [det for det in detections if det.confidence >= self._confidence_min]

        state_now = "high" if high_dets else ("band" if band_dets else "idle")
        exported_this_frame = False

        if state_now == "high" and self._export_state == "band" and self._last_band_candidate is not None:
            self._submit_task("export", self._last_band_candidate)
            exported_this_frame = True

        if state_now == "band":
            candidate = _ExportCandidate(
                frame_id=packet.frame_id,
                source=packet.source,
                timestamp=packet.timestamp,
                frame=frame.copy(),
                detections=self._clone_detections(band_dets),
                reason=(
                    "after-drop"
                    if self._export_state == "high"
                    else "random-sample"
                ),
            )
            self._last_band_candidate = candidate

            if self._export_state == "high":
                self._submit_task("export", candidate)
                exported_this_frame = True

            if (not exported_this_frame) and self._export_sample_percent > 0.0:
                if self._rng.random() < (self._export_sample_percent / 100.0):
                    self._submit_task(
                        "export",
                        _ExportCandidate(
                            frame_id=candidate.frame_id,
                            source=candidate.source,
                            timestamp=candidate.timestamp,
                            frame=candidate.frame.copy(),
                            detections=self._clone_detections(candidate.detections),
                            reason="random-sample",
                        ),
                    )

        if state_now != "band":
            if state_now != "high":
                self._last_band_candidate = None

        self._export_state = state_now

    def list_snapshots(self) -> list[DetectionSnapshot]:
        with self._snapshots_lock:
            return list(self._snapshots)

    def _snapshot_to_buffer(self, snapshot: DetectionSnapshot) -> None:
        with self._snapshots_lock:
            self._snapshots.append(snapshot)

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                kind, payload = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                if kind == "snapshot":
                    assert isinstance(payload, DetectionSnapshot)
                    self._snapshot_to_buffer(payload)
                elif kind == "export":
                    assert isinstance(payload, _ExportCandidate)
                    self._write_export(payload)
            except Exception as exc:
                self._logger.warning("capture task failed kind=%s error=%s", kind, exc)

    def _unique_base(self, source: str, timestamp: datetime, frame_id: int, reason: str) -> Path:
        assert self._export_dir is not None
        self._export_dir.mkdir(parents=True, exist_ok=True)
        source_stem = Path(source).stem or "frame"
        ts = timestamp.strftime("%Y%m%d-%H%M%S-%f")
        self._export_serial += 1
        base = self._export_dir / f"{source_stem}-{ts}-f{frame_id:08d}-{reason}-{self._export_serial:06d}"
        if not base.with_suffix(".jpg").exists() and not base.with_suffix(".xml").exists():
            return base

        suffix = 1
        while True:
            candidate = self._export_dir / f"{base.name}-{suffix:02d}"
            if not candidate.with_suffix(".jpg").exists() and not candidate.with_suffix(".xml").exists():
                return candidate
            suffix += 1

    def _write_export(self, candidate: _ExportCandidate) -> None:
        if not self._export_enabled or self._export_dir is None:
            return
        if candidate.frame is None:
            return

        base = self._unique_base(
            source=candidate.source,
            timestamp=candidate.timestamp,
            frame_id=candidate.frame_id,
            reason=candidate.reason,
        )
        image_path = base.with_suffix(".jpg")
        xml_path = base.with_suffix(".xml")

        ok = cv2.imwrite(str(image_path), candidate.frame)
        if not ok:
            raise RuntimeError(f"Failed to write image: {image_path}")

        self._write_voc_xml(xml_path, image_path, candidate.frame, candidate.detections)
        self._logger.info(
            "exported frame image=%s xml=%s objects=%d reason=%s",
            image_path,
            xml_path,
            len(candidate.detections),
            candidate.reason,
        )

    def _write_voc_xml(
        self,
        xml_path: Path,
        image_path: Path,
        frame: Any,
        detections: list[Detection],
    ) -> None:
        height, width = frame.shape[:2]
        channels = 1 if len(frame.shape) == 2 else int(frame.shape[2])

        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = str(image_path.parent.name)
        ET.SubElement(root, "filename").text = image_path.name
        ET.SubElement(root, "path").text = str(image_path)
        source = ET.SubElement(root, "source")
        ET.SubElement(source, "database").text = "catDetectionSystem"
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(int(width))
        ET.SubElement(size, "height").text = str(int(height))
        ET.SubElement(size, "depth").text = str(int(channels))
        ET.SubElement(root, "segmented").text = "0"

        for det in detections:
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = str(det.label)
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            ET.SubElement(obj, "confidence").text = f"{det.confidence:.6f}"
            ET.SubElement(obj, "area_pixels").text = str(
                int(det.extra.get("area_pixels", det.width * det.height))
            )
            ET.SubElement(obj, "area_percent").text = f"{float(det.extra.get('area_percent', 0.0)):.6f}"
            bnd = ET.SubElement(obj, "bndbox")
            ET.SubElement(bnd, "xmin").text = str(int(det.x1))
            ET.SubElement(bnd, "ymin").text = str(int(det.y1))
            ET.SubElement(bnd, "xmax").text = str(int(det.x2))
            ET.SubElement(bnd, "ymax").text = str(int(det.y2))

        tree = ET.ElementTree(root)
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
