from __future__ import annotations

import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from cds.io.ingest.base import VideoIngest
from cds.types import FramePacket

_GST_LOCK = threading.Lock()
_GST_BINDINGS: Any | None = None
_APPSINK_DEFAULT_NAME = "cdsappsink"


def _load_gst_bindings() -> Any:
    global _GST_BINDINGS
    if _GST_BINDINGS is not None:
        return _GST_BINDINGS

    with _GST_LOCK:
        if _GST_BINDINGS is not None:
            return _GST_BINDINGS
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
        except Exception as exc:
            raise RuntimeError(
                "PyGObject GStreamer bindings are unavailable. Install python3-gi and "
                "gir1.2-gstreamer-1.0 (or use a venv created with --system-site-packages)."
            ) from exc

        Gst.init(None)
        _GST_BINDINGS = Gst
        return _GST_BINDINGS


def _normalize_appsink_pipeline(pipeline: str) -> tuple[str, str]:
    segments = [segment.strip() for segment in pipeline.split("!")]
    if not segments:
        raise RuntimeError("GStreamer pipeline is empty.")

    sink_segment = segments[-1]
    if not re.match(r"^appsink(?:\s|$)", sink_segment):
        raise RuntimeError(
            "GStreamer ingest requires a pipeline ending in appsink so frames can be pulled "
            "into CDS."
        )

    name_match = re.search(r"\bname=([A-Za-z0-9_-]+)", sink_segment)
    sink_name = name_match.group(1) if name_match else _APPSINK_DEFAULT_NAME
    if not name_match:
        sink_segment = f"{sink_segment} name={sink_name}"
        segments[-1] = sink_segment

    return " ! ".join(segments), sink_name


def _sample_to_bgr_frame(sample: Any) -> tuple[np.ndarray, float | None]:
    caps = sample.get_caps()
    if caps is None or caps.get_size() <= 0:
        raise RuntimeError("GStreamer sample is missing caps metadata.")

    structure = caps.get_structure(0)
    width = int(structure.get_value("width"))
    height = int(structure.get_value("height"))
    pixel_format = str(structure.get_value("format"))
    if pixel_format != "BGR":
        raise RuntimeError(
            f"GStreamer appsink produced unsupported format={pixel_format}. Expected BGR."
        )

    nominal_fps = None
    if structure.has_field("framerate"):
        fps_value = structure.get_value("framerate")
        num = getattr(fps_value, "num", None)
        den = getattr(fps_value, "denom", None)
        if num is None and isinstance(fps_value, tuple) and len(fps_value) == 2:
            num, den = fps_value
        if isinstance(num, int) and isinstance(den, int) and den > 0:
            nominal_fps = float(num) / float(den)

    buffer = sample.get_buffer()
    if buffer is None:
        raise RuntimeError("GStreamer sample is missing a buffer.")

    gst = _load_gst_bindings()
    ok, map_info = buffer.map(gst.MapFlags.READ)
    if not ok:
        raise RuntimeError("Failed to map GStreamer buffer for reading.")

    try:
        payload = np.frombuffer(map_info.data, dtype=np.uint8)
        row_width = width * 3
        expected = row_width * height
        if payload.size < expected:
            raise RuntimeError(
                f"GStreamer sample payload too small size={payload.size} expected>={expected}."
            )

        if payload.size == expected:
            frame = payload.reshape((height, width, 3)).copy()
            return frame, nominal_fps

        row_stride = payload.size // height
        if row_stride < row_width:
            raise RuntimeError(
                f"GStreamer sample row stride too small stride={row_stride} min={row_width}."
            )
        rows = payload[: row_stride * height].reshape((height, row_stride))
        frame = rows[:, :row_width].reshape((height, width, 3)).copy()
        return frame, nominal_fps
    finally:
        buffer.unmap(map_info)


class GStreamerIngest(VideoIngest):
    def __init__(self) -> None:
        self._uri = ""
        self._frame_id = 0
        self._source_mode = "live-stream"
        self._nominal_fps: float | None = None
        self._gst: Any | None = None
        self._pipeline: Any | None = None
        self._appsink: Any | None = None
        self._bus: Any | None = None
        self._prefetched_frame: np.ndarray | None = None
        self._eos = False

    def open(self, uri: str, options: dict[str, Any] | None = None) -> None:
        self.close()
        options = options or {}
        raw_pipeline = options.get("gstreamer_pipeline")
        if not isinstance(raw_pipeline, str) or not raw_pipeline.strip():
            raise RuntimeError(
                "GStreamer ingest requires --gstreamer-pipeline (or ingest.gstreamer_pipeline)."
            )

        pipeline_text, sink_name = _normalize_appsink_pipeline(raw_pipeline)
        self._gst = _load_gst_bindings()
        self._uri = uri
        self._frame_id = 0
        self._source_mode = "video-file" if Path(uri).is_file() else "live-stream"
        self._nominal_fps = None
        self._prefetched_frame = None
        self._eos = False

        try:
            self._pipeline = self._gst.parse_launch(pipeline_text)
        except Exception as exc:
            raise RuntimeError(f"Failed to parse GStreamer pipeline: {exc}") from exc

        self._appsink = self._pipeline.get_by_name(sink_name)
        if self._appsink is None:
            self.close()
            raise RuntimeError(
                "Failed to locate appsink in the parsed GStreamer pipeline. "
                "Ensure the pipeline ends with appsink."
            )

        for prop, value in (
            ("emit-signals", False),
            ("sync", False),
            ("drop", True),
            ("max-buffers", 1),
        ):
            try:
                self._appsink.set_property(prop, value)
            except Exception:
                pass

        self._bus = self._pipeline.get_bus()
        result = self._pipeline.set_state(self._gst.State.PLAYING)
        if result == self._gst.StateChangeReturn.FAILURE:
            self.close()
            raise RuntimeError("Failed to start the GStreamer pipeline.")

        if self._source_mode == "video-file":
            first_frame = self._pull_frame(timeout_ns=5 * int(getattr(self._gst, "SECOND", 1_000_000_000)))
            if first_frame is None:
                self.close()
                raise RuntimeError(
                    "GStreamer pipeline started, but no frame was delivered to appsink. "
                    "The pipeline may work in gst-launch while still being incompatible "
                    "with Python appsink capture."
                )
            self._prefetched_frame = first_frame

    def _drain_bus(self) -> None:
        if self._bus is None or self._gst is None:
            return

        mask = self._gst.MessageType.ERROR | self._gst.MessageType.EOS
        while True:
            message = self._bus.timed_pop_filtered(0, mask)
            if message is None:
                return
            if message.type == self._gst.MessageType.EOS:
                self._eos = True
                continue
            if message.type == self._gst.MessageType.ERROR:
                err, debug = message.parse_error()
                detail = getattr(err, "message", str(err))
                if debug:
                    detail = f"{detail} ({debug})"
                raise RuntimeError(f"GStreamer pipeline error: {detail}")

    def _pull_sample(self, timeout_ns: int) -> Any | None:
        if self._appsink is None:
            return None
        try:
            return self._appsink.emit("try-pull-sample", timeout_ns)
        except Exception as exc:
            raise RuntimeError(f"Failed to pull a sample from GStreamer appsink: {exc}") from exc

    def _pull_frame(self, *, timeout_ns: int) -> np.ndarray | None:
        self._drain_bus()
        if self._eos:
            return None

        sample = self._pull_sample(timeout_ns)
        if sample is None:
            self._drain_bus()
            return None

        frame, fps = _sample_to_bgr_frame(sample)
        if fps and fps > 0 and not self._nominal_fps:
            self._nominal_fps = fps
        self._drain_bus()
        return frame

    def read_latest(self) -> FramePacket | None:
        if self._prefetched_frame is not None:
            frame = self._prefetched_frame
            self._prefetched_frame = None
        else:
            timeout_ns = 50_000_000 if self._source_mode == "video-file" else 0
            frame = self._pull_frame(timeout_ns=timeout_ns)
            if frame is None:
                return None

        self._frame_id += 1
        return FramePacket(
            frame_id=self._frame_id,
            frame=frame,
            source=self._uri,
            timestamp=datetime.now(),
        )

    def close(self) -> None:
        self._prefetched_frame = None
        self._eos = False
        if self._pipeline is not None and self._gst is not None:
            try:
                self._pipeline.set_state(self._gst.State.NULL)
            except Exception:
                pass
        self._bus = None
        self._appsink = None
        self._pipeline = None

    def name(self) -> str:
        return "gstreamer"

    def source_mode(self) -> str:
        return self._source_mode

    def nominal_fps(self) -> float | None:
        return self._nominal_fps
