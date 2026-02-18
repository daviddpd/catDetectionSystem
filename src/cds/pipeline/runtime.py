from __future__ import annotations

import logging
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from cds.config.models import RuntimeConfig
from cds.detector import select_backend
from cds.detector.models import ModelSpec
from cds.io.ingest import probe_decoder_path, select_ingest_backend
from cds.io.output import DisplaySink, JsonEventSink, MjpegSink
from cds.monitoring import PeriodicStatsLogger, RuntimeIdentity, RuntimeMetrics
from cds.pipeline.annotate import draw_overlays
from cds.pipeline.frame_queue import LatestFrameQueue
from cds.triggers import TriggerManager
from cds.types import Detection, FramePacket


@dataclass
class _InferPacket:
    packet: FramePacket
    detections: list[Detection]


@dataclass
class _EventPacket:
    timestamp: datetime
    frame_id: int
    source: str
    backend_name: str
    detections: list[Detection]


class DetectionRuntime:
    def __init__(self, repo_root: Path, config: RuntimeConfig) -> None:
        self._repo_root = repo_root
        self._config = config
        self._logger = logging.getLogger("cds.runtime")

    def _build_model_spec(self) -> ModelSpec:
        return ModelSpec(
            name=self._config.model.name,
            model_path=self._config.model.path,
            cfg_path=self._config.model.cfg_path,
            weights_path=self._config.model.weights_path,
            labels_path=self._config.model.labels_path,
            confidence=self._config.model.confidence,
            nms=self._config.model.nms,
            imgsz=self._config.model.imgsz,
            class_filter=set(self._config.model.class_filter),
        )

    def run(self) -> int:
        if not self._config.ingest.uri:
            raise RuntimeError("Missing ingest URI. Provide --uri or ingest.uri in config.")

        decoder_probe = probe_decoder_path()
        self._logger.info(
            "decoder selected=%s reason=%s available=%s",
            decoder_probe.selected_decoder,
            decoder_probe.reason,
            ",".join(decoder_probe.available),
        )

        model_spec = self._build_model_spec()
        backend_selection = select_backend(model_spec, self._config.backend_policy)
        backend = backend_selection.backend
        self._logger.info(
            "backend selected=%s reason=%s device=%s",
            backend.name(),
            backend_selection.reason,
            backend.device_info(),
        )

        backend.warmup()

        ingest, ingest_reason = select_ingest_backend(self._config.ingest)
        self._logger.info(
            "ingest selected=%s reason=%s",
            ingest.name(),
            ingest_reason,
        )
        ingest.open(
            self._config.ingest.uri,
            {
                "pyav_options": self._config.ingest.pyav_options,
                "gstreamer_pipeline": self._config.ingest.gstreamer_pipeline,
            },
        )

        metrics = RuntimeMetrics()
        if self._config.monitoring.prometheus_enabled:
            enabled = metrics.enable_prometheus(
                self._config.monitoring.prometheus_host,
                self._config.monitoring.prometheus_port,
            )
            if enabled:
                self._logger.info(
                    "prometheus endpoint enabled at %s:%d",
                    self._config.monitoring.prometheus_host,
                    self._config.monitoring.prometheus_port,
                )
            else:
                self._logger.warning(
                    "prometheus requested but prometheus_client is not installed"
                )

        frame_queue: LatestFrameQueue[FramePacket] = LatestFrameQueue(
            maxsize=self._config.ingest.queue_size
        )
        infer_queue: LatestFrameQueue[_InferPacket] = LatestFrameQueue(
            maxsize=self._config.ingest.queue_size
        )
        stop_event = threading.Event()
        ingest_eof = threading.Event()
        infer_eof = threading.Event()
        source_shape_logged = threading.Event()

        is_live_source = str(self._config.ingest.uri).startswith(("rtsp://", "http://", "https://"))

        display_sink = None
        remote_sink = None

        if not self._config.output.headless:
            display_sink = DisplaySink(window_name=self._config.output.window_name)
            display_sink.open()

            if self._config.output.remote_enabled:
                remote_sink = MjpegSink(
                    host=self._config.output.remote_host,
                    port=self._config.output.remote_port,
                    path=self._config.output.remote_path,
                )
                remote_sink.open()
                self._logger.info("remote sink enabled endpoint=%s", remote_sink.endpoint_url)

        event_sink = JsonEventSink(
            stdout_enabled=self._config.monitoring.event_stdout,
            file_path=self._config.monitoring.event_file,
        )
        event_sink.open()

        trigger_config = deepcopy(self._config.triggers)
        if self._config.output.headless:
            trigger_config.audio.enabled = False
        triggers = TriggerManager(trigger_config)

        event_sink_enabled = event_sink.enabled()
        triggers_enabled = triggers.enabled()
        event_queue: LatestFrameQueue[_EventPacket] | None = None
        if event_sink_enabled or triggers_enabled:
            event_queue = LatestFrameQueue(maxsize=128)

        model_path = model_spec.model_path or ""
        if model_path:
            model_format = Path(model_path).suffix.lower() or "unknown"
        elif model_spec.cfg_path or model_spec.weights_path:
            model_format = "darknet"
        else:
            model_format = "unknown"
        self._logger.info(
            "perf config model_path=%s model_format=%s imgsz=%d backend=%s device=%s ingest=%s queue_size=%d rate_limit_fps=%s display=%s remote_mjpeg=%s events=%s triggers=%s",
            model_path,
            model_format,
            self._config.model.imgsz,
            backend.name(),
            backend.device_info(),
            ingest.name(),
            self._config.ingest.queue_size,
            self._config.ingest.rate_limit_fps,
            display_sink is not None,
            remote_sink is not None,
            event_sink_enabled,
            triggers_enabled,
        )

        def ingest_loop() -> None:
            none_count = 0
            last_emit = time.monotonic()
            while not stop_event.is_set():
                packet = ingest.read_latest()
                if packet is None:
                    none_count += 1
                    if not is_live_source and none_count >= 5:
                        ingest_eof.set()
                        return
                    time.sleep(0.01)
                    continue

                if not source_shape_logged.is_set():
                    try:
                        height, width = packet.frame.shape[:2]
                        self._logger.info(
                            "source dimensions width=%d height=%d source=%s",
                            width,
                            height,
                            packet.source,
                        )
                        source_shape_logged.set()
                    except Exception:
                        pass

                none_count = 0
                dropped = frame_queue.put_latest(packet)
                metrics.mark_ingest()
                metrics.set_queue_depth(frame_queue.qsize())
                if dropped:
                    metrics.add_dropped(dropped)

                if self._config.ingest.rate_limit_fps:
                    interval = 1.0 / max(0.1, self._config.ingest.rate_limit_fps)
                    elapsed = time.monotonic() - last_emit
                    if elapsed < interval:
                        time.sleep(interval - elapsed)
                    last_emit = time.monotonic()

        def infer_loop() -> None:
            while True:
                if stop_event.is_set() and frame_queue.empty():
                    infer_eof.set()
                    return

                packet = frame_queue.get(timeout=0.1)
                metrics.set_queue_depth(frame_queue.qsize())
                if packet is None:
                    if ingest_eof.is_set() and frame_queue.empty():
                        infer_eof.set()
                        return
                    continue

                if self._config.stress_sleep_ms > 0:
                    time.sleep(self._config.stress_sleep_ms / 1000.0)

                detections = backend.infer(packet.frame)
                metrics.mark_infer()

                dropped = infer_queue.put_latest(
                    _InferPacket(packet=packet, detections=detections)
                )
                if dropped:
                    metrics.add_dropped(dropped)

                if event_queue is not None:
                    dropped_event = event_queue.put_latest(
                        _EventPacket(
                            timestamp=packet.timestamp,
                            frame_id=packet.frame_id,
                            source=packet.source,
                            backend_name=backend.name(),
                            detections=detections,
                        )
                    )
                    if dropped_event:
                        self._logger.debug(
                            "event queue saturated dropped=%d frame_id=%d",
                            dropped_event,
                            packet.frame_id,
                        )

        def event_loop() -> None:
            if event_queue is None:
                return

            while True:
                if stop_event.is_set() and event_queue.empty():
                    return

                event_packet = event_queue.get(timeout=0.1)
                if event_packet is None:
                    if infer_eof.is_set() and event_queue.empty():
                        return
                    continue

                if event_sink_enabled:
                    timestamp = event_packet.timestamp.astimezone(timezone.utc).isoformat()
                    for detection in event_packet.detections:
                        event_sink.emit(
                            {
                                "ts": timestamp,
                                "frame_id": event_packet.frame_id,
                                "source": event_packet.source,
                                "label": detection.label,
                                "class_id": detection.class_id,
                                "confidence": detection.confidence,
                                "bbox": [
                                    detection.x1,
                                    detection.y1,
                                    detection.x2,
                                    detection.y2,
                                ],
                                "backend": event_packet.backend_name,
                            }
                        )

                if triggers_enabled:
                    triggers.process(
                        FramePacket(
                            frame_id=event_packet.frame_id,
                            frame=None,
                            source=event_packet.source,
                            timestamp=event_packet.timestamp,
                        ),
                        event_packet.detections,
                        event_packet.backend_name,
                    )

        ingest_thread = threading.Thread(target=ingest_loop, name="cds-ingest", daemon=True)
        infer_thread = threading.Thread(target=infer_loop, name="cds-infer", daemon=True)
        event_thread = threading.Thread(target=event_loop, name="cds-events", daemon=True)
        ingest_thread.start()
        infer_thread.start()
        event_thread.start()

        stats_logger = PeriodicStatsLogger(
            metrics=metrics,
            identity=RuntimeIdentity(
                backend=backend.name(),
                decoder=decoder_probe.selected_decoder,
            ),
            interval_seconds=self._config.monitoring.stats_interval_seconds,
        )

        try:
            render_enabled = display_sink is not None or remote_sink is not None
            while not stop_event.is_set():
                infer_packet = infer_queue.get(timeout=0.1)
                if infer_packet is None:
                    if infer_eof.is_set() and infer_queue.empty():
                        break
                    stats_logger.maybe_emit()
                    continue

                packet = infer_packet.packet
                detections = infer_packet.detections

                if render_enabled:
                    snapshot = metrics.snapshot()
                    draw_overlays(packet.frame, detections, backend.name(), snapshot.fps_infer)

                if display_sink is not None:
                    should_continue = display_sink.write(packet.frame)
                    if not should_continue:
                        stop_event.set()
                        break

                if remote_sink is not None:
                    remote_sink.write(packet.frame)

                stats_logger.maybe_emit()

            return 0
        finally:
            stop_event.set()
            ingest_thread.join(timeout=2)
            infer_thread.join(timeout=2)
            event_thread.join(timeout=2)
            ingest.close()
            triggers.close()
            event_sink.close()
            if remote_sink is not None:
                remote_sink.close()
            if display_sink is not None:
                display_sink.close()


def runtime_summary(config: RuntimeConfig) -> dict:
    return asdict(config)
