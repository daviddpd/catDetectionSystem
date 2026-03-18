from __future__ import annotations

import logging
import os
import platform
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from cds.config.models import RuntimeConfig
from cds.detector import select_backend
from cds.detector.models import ModelSpec
from cds.io.ingest import probe_decoder_path, select_ingest_backend
from cds.io.output import DetectionsGallerySink, DisplaySink, JsonEventSink, MjpegSink
from cds.monitoring import PeriodicStatsLogger, RuntimeIdentity, RuntimeMetrics
from cds.pipeline.annotate import draw_overlays
from cds.pipeline.detection_capture import DetectionCaptureManager
from cds.pipeline.frame_queue import LatestFrameQueue
from cds.pipeline.snapshot_gate import WindowSnapshotGate
from cds.triggers import TriggerManager
from cds.types import Detection, FramePacket
from cds.utils import redact_uri_password
from cds.utils.window_layout import place_single_window, place_windows_side_by_side

_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm"}


def _is_video_source(source: str) -> bool:
    if source.startswith(("rtsp://", "rtsps://", "http://", "https://", "udp://", "tcp://")):
        return True
    try:
        return Path(source).suffix.lower() in _VIDEO_EXTENSIONS
    except Exception:
        return False


def _local_gui_available() -> bool:
    system = platform.system().lower()
    if system == "linux":
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def _attach_detection_area_metrics(frame, detections: list[Detection]) -> None:
    try:
        height, width = frame.shape[:2]
        frame_area = max(1, int(height) * int(width))
    except Exception:
        frame_area = 1
    for det in detections:
        area_pixels = max(0, det.width * det.height)
        det.extra["area_pixels"] = int(area_pixels)
        det.extra["area_percent"] = float((100.0 * area_pixels) / frame_area)


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

        source_mode = ingest.source_mode()
        is_live_source = source_mode == "live-stream"
        is_file_like_source = source_mode in {"video-file", "directory"}

        benchmark_requested = bool(self._config.ingest.benchmark)
        benchmark_active = benchmark_requested and is_file_like_source
        if benchmark_requested and not is_file_like_source:
            self._logger.warning(
                "benchmark mode requested but source_mode=%s is not file-like; ignoring benchmark",
                source_mode,
            )

        clock_mode_requested = self._config.ingest.clock_mode
        if benchmark_active:
            clock_mode_effective = "asfast"
        elif clock_mode_requested == "auto":
            clock_mode_effective = "source"
        else:
            clock_mode_effective = clock_mode_requested

        source_clock_enabled = clock_mode_effective == "source" and not benchmark_active
        sample_interval = (
            (1.0 / max(0.1, self._config.ingest.rate_limit_fps))
            if self._config.ingest.rate_limit_fps and not benchmark_active
            else None
        )
        if benchmark_active and self._config.ingest.rate_limit_fps:
            self._logger.warning(
                "benchmark mode ignores rate_limit_fps=%s",
                self._config.ingest.rate_limit_fps,
            )

        frame_queue: LatestFrameQueue[FramePacket] = LatestFrameQueue(
            maxsize=self._config.ingest.queue_size,
            drop_oldest=not benchmark_active,
        )
        infer_queue: LatestFrameQueue[_InferPacket] = LatestFrameQueue(
            maxsize=self._config.ingest.queue_size,
            drop_oldest=not benchmark_active,
        )
        render_queue: LatestFrameQueue[_InferPacket] = LatestFrameQueue(
            maxsize=self._config.ingest.queue_size,
            drop_oldest=True,
        )
        stop_event = threading.Event()
        ingest_eof = threading.Event()
        infer_eof = threading.Event()
        detect_eof = threading.Event()
        source_shape_logged = threading.Event()
        effective_input_logged = threading.Event()
        local_gui_enabled = (not self._config.output.headless) and _local_gui_available()

        display_sink = None
        remote_sink = None

        if not self._config.output.headless and not local_gui_enabled:
            self._logger.warning(
                "local GUI output disabled because no display session is available; continuing without local windows"
            )

        if local_gui_enabled:
            display_sink = DisplaySink(window_name=self._config.output.window_name)
            display_sink.open()
            if not place_single_window(display_sink.window_name):
                self._logger.debug(
                    "single window placement unavailable; using platform defaults"
                )

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
        detections_window_enabled = (
            local_gui_enabled
            and bool(self._config.output.detections_window_enabled)
        )
        macos_detections_window_scale = None
        if platform.system().lower() == "darwin":
            configured = self._config.output.detections_window_macos_scale
            macos_detections_window_scale = configured if configured is not None else 0.25

        default_detect_on = max(
            1,
            int(trigger_config.audio.frames_detect_on if trigger_config.audio.enabled else 1),
            int(trigger_config.hooks.frames_detect_on if trigger_config.hooks.enabled else 1),
        )
        window_detect_on = max(
            1,
            int(trigger_config.audio.frames_detect_on),
            int(trigger_config.hooks.frames_detect_on),
        )
        window_detect_off = max(
            1,
            int(
                trigger_config.audio.frames_detect_off
                if trigger_config.audio.frames_detect_off is not None
                else max(1, int(trigger_config.audio.frames_detect_on) // 2)
            ),
            int(
                trigger_config.hooks.frames_detect_off
                if trigger_config.hooks.frames_detect_off is not None
                else max(1, int(trigger_config.hooks.frames_detect_on) // 2)
            ),
        )
        window_min_area_pixels = max(
            0,
            int(trigger_config.audio.min_area_pixels),
            int(trigger_config.hooks.min_area_pixels),
        )
        window_min_area_percent = max(
            0.0,
            float(trigger_config.audio.min_area_percent),
            float(trigger_config.hooks.min_area_percent),
        )
        capture_buffer_frames = int(self._config.output.detections_buffer_frames or 0)
        if capture_buffer_frames <= 0:
            baseline_detect_on = default_detect_on
            if detections_window_enabled and not triggers_enabled:
                baseline_detect_on = max(baseline_detect_on, window_detect_on)
            capture_buffer_frames = max(1, baseline_detect_on * 3)

        export_frames_requested = bool(self._config.output.export_frames)
        export_frames_active = bool(export_frames_requested and benchmark_active)
        if export_frames_requested and not benchmark_active:
            self._logger.warning(
                "export_frames requested but requires --benchmark on file/directory sources; disabling"
            )

        capture_manager: DetectionCaptureManager | None = None
        if detections_window_enabled or export_frames_active:
            capture_manager = DetectionCaptureManager(
                buffer_frames=capture_buffer_frames,
                export_enabled=export_frames_active,
                export_dir=self._config.output.export_frames_dir,
                export_sample_percent=self._config.output.export_frames_sample_percent,
                confidence_low=self._config.model.confidence,
                confidence_min=self._config.model.confidence_min,
            )
            capture_manager.start()

        detections_gallery = None
        if capture_manager is not None and detections_window_enabled:
            detections_gallery = DetectionsGallerySink(
                window_name=self._config.output.detections_window_name,
                slots=self._config.output.detections_window_slots,
                scale=self._config.output.detections_window_scale,
            )
            detections_gallery.open()
            if display_sink is not None:
                if not place_windows_side_by_side(
                    display_sink.window_name,
                    detections_gallery.window_name,
                    right_window_scale=(
                        macos_detections_window_scale
                        if macos_detections_window_scale is not None
                        else 1.0
                    ),
                ):
                    self._logger.debug(
                        "window side-by-side placement unavailable; using platform defaults"
                    )

        event_queue: LatestFrameQueue[_EventPacket] | None = None
        if event_sink_enabled:
            event_queue = LatestFrameQueue(maxsize=128)

        model_path = model_spec.model_path or ""
        if model_path:
            model_format = Path(model_path).suffix.lower() or "unknown"
        elif model_spec.cfg_path or model_spec.weights_path:
            model_format = "darknet"
        else:
            model_format = "unknown"
        self._logger.info(
            "perf config model_path=%s model_format=%s configured_imgsz=%d backend=%s device=%s ingest=%s source_mode=%s clock=%s benchmark=%s queue_size=%d queue_policy=%s rate_limit_fps=%s sample_interval_s=%s display=%s detections_window=%s detections_buffer_frames=%d export_frames=%s remote_mjpeg=%s events=%s triggers=%s macos_det_window_scale=%s",
            model_path,
            model_format,
            self._config.model.imgsz,
            backend.name(),
            backend.device_info(),
            ingest.name(),
            source_mode,
            clock_mode_effective,
            benchmark_active,
            self._config.ingest.queue_size,
            ("no-drop" if benchmark_active else "latest-wins"),
            self._config.ingest.rate_limit_fps,
            sample_interval,
            display_sink is not None,
            detections_gallery is not None,
            capture_buffer_frames,
            export_frames_active,
            remote_sink is not None,
            event_sink_enabled,
            triggers_enabled,
            macos_detections_window_scale,
        )

        snapshot_episode_active = False
        fallback_snapshot_gate: WindowSnapshotGate | None = None
        if detections_window_enabled and not triggers_enabled:
            fallback_snapshot_gate = WindowSnapshotGate(
                on_frames=window_detect_on,
                off_frames=window_detect_off,
                min_area_pixels=window_min_area_pixels,
                min_area_percent=window_min_area_percent,
            )
            self._logger.info(
                "detections window gate active without triggers on_frames=%d off_frames=%d min_area_px=%d min_area_pct=%.3f",
                window_detect_on,
                window_detect_off,
                window_min_area_pixels,
                window_min_area_percent,
            )

        def _enqueue_with_policy(queue_obj: LatestFrameQueue, item: object) -> int:
            if not benchmark_active:
                return queue_obj.put_latest(item)

            while not stop_event.is_set():
                result = queue_obj.put_latest(item, block=True, timeout=0.1)
                if result == 0:
                    return 0
            return 1

        def ingest_loop() -> None:
            none_count = 0
            last_sample_emit = 0.0
            source_clock_warned = False
            source_clock_started = False
            source_clock_source = ""
            source_clock_fps = 0.0
            source_clock_start = 0.0
            source_clock_frames = 0
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
                            redact_uri_password(packet.source),
                        )
                        source_shape_logged.set()
                    except Exception:
                        pass

                none_count = 0
                metrics.mark_decode()

                now = time.monotonic()
                if source_clock_enabled and _is_video_source(packet.source):
                    nominal_fps = ingest.nominal_fps()
                    if not nominal_fps or nominal_fps <= 0:
                        nominal_fps = 30.0
                        if not source_clock_warned:
                            source_clock_warned = True
                            self._logger.warning(
                                "source clock requested but source FPS unavailable; defaulting to %.1f",
                                nominal_fps,
                            )

                    if (
                        not source_clock_started
                        or packet.source != source_clock_source
                        or abs(nominal_fps - source_clock_fps) > 0.01
                    ):
                        source_clock_started = True
                        source_clock_source = packet.source
                        source_clock_fps = nominal_fps
                        source_clock_start = now
                        source_clock_frames = 0
                        self._logger.info(
                            "source clock active fps=%.3f source=%s",
                            source_clock_fps,
                            redact_uri_password(source_clock_source),
                        )

                    target_time = source_clock_start + (source_clock_frames / source_clock_fps)
                    if now < target_time:
                        time.sleep(target_time - now)
                        now = time.monotonic()
                    source_clock_frames += 1
                elif source_clock_started and packet.source != source_clock_source:
                    source_clock_started = False
                    source_clock_source = ""
                    source_clock_frames = 0

                if sample_interval is not None:
                    if (now - last_sample_emit) < sample_interval:
                        metrics.add_sampled_out(1)
                        continue
                    last_sample_emit = now

                dropped = _enqueue_with_policy(frame_queue, packet)
                if stop_event.is_set():
                    return
                metrics.mark_ingest()
                metrics.set_queue_depth(frame_queue.qsize())
                if dropped:
                    metrics.add_dropped(dropped)

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
                if not effective_input_logged.is_set():
                    profile_getter = getattr(backend, "runtime_input_profile", None)
                    if callable(profile_getter):
                        try:
                            profile = profile_getter()
                        except Exception:
                            profile = None
                        if profile:
                            self._logger.info(
                                "effective model input height=%s width=%s format=%s dtype=%s batch=%s scale=%s color=%s backend=%s",
                                profile.get("height"),
                                profile.get("width"),
                                profile.get("format"),
                                profile.get("dtype"),
                                profile.get("batch"),
                                profile.get("scale"),
                                profile.get("color"),
                                backend.name(),
                            )
                            effective_input_logged.set()
                metrics.mark_infer()

                dropped = _enqueue_with_policy(
                    infer_queue,
                    _InferPacket(packet=packet, detections=detections),
                )
                if stop_event.is_set():
                    infer_eof.set()
                    return
                if dropped:
                    metrics.add_dropped(dropped)

        def detect_loop() -> None:
            nonlocal snapshot_episode_active
            while True:
                if stop_event.is_set() and infer_queue.empty():
                    detect_eof.set()
                    return

                infer_packet = infer_queue.get(timeout=0.1)
                if infer_packet is None:
                    if infer_eof.is_set() and infer_queue.empty():
                        detect_eof.set()
                        return
                    continue

                packet = infer_packet.packet
                detections = infer_packet.detections
                frame_age_ms = max(
                    0.0,
                    (time.monotonic() - packet.created_monotonic_s) * 1000.0,
                )
                metrics.set_frame_age_ms(frame_age_ms)

                _attach_detection_area_metrics(packet.frame, detections)

                trigger_result = triggers.process(packet, detections, backend.name())
                if capture_manager is not None:
                    activated_detections = trigger_result.activated_detections
                    any_active = trigger_result.any_active

                    if fallback_snapshot_gate is not None:
                        activated_detections, any_active = fallback_snapshot_gate.observe(detections)

                    if activated_detections and not snapshot_episode_active:
                        capture_manager.submit_activation_snapshot(
                            packet=packet,
                            frame=packet.frame,
                            detections=detections,
                            activated_detections=activated_detections,
                        )
                        snapshot_episode_active = True
                    if not any_active:
                        snapshot_episode_active = False

                    if export_frames_active and capture_manager.export_enabled():
                        capture_manager.observe_benchmark_export(
                            packet=packet,
                            frame=packet.frame,
                            detections=detections,
                        )

                dropped = render_queue.put_latest(
                    _InferPacket(packet=packet, detections=detections),
                )
                if dropped:
                    self._logger.debug(
                        "render queue saturated dropped=%d frame_id=%d",
                        dropped,
                        packet.frame_id,
                    )

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
                    if detect_eof.is_set() and event_queue.empty():
                        return
                    continue

                if event_sink_enabled:
                    timestamp = event_packet.timestamp.astimezone(timezone.utc).isoformat()
                    redacted_source = redact_uri_password(event_packet.source)
                    for detection in event_packet.detections:
                        event_sink.emit(
                            {
                                "ts": timestamp,
                                "frame_id": event_packet.frame_id,
                                "source": redacted_source,
                                "label": detection.label,
                                "class_id": detection.class_id,
                                "confidence": detection.confidence,
                                "area_pixels": int(detection.extra.get("area_pixels", detection.width * detection.height)),
                                "area_percent": float(detection.extra.get("area_percent", 0.0)),
                                "bbox": [
                                    detection.x1,
                                    detection.y1,
                                    detection.x2,
                                    detection.y2,
                                ],
                                "backend": event_packet.backend_name,
                            }
                        )

        def _run_worker(
            name: str,
            target: Callable[[], None],
            eof_event: threading.Event | None = None,
        ) -> None:
            try:
                target()
            except Exception:
                self._logger.exception("%s thread failed", name)
                if eof_event is not None:
                    eof_event.set()
                stop_event.set()

        ingest_thread = threading.Thread(
            target=lambda: _run_worker("cds-ingest", ingest_loop, ingest_eof),
            name="cds-ingest",
            daemon=True,
        )
        infer_thread = threading.Thread(
            target=lambda: _run_worker("cds-infer", infer_loop, infer_eof),
            name="cds-infer",
            daemon=True,
        )
        detect_thread = threading.Thread(
            target=lambda: _run_worker("cds-detect-logic", detect_loop, detect_eof),
            name="cds-detect-logic",
            daemon=True,
        )
        event_thread = threading.Thread(
            target=lambda: _run_worker("cds-events", event_loop, None),
            name="cds-events",
            daemon=True,
        )
        ingest_thread.start()
        infer_thread.start()
        detect_thread.start()
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
                render_packet = render_queue.get(timeout=0.1)
                if render_packet is None:
                    if detect_eof.is_set() and render_queue.empty():
                        break
                    stats_logger.maybe_emit()
                    continue

                packet = render_packet.packet
                detections = render_packet.detections

                if render_enabled:
                    snapshot = metrics.snapshot()
                    draw_overlays(packet.frame, detections, backend.name(), snapshot.fps_infer)

                last_display_key = -1
                if display_sink is not None:
                    should_continue = display_sink.write(packet.frame)
                    last_display_key = display_sink.consume_key()
                    if not should_continue:
                        stop_event.set()
                        break

                if detections_gallery is not None and capture_manager is not None:
                    detections_gallery.render(
                        snapshots=capture_manager.list_snapshots(),
                        key=last_display_key,
                    )

                if remote_sink is not None:
                    remote_sink.write(packet.frame)

                stats_logger.maybe_emit()

            return 0
        finally:
            stop_event.set()
            ingest_thread.join(timeout=2)
            infer_thread.join(timeout=2)
            detect_thread.join(timeout=2)
            event_thread.join(timeout=2)
            ingest.close()
            triggers.close()
            event_sink.close()
            if detections_gallery is not None:
                detections_gallery.close()
            if capture_manager is not None:
                capture_manager.close()
            if remote_sink is not None:
                remote_sink.close()
            if display_sink is not None:
                display_sink.close()


def runtime_summary(config: RuntimeConfig) -> dict:
    return asdict(config)
