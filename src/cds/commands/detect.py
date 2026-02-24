from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from cds.config import load_runtime_config
from cds.monitoring import configure_logging


def _clean_overrides(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            nested = _clean_overrides(value)
            if nested:
                cleaned[key] = nested
            continue
        if value is not None:
            cleaned[key] = value
    return cleaned


def _parse_pyav_options(raw_items: list[str] | None) -> dict[str, str] | None:
    if not raw_items:
        return None
    parsed: dict[str, str] = {}
    for item in raw_items:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed if parsed else None


def build_detect_overrides(args: Any) -> dict[str, Any]:
    pyav_options = _parse_pyav_options(args.pyav_option)

    overrides = {
        "model": {
            "name": args.model_name,
            "path": args.model_path,
            "cfg_path": args.cfg_path,
            "weights_path": args.weights_path,
            "labels_path": args.labels_path,
            "confidence": args.confidence,
            "confidence_min": args.confidence_min,
            "nms": args.nms,
            "imgsz": args.imgsz,
            "class_filter": args.class_filter,
        },
        "backend_policy": {
            "requested": args.backend,
        },
        "ingest": {
            "uri": args.uri,
            "backend": args.ingest_backend,
            "queue_size": args.queue_size,
            "rate_limit_fps": args.rate_limit_fps,
            "clock_mode": args.clock,
            "benchmark": (True if args.benchmark else None),
            "gstreamer_pipeline": args.gstreamer_pipeline,
            "pyav_options": pyav_options,
        },
        "output": {
            "headless": args.headless,
            "window_name": args.window_name,
            "export_frames": (True if args.export_frames else None),
            "export_frames_dir": args.export_frames_dir,
            "export_frames_sample_percent": args.export_frames_sample_pct,
            "remote_enabled": args.remote_mjpeg,
            "remote_host": args.remote_host,
            "remote_port": args.remote_port,
            "remote_path": args.remote_path,
        },
        "monitoring": {
            "json_logs": args.json_logs,
            "log_level": args.log_level,
            "prometheus_enabled": args.prometheus,
            "prometheus_host": args.prometheus_host,
            "prometheus_port": args.prometheus_port,
            "event_stdout": (False if args.no_event_stdout else None),
            "event_file": args.event_file,
        },
        "stress_sleep_ms": args.stress_sleep_ms,
    }
    return _clean_overrides(overrides)


def run_detect(args: Any, repo_root: Path) -> int:
    config = load_runtime_config(
        repo_root=repo_root,
        config_path=args.config,
        cli_overrides=build_detect_overrides(args),
    )

    if config.output.headless:
        config.monitoring.event_stdout = False
        config.output.remote_enabled = False
        if args.log_level is None and not args.quiet:
            config.monitoring.log_level = "WARNING"

    if args.quiet:
        config.monitoring.log_level = "WARNING"

    configure_logging(
        level=config.monitoring.log_level,
        json_logs=config.monitoring.json_logs,
    )

    logger = logging.getLogger("cds.detect")
    logger.info("starting detect with config=%s", config.as_log_context())

    try:
        from cds.pipeline.runtime import DetectionRuntime

        runtime = DetectionRuntime(repo_root=repo_root, config=config)
        return runtime.run()
    except ModuleNotFoundError as exc:
        logger.error(
            "missing dependency: %s. Install requirements before running detect.",
            exc.name,
        )
        return 2
