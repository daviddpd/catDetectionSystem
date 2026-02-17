from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cds.config.defaults import DEFAULT_CONFIG
from cds.config.models import (
    AudioTriggerConfig,
    BackendPolicyConfig,
    HookRuleConfig,
    HookTriggerConfig,
    IngestConfig,
    ModelConfig,
    MonitoringConfig,
    OutputConfig,
    RuntimeConfig,
    TriggerConfig,
)


def _merge_dict(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def _lower_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k).lower(): _lower_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_lower_keys(item) for item in obj]
    return obj


def _maybe_parse_simple_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix == ".json":
        return json.loads(text)

    if suffix == ".toml":
        try:
            import tomllib
        except ImportError:  # pragma: no cover - Python < 3.11
            import tomli as tomllib  # type: ignore

        return tomllib.loads(text)

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - fallback path
            raise RuntimeError(
                "YAML configuration requested but PyYAML is not installed. "
                "Install pyyaml or use TOML/JSON."
            ) from exc
        loaded = yaml.safe_load(text)
        return loaded if loaded else {}

    raise RuntimeError(f"Unsupported config extension: {suffix}")


def _load_with_dynaconf(config_paths: list[Path]) -> dict[str, Any]:
    try:
        from dynaconf import Dynaconf
    except ImportError:
        return {}

    settings = Dynaconf(
        envvar_prefix="CDS",
        settings_files=[str(path) for path in config_paths if path.exists()],
        merge_enabled=True,
        environments=False,
        load_dotenv=True,
    )
    return _lower_keys(settings.as_dict())


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _resolve_repo_relative(path_value: str | None, repo_root: Path) -> str | None:
    if not path_value:
        return path_value
    p = Path(path_value)
    if p.is_absolute():
        return str(p)
    return str((repo_root / p).resolve())


def _normalize(data: dict[str, Any], repo_root: Path) -> RuntimeConfig:
    model_data = data.get("model", {})
    backend_data = data.get("backend_policy", {})
    ingest_data = data.get("ingest", {})
    output_data = data.get("output", {})
    trigger_data = data.get("triggers", {})
    monitoring_data = data.get("monitoring", {})

    audio_data = trigger_data.get("audio", {})
    hook_data = trigger_data.get("hooks", {})

    hook_rules = [
        HookRuleConfig(
            classes=list(rule.get("classes", [])),
            command=list(rule.get("command", [])),
            timeout_seconds=float(rule.get("timeout_seconds", 5.0)),
            cooldown_seconds=float(rule.get("cooldown_seconds", 5.0)),
            payload_mode=str(rule.get("payload_mode", "stdin")),
        )
        for rule in hook_data.get("rules", [])
    ]

    audio_map = {
        str(label): _resolve_repo_relative(str(audio_path), repo_root)
        for label, audio_path in audio_data.get("class_to_audio", {}).items()
    }

    config = RuntimeConfig(
        model=ModelConfig(
            name=str(model_data.get("name", "communitycats")),
            path=_resolve_repo_relative(model_data.get("path"), repo_root),
            cfg_path=_resolve_repo_relative(model_data.get("cfg_path"), repo_root),
            weights_path=_resolve_repo_relative(model_data.get("weights_path"), repo_root),
            labels_path=_resolve_repo_relative(
                model_data.get("labels_path", "yolo/cfg/custom-names-v4.txt"),
                repo_root,
            )
            or str((repo_root / "yolo/cfg/custom-names-v4.txt").resolve()),
            confidence=float(model_data.get("confidence", 0.5)),
            nms=float(model_data.get("nms", 0.5)),
            imgsz=int(model_data.get("imgsz", 640)),
            class_filter=[str(v) for v in model_data.get("class_filter", [])],
        ),
        backend_policy=BackendPolicyConfig(
            requested=str(backend_data.get("requested", "auto")).lower(),
            allow_darknet_fallback=_coerce_bool(
                backend_data.get("allow_darknet_fallback", True)
            ),
            allow_rknn=_coerce_bool(backend_data.get("allow_rknn", True)),
            allow_tensorrt=_coerce_bool(backend_data.get("allow_tensorrt", True)),
        ),
        ingest=IngestConfig(
            uri=ingest_data.get("uri"),
            backend=str(ingest_data.get("backend", "auto")).lower(),
            queue_size=max(1, min(2, int(ingest_data.get("queue_size", 2)))),
            rate_limit_fps=(
                float(ingest_data["rate_limit_fps"])
                if ingest_data.get("rate_limit_fps")
                else None
            ),
            gstreamer_pipeline=ingest_data.get("gstreamer_pipeline"),
            pyav_options={
                str(k): str(v) for k, v in ingest_data.get("pyav_options", {}).items()
            },
        ),
        output=OutputConfig(
            headless=_coerce_bool(output_data.get("headless", False)),
            window_name=str(output_data.get("window_name", "catDetectionSystem")),
            remote_enabled=_coerce_bool(output_data.get("remote_enabled", False)),
            remote_host=str(output_data.get("remote_host", "0.0.0.0")),
            remote_port=int(output_data.get("remote_port", 8080)),
            remote_path=str(output_data.get("remote_path", "/stream.mjpg")),
        ),
        triggers=TriggerConfig(
            audio=AudioTriggerConfig(
                enabled=_coerce_bool(audio_data.get("enabled", True)),
                class_to_audio=audio_map,
                cooldown_seconds=float(audio_data.get("cooldown_seconds", 15.0)),
            ),
            hooks=HookTriggerConfig(
                enabled=_coerce_bool(hook_data.get("enabled", False)),
                allowlist=[str(v) for v in hook_data.get("allowlist", [])],
                rules=hook_rules,
                max_workers=max(1, int(hook_data.get("max_workers", 4))),
            ),
        ),
        monitoring=MonitoringConfig(
            json_logs=_coerce_bool(monitoring_data.get("json_logs", False)),
            log_level=str(monitoring_data.get("log_level", "INFO")).upper(),
            stats_interval_seconds=float(
                monitoring_data.get("stats_interval_seconds", 5.0)
            ),
            prometheus_enabled=_coerce_bool(
                monitoring_data.get("prometheus_enabled", False)
            ),
            prometheus_host=str(monitoring_data.get("prometheus_host", "0.0.0.0")),
            prometheus_port=int(monitoring_data.get("prometheus_port", 9108)),
            event_stdout=_coerce_bool(monitoring_data.get("event_stdout", True)),
            event_file=_resolve_repo_relative(
                monitoring_data.get("event_file"),
                repo_root,
            ),
        ),
        stress_sleep_ms=max(0, int(data.get("stress_sleep_ms", 0))),
    )

    return config


def _default_config_copy() -> dict[str, Any]:
    return json.loads(json.dumps(DEFAULT_CONFIG))


def load_runtime_config(
    repo_root: Path,
    config_path: str | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> RuntimeConfig:
    config_paths: list[Path] = []
    if config_path:
        config_paths.append(Path(config_path))
    else:
        for name in (
            "cds.toml",
            "cds.yaml",
            "cds.yml",
            "cds.json",
            "settings.toml",
            "settings.yaml",
            "settings.yml",
            "settings.json",
        ):
            candidate = repo_root / name
            if candidate.exists():
                config_paths.append(candidate)

    merged = _default_config_copy()

    dynaconf_data = _load_with_dynaconf(config_paths)
    if dynaconf_data:
        _merge_dict(merged, dynaconf_data)
    else:
        for path in config_paths:
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")
            _merge_dict(merged, _lower_keys(_maybe_parse_simple_config(path)))

    if cli_overrides:
        _merge_dict(merged, _lower_keys(cli_overrides))

    # Headless hard-disable remote video sinks by requirement.
    if merged.get("output", {}).get("headless", False):
        merged["output"]["remote_enabled"] = False
        merged["monitoring"]["event_stdout"] = False

    config = _normalize(merged, repo_root)

    # Keep compatibility for legacy typo path while preferring canonical spelling.
    if not os.path.isfile(config.model.labels_path):
        legacy = repo_root / "yolo/cfg/custom-names-7v3.txt"
        if legacy.exists():
            config.model.labels_path = str(legacy.resolve())

    return config


def runtime_config_to_dict(config: RuntimeConfig) -> dict[str, Any]:
    return asdict(config)
