from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    name: str = "communitycats"
    path: str | None = None
    cfg_path: str | None = None
    weights_path: str | None = None
    labels_path: str = "yolo/cfg/custom-names-v4.txt"
    confidence: float = 0.5
    nms: float = 0.5
    imgsz: int = 640
    class_filter: list[str] = field(default_factory=list)


@dataclass
class BackendPolicyConfig:
    requested: str = "auto"
    allow_darknet_fallback: bool = True
    allow_rknn: bool = True
    allow_tensorrt: bool = True


@dataclass
class IngestConfig:
    uri: str | None = None
    backend: str = "auto"
    queue_size: int = 2
    rate_limit_fps: float | None = None
    clock_mode: str = "auto"
    benchmark: bool = False
    gstreamer_pipeline: str | None = None
    pyav_options: dict[str, str] = field(default_factory=dict)


@dataclass
class OutputConfig:
    headless: bool = False
    window_name: str = "catDetectionSystem"
    remote_enabled: bool = False
    remote_host: str = "0.0.0.0"
    remote_port: int = 8080
    remote_path: str = "/stream.mjpg"


@dataclass
class AudioTriggerConfig:
    enabled: bool = True
    class_to_audio: dict[str, str] = field(default_factory=dict)
    cooldown_seconds: float = 15.0


@dataclass
class HookRuleConfig:
    classes: list[str] = field(default_factory=list)
    command: list[str] = field(default_factory=list)
    timeout_seconds: float = 5.0
    cooldown_seconds: float = 5.0
    payload_mode: str = "stdin"


@dataclass
class HookTriggerConfig:
    enabled: bool = False
    allowlist: list[str] = field(default_factory=list)
    rules: list[HookRuleConfig] = field(default_factory=list)
    max_workers: int = 4


@dataclass
class TriggerConfig:
    audio: AudioTriggerConfig = field(default_factory=AudioTriggerConfig)
    hooks: HookTriggerConfig = field(default_factory=HookTriggerConfig)


@dataclass
class MonitoringConfig:
    json_logs: bool = False
    log_level: str = "INFO"
    stats_interval_seconds: float = 5.0
    prometheus_enabled: bool = False
    prometheus_host: str = "0.0.0.0"
    prometheus_port: int = 9108
    event_stdout: bool = True
    event_file: str | None = None


@dataclass
class RuntimeConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    backend_policy: BackendPolicyConfig = field(default_factory=BackendPolicyConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    triggers: TriggerConfig = field(default_factory=TriggerConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    stress_sleep_ms: int = 0

    def as_log_context(self) -> dict[str, Any]:
        return {
            "model": self.model.name,
            "backend_requested": self.backend_policy.requested,
            "ingest_backend": self.ingest.backend,
            "ingest_clock": self.ingest.clock_mode,
            "benchmark": self.ingest.benchmark,
            "headless": self.output.headless,
            "remote_enabled": self.output.remote_enabled,
            "json_logs": self.monitoring.json_logs,
        }
