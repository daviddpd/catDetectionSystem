from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from cds.config.models import AudioTriggerConfig, HookTriggerConfig, TriggerConfig
from cds.triggers.audio import AudioTrigger
from cds.triggers.hooks import HookTrigger
from cds.types import Detection, FramePacket


@dataclass
class _LatchState:
    on_count: int = 0
    off_count: int = 0
    active: bool = False


@dataclass
class _TriggerGateConfig:
    enabled: bool
    on_frames: int
    off_frames: int
    min_area_pixels: int
    min_area_percent: float


@dataclass
class TriggerFrameResult:
    activated_detections: list[Detection]
    deactivated_labels: list[str]
    any_active: bool


class _HysteresisGate:
    def __init__(self, name: str, config: _TriggerGateConfig) -> None:
        self._name = name
        self._cfg = config
        self._states: dict[str, _LatchState] = {}

    def enabled(self) -> bool:
        return self._cfg.enabled

    def has_active(self) -> bool:
        if not self._cfg.enabled:
            return False
        return any(state.active for state in self._states.values())

    def _meets_area(self, detection: Detection) -> bool:
        area_pixels = int(detection.extra.get("area_pixels", detection.width * detection.height))
        area_percent = float(detection.extra.get("area_percent", 0.0))
        if self._cfg.min_area_pixels > 0 and area_pixels < self._cfg.min_area_pixels:
            return False
        if self._cfg.min_area_percent > 0 and area_percent < self._cfg.min_area_percent:
            return False
        return True

    def _pick_best(self, detections: list[Detection]) -> dict[str, Detection]:
        best: dict[str, Detection] = {}
        for det in detections:
            if not self._meets_area(det):
                continue
            prev = best.get(det.label)
            if prev is None or det.confidence > prev.confidence:
                best[det.label] = det
        return best

    def observe(self, detections: list[Detection]) -> TriggerFrameResult:
        if not self._cfg.enabled:
            return TriggerFrameResult(
                activated_detections=[],
                deactivated_labels=[],
                any_active=False,
            )

        present = self._pick_best(detections)
        activated: list[Detection] = []
        deactivated: list[str] = []
        labels = set(self._states) | set(present)

        for label in labels:
            state = self._states.setdefault(label, _LatchState())
            det = present.get(label)
            if det is not None:
                state.on_count += 1
                state.off_count = 0
                if not state.active and state.on_count >= self._cfg.on_frames:
                    state.active = True
                    activated.append(det)
                continue

            if state.active:
                state.off_count += 1
                if state.off_count >= self._cfg.off_frames:
                    state.active = False
                    state.on_count = 0
                    state.off_count = 0
                    deactivated.append(label)
            else:
                state.on_count = 0
                state.off_count = 0

        return TriggerFrameResult(
            activated_detections=activated,
            deactivated_labels=deactivated,
            any_active=self.has_active(),
        )


def _audio_gate_config(config: AudioTriggerConfig) -> _TriggerGateConfig:
    on_frames = max(1, int(config.frames_detect_on))
    off_frames = (
        max(1, int(config.frames_detect_off))
        if config.frames_detect_off is not None
        else max(1, on_frames // 2)
    )
    return _TriggerGateConfig(
        enabled=bool(config.enabled),
        on_frames=on_frames,
        off_frames=off_frames,
        min_area_pixels=max(0, int(config.min_area_pixels)),
        min_area_percent=max(0.0, float(config.min_area_percent)),
    )


def _hook_gate_config(config: HookTriggerConfig) -> _TriggerGateConfig:
    on_frames = max(1, int(config.frames_detect_on))
    off_frames = (
        max(1, int(config.frames_detect_off))
        if config.frames_detect_off is not None
        else max(1, on_frames // 2)
    )
    return _TriggerGateConfig(
        enabled=bool(config.enabled),
        on_frames=on_frames,
        off_frames=off_frames,
        min_area_pixels=max(0, int(config.min_area_pixels)),
        min_area_percent=max(0.0, float(config.min_area_percent)),
    )


class TriggerManager:
    def __init__(self, config: TriggerConfig) -> None:
        self._logger = logging.getLogger("cds.trigger")
        self._enabled = bool(config.audio.enabled or config.hooks.enabled)
        self._audio = AudioTrigger(config.audio)
        self._hooks = HookTrigger(config.hooks)
        self._audio_gate = _HysteresisGate("audio", _audio_gate_config(config.audio))
        self._hook_gate = _HysteresisGate("hooks", _hook_gate_config(config.hooks))

    def enabled(self) -> bool:
        return self._enabled

    def _log_transition(self, state: str, kind: str, det: Detection | None, label: str | None = None) -> None:
        if det is not None:
            area_pixels = int(det.extra.get("area_pixels", det.width * det.height))
            area_percent = float(det.extra.get("area_percent", 0.0))
            self._logger.info(
                "trigger_gate state=%s kind=%s label=%s conf=%.3f area_px=%d area_pct=%.3f",
                state,
                kind,
                det.label,
                det.confidence,
                area_pixels,
                area_percent,
            )
            return
        self._logger.info("trigger_gate state=%s kind=%s label=%s", state, kind, label or "")

    def process(
        self,
        packet: FramePacket,
        detections: list[Detection],
        backend_name: str,
    ) -> TriggerFrameResult:
        if not self._enabled:
            return TriggerFrameResult(
                activated_detections=[],
                deactivated_labels=[],
                any_active=False,
            )

        audio_result = self._audio_gate.observe(detections)
        hook_result = self._hook_gate.observe(detections)

        activation_map: dict[tuple[str, int, int, int, int], Detection] = {}
        for det in audio_result.activated_detections:
            self._log_transition("on", "audio", det)
            self._audio.emit(det.label)
            key = (det.label, det.x1, det.y1, det.x2, det.y2)
            activation_map[key] = det

        for label in audio_result.deactivated_labels:
            self._log_transition("off", "audio", None, label=label)

        for det in hook_result.activated_detections:
            self._log_transition("on", "hooks", det)
            payload = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "frame_id": packet.frame_id,
                "source": packet.source,
                "backend": backend_name,
                "label": det.label,
                "confidence": det.confidence,
                "bbox": [det.x1, det.y1, det.x2, det.y2],
                "area_pixels": int(det.extra.get("area_pixels", det.width * det.height)),
                "area_percent": float(det.extra.get("area_percent", 0.0)),
            }
            self._hooks.emit(det.label, payload)
            key = (det.label, det.x1, det.y1, det.x2, det.y2)
            activation_map[key] = det

        for label in hook_result.deactivated_labels:
            self._log_transition("off", "hooks", None, label=label)

        return TriggerFrameResult(
            activated_detections=list(activation_map.values()),
            deactivated_labels=sorted(set(audio_result.deactivated_labels + hook_result.deactivated_labels)),
            any_active=(self._audio_gate.has_active() or self._hook_gate.has_active()),
        )

    def close(self) -> None:
        self._audio.close()
        self._hooks.close()
