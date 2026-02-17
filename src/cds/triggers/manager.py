from __future__ import annotations

from datetime import datetime, timezone

from cds.config.models import TriggerConfig
from cds.triggers.audio import AudioTrigger
from cds.triggers.hooks import HookTrigger
from cds.types import Detection, FramePacket


class TriggerManager:
    def __init__(self, config: TriggerConfig) -> None:
        self._audio = AudioTrigger(config.audio)
        self._hooks = HookTrigger(config.hooks)

    def process(
        self,
        packet: FramePacket,
        detections: list[Detection],
        backend_name: str,
    ) -> None:
        for detection in detections:
            self._audio.emit(detection.label)
            payload = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "frame_id": packet.frame_id,
                "source": packet.source,
                "backend": backend_name,
                "label": detection.label,
                "confidence": detection.confidence,
                "bbox": [detection.x1, detection.y1, detection.x2, detection.y2],
            }
            self._hooks.emit(detection.label, payload)

    def close(self) -> None:
        self._audio.close()
        self._hooks.close()
