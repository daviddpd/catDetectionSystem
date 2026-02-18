from __future__ import annotations

from datetime import datetime
import unittest

from cds.config.models import AudioTriggerConfig, HookTriggerConfig, TriggerConfig
from cds.triggers.manager import TriggerManager
from cds.types import Detection, FramePacket


class TriggerManagerTests(unittest.TestCase):
    def test_enabled_false_when_audio_and_hooks_disabled(self) -> None:
        config = TriggerConfig(
            audio=AudioTriggerConfig(enabled=False),
            hooks=HookTriggerConfig(enabled=False),
        )
        manager = TriggerManager(config)
        try:
            self.assertFalse(manager.enabled())
            manager.process(
                packet=FramePacket(
                    frame_id=1,
                    frame=None,
                    source="unit-test",
                    timestamp=datetime.now(),
                ),
                detections=[
                    Detection(
                        class_id=0,
                        label="cat",
                        confidence=0.9,
                        x1=0,
                        y1=0,
                        x2=10,
                        y2=10,
                    )
                ],
                backend_name="unit",
            )
        finally:
            manager.close()

    def test_enabled_true_when_audio_enabled(self) -> None:
        config = TriggerConfig(
            audio=AudioTriggerConfig(enabled=True),
            hooks=HookTriggerConfig(enabled=False),
        )
        manager = TriggerManager(config)
        try:
            self.assertTrue(manager.enabled())
        finally:
            manager.close()


if __name__ == "__main__":
    unittest.main()
