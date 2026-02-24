from __future__ import annotations

from datetime import datetime
import unittest

from cds.config.models import AudioTriggerConfig, HookTriggerConfig, TriggerConfig
from cds.triggers.manager import TriggerManager
from cds.types import Detection, FramePacket


class TriggerManagerTests(unittest.TestCase):
    def _packet(self, frame_id: int = 1) -> FramePacket:
        return FramePacket(
            frame_id=frame_id,
            frame=None,
            source="unit-test",
            timestamp=datetime.now(),
        )

    def _det(self, label: str = "cat", conf: float = 0.9, area_px: int = 100, area_pct: float = 1.0) -> Detection:
        det = Detection(
            class_id=0,
            label=label,
            confidence=conf,
            x1=0,
            y1=0,
            x2=10,
            y2=10,
        )
        det.extra["area_pixels"] = area_px
        det.extra["area_percent"] = area_pct
        return det

    def test_enabled_false_when_audio_and_hooks_disabled(self) -> None:
        config = TriggerConfig(
            audio=AudioTriggerConfig(enabled=False),
            hooks=HookTriggerConfig(enabled=False),
        )
        manager = TriggerManager(config)
        try:
            self.assertFalse(manager.enabled())
            result = manager.process(
                packet=self._packet(1),
                detections=[self._det()],
                backend_name="unit",
            )
            self.assertFalse(result.any_active)
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

    def test_hysteresis_requires_sequential_frames_on_and_off(self) -> None:
        config = TriggerConfig(
            audio=AudioTriggerConfig(
                enabled=True,
                frames_detect_on=3,
                frames_detect_off=2,
            ),
            hooks=HookTriggerConfig(enabled=False),
        )
        manager = TriggerManager(config)
        try:
            r1 = manager.process(self._packet(1), [self._det()], "unit")
            self.assertFalse(r1.any_active)
            self.assertEqual(len(r1.activated_detections), 0)

            r2 = manager.process(self._packet(2), [self._det()], "unit")
            self.assertFalse(r2.any_active)
            self.assertEqual(len(r2.activated_detections), 0)

            r3 = manager.process(self._packet(3), [self._det()], "unit")
            self.assertTrue(r3.any_active)
            self.assertEqual(len(r3.activated_detections), 1)

            r4 = manager.process(self._packet(4), [], "unit")
            self.assertTrue(r4.any_active)

            r5 = manager.process(self._packet(5), [], "unit")
            self.assertFalse(r5.any_active)
        finally:
            manager.close()

    def test_area_gating_blocks_activation_until_size_threshold_met(self) -> None:
        config = TriggerConfig(
            audio=AudioTriggerConfig(
                enabled=True,
                frames_detect_on=1,
                min_area_pixels=500,
                min_area_percent=2.0,
            ),
            hooks=HookTriggerConfig(enabled=False),
        )
        manager = TriggerManager(config)
        try:
            blocked = manager.process(
                self._packet(1),
                [self._det(area_px=400, area_pct=3.0)],
                "unit",
            )
            self.assertFalse(blocked.any_active)
            self.assertEqual(len(blocked.activated_detections), 0)

            blocked_pct = manager.process(
                self._packet(2),
                [self._det(area_px=800, area_pct=1.0)],
                "unit",
            )
            self.assertFalse(blocked_pct.any_active)
            self.assertEqual(len(blocked_pct.activated_detections), 0)

            activated = manager.process(
                self._packet(3),
                [self._det(area_px=800, area_pct=2.5)],
                "unit",
            )
            self.assertTrue(activated.any_active)
            self.assertEqual(len(activated.activated_detections), 1)
        finally:
            manager.close()


if __name__ == "__main__":
    unittest.main()
