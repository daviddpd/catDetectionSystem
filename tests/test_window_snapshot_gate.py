from __future__ import annotations

import unittest

from cds.pipeline.snapshot_gate import WindowSnapshotGate
from cds.types import Detection


def _det(confidence: float = 0.9, area_pixels: int = 100, area_percent: float = 1.0) -> Detection:
    det = Detection(
        class_id=0,
        label="cat",
        confidence=confidence,
        x1=0,
        y1=0,
        x2=10,
        y2=10,
    )
    det.extra["area_pixels"] = area_pixels
    det.extra["area_percent"] = area_percent
    return det


class WindowSnapshotGateTests(unittest.TestCase):
    def test_hysteresis_activation_and_release(self) -> None:
        gate = WindowSnapshotGate(
            on_frames=3,
            off_frames=2,
            min_area_pixels=0,
            min_area_percent=0.0,
        )

        activated, active = gate.observe([_det()])
        self.assertFalse(active)
        self.assertEqual(activated, [])

        activated, active = gate.observe([_det()])
        self.assertFalse(active)
        self.assertEqual(activated, [])

        activated, active = gate.observe([_det()])
        self.assertTrue(active)
        self.assertEqual(len(activated), 1)

        activated, active = gate.observe([])
        self.assertTrue(active)
        self.assertEqual(activated, [])

        activated, active = gate.observe([])
        self.assertFalse(active)
        self.assertEqual(activated, [])

    def test_area_thresholds_filter_candidates(self) -> None:
        gate = WindowSnapshotGate(
            on_frames=1,
            off_frames=1,
            min_area_pixels=500,
            min_area_percent=2.0,
        )

        activated, active = gate.observe([_det(area_pixels=400, area_percent=3.0)])
        self.assertFalse(active)
        self.assertEqual(activated, [])

        activated, active = gate.observe([_det(area_pixels=800, area_percent=1.0)])
        self.assertFalse(active)
        self.assertEqual(activated, [])

        activated, active = gate.observe([_det(area_pixels=800, area_percent=2.5)])
        self.assertTrue(active)
        self.assertEqual(len(activated), 1)


if __name__ == "__main__":
    unittest.main()
