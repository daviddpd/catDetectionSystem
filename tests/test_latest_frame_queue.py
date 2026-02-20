from __future__ import annotations

import unittest

from cds.pipeline.frame_queue import LatestFrameQueue


class LatestFrameQueueTests(unittest.TestCase):
    def test_drops_oldest_when_full(self) -> None:
        queue = LatestFrameQueue[int](maxsize=2)

        self.assertEqual(queue.put_latest(1), 0)
        self.assertEqual(queue.put_latest(2), 0)
        self.assertEqual(queue.put_latest(3), 1)

        first = queue.get(timeout=0.01)
        second = queue.get(timeout=0.01)

        self.assertEqual(first, 2)
        self.assertEqual(second, 3)

    def test_no_drop_mode_reports_full_without_overwriting(self) -> None:
        queue = LatestFrameQueue[int](maxsize=2, drop_oldest=False)

        self.assertEqual(queue.put_latest(1), 0)
        self.assertEqual(queue.put_latest(2), 0)
        self.assertEqual(queue.put_latest(3), 1)

        first = queue.get(timeout=0.01)
        second = queue.get(timeout=0.01)

        self.assertEqual(first, 1)
        self.assertEqual(second, 2)


if __name__ == "__main__":
    unittest.main()
