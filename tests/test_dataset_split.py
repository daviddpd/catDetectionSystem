from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cds.training.dataset.split import deterministic_split, time_aware_split


class DatasetSplitTests(unittest.TestCase):
    def test_deterministic_split_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images = [root / f"img_{i:03d}.jpg" for i in range(20)]
            split_a = deterministic_split(images)
            split_b = deterministic_split(images)

            self.assertEqual(split_a["train"], split_b["train"])
            self.assertEqual(split_a["val"], split_b["val"])
            self.assertEqual(split_a["test"], split_b["test"])

    def test_time_aware_split_groups_by_source(self) -> None:
        records = [
            {"image": f"/tmp/cam1_{i}.jpg", "source_id": "cam1", "timestamp": f"20240101010{i}"}
            for i in range(5)
        ]
        records += [
            {"image": f"/tmp/cam2_{i}.jpg", "source_id": "cam2", "timestamp": f"20240102010{i}"}
            for i in range(5)
        ]

        split = time_aware_split(records, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

        self.assertEqual(len(split["train"]), 6)
        self.assertEqual(len(split["val"]), 2)
        self.assertEqual(len(split["test"]), 2)


if __name__ == "__main__":
    unittest.main()
