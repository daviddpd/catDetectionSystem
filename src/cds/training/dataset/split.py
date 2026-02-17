from __future__ import annotations

import hashlib
from pathlib import Path


def _stable_bucket(key: str) -> float:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16)
    return value / 0xFFFFFFFF


def deterministic_split(
    image_paths: list[Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, list[Path]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 0.0001:
        raise ValueError("Split ratios must sum to 1.0")

    train: list[Path] = []
    val: list[Path] = []
    test: list[Path] = []

    train_cut = train_ratio
    val_cut = train_ratio + val_ratio

    for image_path in sorted(image_paths):
        bucket = _stable_bucket(str(image_path.resolve()))
        if bucket < train_cut:
            train.append(image_path)
        elif bucket < val_cut:
            val.append(image_path)
        else:
            test.append(image_path)

    return {"train": train, "val": val, "test": test}


def time_aware_split(
    records: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, list[Path]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 0.0001:
        raise ValueError("Split ratios must sum to 1.0")

    sorted_records = sorted(
        records,
        key=lambda item: (item.get("source_id", ""), item.get("timestamp") or ""),
    )

    grouped: dict[str, list[dict]] = {}
    for record in sorted_records:
        source_id = record.get("source_id", "unknown")
        grouped.setdefault(source_id, []).append(record)

    train: list[Path] = []
    val: list[Path] = []
    test: list[Path] = []

    for source_id in sorted(grouped):
        rows = grouped[source_id]
        n = len(rows)
        train_n = int(n * train_ratio)
        val_n = int(n * val_ratio)

        for idx, row in enumerate(rows):
            path = Path(row["image"]) if isinstance(row["image"], str) else row["image"]
            if idx < train_n:
                train.append(path)
            elif idx < train_n + val_n:
                val.append(path)
            else:
                test.append(path)

    return {"train": train, "val": val, "test": test}
