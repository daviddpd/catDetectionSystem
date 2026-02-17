from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def queue_uncertain_events(
    events_jsonl: Path,
    output_path: Path,
    min_conf: float,
    max_conf: float,
    class_filter: list[str] | None = None,
    truth_jsonl: Path | None = None,
) -> dict[str, Any]:
    class_filter = class_filter or []
    queued: list[dict[str, Any]] = []
    predicted_by_frame: dict[str, set[str]] = {}
    truth_by_frame: dict[str, dict[str, Any]] = {}

    if truth_jsonl is not None and truth_jsonl.exists():
        with truth_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                frame_id = str(row.get("frame_id", ""))
                if not frame_id:
                    continue
                truth_by_frame[frame_id] = row

    with events_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            conf = float(event.get("confidence", 0.0))
            label = str(event.get("label", ""))
            frame_id = str(event.get("frame_id", ""))
            if frame_id:
                predicted_by_frame.setdefault(frame_id, set()).add(label)

            if not (conf < min_conf or conf > max_conf):
                if not class_filter or label in class_filter:
                    queued.append(
                        {
                            "reason": "uncertain_confidence",
                            "confidence": conf,
                            "label": label,
                            "frame_id": event.get("frame_id"),
                            "source": event.get("source"),
                            "bbox": event.get("bbox"),
                            "event": event,
                        }
                    )

            if frame_id and frame_id in truth_by_frame:
                truth_labels = set(str(v) for v in truth_by_frame[frame_id].get("labels", []))
                if truth_labels and label not in truth_labels:
                    queued.append(
                        {
                            "reason": "false_positive_candidate",
                            "confidence": conf,
                            "label": label,
                            "frame_id": event.get("frame_id"),
                            "source": event.get("source"),
                            "bbox": event.get("bbox"),
                            "event": event,
                        }
                    )

    for frame_id, truth in truth_by_frame.items():
        truth_labels = set(str(v) for v in truth.get("labels", []))
        predicted_labels = predicted_by_frame.get(frame_id, set())
        missing = sorted(truth_labels - predicted_labels)
        for label in missing:
            if class_filter and label not in class_filter:
                continue
            queued.append(
                {
                    "reason": "false_negative_candidate",
                    "confidence": None,
                    "label": label,
                    "frame_id": frame_id,
                    "source": truth.get("source"),
                    "bbox": None,
                    "event": {"truth": truth},
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in queued:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    return {
        "queue_path": str(output_path),
        "queued_items": len(queued),
        "min_conf": min_conf,
        "max_conf": max_conf,
        "truth_frames": len(truth_by_frame),
    }


def merge_review_queue(
    queue_jsonl: Path,
    source_images_root: Path,
    dataset_root: Path,
    target_split: str,
) -> dict[str, Any]:
    images_dir = dataset_root / "images" / target_split
    labels_dir = dataset_root / "labels" / target_split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    with queue_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            src = row.get("source")
            if not src:
                skipped += 1
                continue

            src_path = Path(src)
            if not src_path.is_absolute():
                src_path = source_images_root / src_path
            if not src_path.exists():
                skipped += 1
                continue

            dst_img = images_dir / src_path.name
            if not dst_img.exists():
                shutil.copy2(src_path, dst_img)
                copied += 1

            dst_lbl = labels_dir / f"{dst_img.stem}.txt"
            if not dst_lbl.exists():
                dst_lbl.write_text("", encoding="utf-8")

    summary = {
        "queue": str(queue_jsonl),
        "dataset_root": str(dataset_root),
        "target_split": target_split,
        "copied_images": copied,
        "skipped_rows": skipped,
    }

    summary_path = dataset_root / "reports" / "active_learning_merge.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
