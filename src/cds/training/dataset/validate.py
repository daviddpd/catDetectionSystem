from __future__ import annotations

from pathlib import Path
from typing import Any

from cds.training.dataset.types import ValidationIssue


def _parse_line(line: str) -> tuple[int, float, float, float, float] | None:
    parts = line.strip().split()
    if not parts:
        return None
    if len(parts) != 5:
        raise ValueError("Label row must have 5 fields")
    class_id = int(parts[0])
    x, y, w, h = map(float, parts[1:])
    return class_id, x, y, w, h


def validate_yolo_dataset(
    dataset_root: Path,
    class_names: list[str],
    split_names: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, Any]:
    issues: list[ValidationIssue] = []

    expected_ids = set(range(len(class_names)))

    duplicate_detector: set[tuple[str, str]] = set()
    counts = {
        "images": 0,
        "labels": 0,
        "empty_labels": 0,
        "out_of_bounds_boxes": 0,
        "unknown_class_ids": 0,
        "duplicates": 0,
        "missing_labels": 0,
        "missing_images": 0,
    }

    for split in split_names:
        img_dir = dataset_root / "images" / split
        lbl_dir = dataset_root / "labels" / split
        if not img_dir.exists():
            continue

        for image_path in sorted(img_dir.rglob("*")):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                continue
            counts["images"] += 1

            label_path = lbl_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                counts["missing_labels"] += 1
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="missing_label",
                        message="Image missing label file",
                        image_path=str(image_path),
                        label_path=str(label_path),
                    )
                )
                continue

            counts["labels"] += 1
            rows = label_path.read_text(encoding="utf-8").splitlines()
            if not rows:
                counts["empty_labels"] += 1
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="empty_label",
                        message="Label file has no boxes",
                        image_path=str(image_path),
                        label_path=str(label_path),
                    )
                )
                continue

            seen_rows: set[str] = set()
            for raw in rows:
                try:
                    parsed = _parse_line(raw)
                except Exception as exc:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="parse_error",
                            message=f"Invalid label row: {exc}",
                            image_path=str(image_path),
                            label_path=str(label_path),
                        )
                    )
                    continue
                if parsed is None:
                    continue

                class_id, x, y, w, h = parsed
                if class_id not in expected_ids:
                    counts["unknown_class_ids"] += 1
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="unknown_class_id",
                            message=f"Class id {class_id} is outside expected range 0..{len(class_names)-1}",
                            image_path=str(image_path),
                            label_path=str(label_path),
                        )
                    )

                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    counts["out_of_bounds_boxes"] += 1
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="out_of_bounds",
                            message="Bounding box is out of normalized YOLO bounds",
                            image_path=str(image_path),
                            label_path=str(label_path),
                        )
                    )

                row_key = raw.strip()
                if row_key in seen_rows:
                    counts["duplicates"] += 1
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            code="duplicate_row",
                            message="Duplicate label row in file",
                            image_path=str(image_path),
                            label_path=str(label_path),
                        )
                    )
                seen_rows.add(row_key)

                global_key = (str(image_path), row_key)
                if global_key in duplicate_detector:
                    counts["duplicates"] += 1
                duplicate_detector.add(global_key)

        if lbl_dir.exists():
            image_stems = {
                p.stem
                for p in img_dir.rglob("*")
                if p.is_file()
                and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            }
            for label_path in sorted(lbl_dir.rglob("*.txt")):
                if label_path.stem not in image_stems:
                    counts["missing_images"] += 1
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="missing_image",
                            message="Label file has no matching image",
                            image_path=str(img_dir / f"{label_path.stem}.jpg"),
                            label_path=str(label_path),
                        )
                    )

    status = "pass"
    if counts["images"] == 0:
        issues.append(
            ValidationIssue(
                severity="error",
                code="no_images",
                message="Dataset contains zero images across requested splits",
                image_path=str(dataset_root),
                label_path=None,
            )
        )
        status = "fail"
    if any(issue.severity == "error" for issue in issues):
        status = "fail"

    return {
        "status": status,
        "counts": counts,
        "issues": [issue.__dict__ for issue in issues],
        "class_names": class_names,
    }
