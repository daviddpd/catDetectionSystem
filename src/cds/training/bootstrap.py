from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cds.training.constants import DEFAULT_BOOTSTRAP_MODEL, REQUIRED_CLASSES
from cds.training.paths import new_run_id, prepare_artifact_dirs

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm"}
_MEDIA_EXTS = _IMAGE_EXTS | _VIDEO_EXTS


def _parse_classes(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def _ensure_required_classes(classes: list[str]) -> None:
    missing = [name for name in REQUIRED_CLASSES if name not in classes]
    if missing:
        raise ValueError(
            "bootstrap-openvocab class list must include required classes: "
            + ", ".join(missing)
        )


def _expand_sources(source: str) -> list[str]:
    path = Path(source).expanduser()
    if not path.exists():
        return [source]
    if path.is_file():
        if path.suffix.lower() in _MEDIA_EXTS:
            return [str(path.resolve())]
        return []

    # Directory input: recursively collect all supported media files.
    sources: list[str] = []
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() in _MEDIA_EXTS:
            sources.append(str(candidate.resolve()))
    return sources


def run_bootstrap_openvocab(
    source: str,
    classes_csv: str,
    output_dir: Path,
    model_name: str = DEFAULT_BOOTSTRAP_MODEL,
    conf: float = 0.25,
    imgsz: int = 640,
    max_frames: int = 0,
) -> dict[str, Any]:
    classes = _parse_classes(classes_csv)
    _ensure_required_classes(classes)

    try:
        from ultralytics import YOLOWorld
    except ImportError as exc:
        raise RuntimeError(
            "bootstrap-openvocab requires ultralytics package. Install dependencies first."
        ) from exc

    run_id = new_run_id("bootstrap-openvocab")
    dirs = prepare_artifact_dirs(output_dir, run_id)

    review_images = dirs["bootstrap"] / "review" / "images"
    review_labels = dirs["bootstrap"] / "review" / "labels"
    review_images.mkdir(parents=True, exist_ok=True)
    review_labels.mkdir(parents=True, exist_ok=True)

    model = YOLOWorld(model_name)
    model.set_classes(classes)

    expanded_sources = _expand_sources(source)
    if not expanded_sources:
        raise RuntimeError(
            f"No supported media files found for source: {source}. "
            f"Supported suffixes: {sorted(_MEDIA_EXTS)}"
        )

    rows: list[dict[str, Any]] = []
    frame_count = 0
    detection_count = 0
    source_count = len(expanded_sources)

    try:
        import cv2
    except Exception:
        cv2 = None

    stop = False
    for source_item in expanded_sources:
        if stop:
            break
        source_tag = Path(source_item).stem.replace(" ", "_")
        results = model.predict(
            source=source_item,
            conf=conf,
            imgsz=imgsz,
            stream=True,
            verbose=False,
        )

        for result in results:
            if max_frames > 0 and frame_count >= max_frames:
                stop = True
                break
            frame_count += 1

            image_name = f"{source_tag}_frame-{frame_count:08d}.jpg"
            image_dest = review_images / image_name
            label_dest = review_labels / f"{Path(image_name).stem}.txt"

            if cv2 is not None and hasattr(result, "orig_img"):
                cv2.imwrite(str(image_dest), result.orig_img)

            boxes = getattr(result, "boxes", None)
            names = getattr(result, "names", {}) or {}

            label_lines: list[str] = []
            frame_events: list[dict[str, Any]] = []
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    xywhn = [float(v) for v in box.xywhn[0].tolist()]
                    if len(xywhn) != 4:
                        continue

                    class_name = str(names.get(class_id, class_id))
                    line = (
                        f"{class_id} {xywhn[0]:.6f} {xywhn[1]:.6f} "
                        f"{xywhn[2]:.6f} {xywhn[3]:.6f}"
                    )
                    label_lines.append(line)
                    detection_count += 1

                    frame_events.append(
                        {
                            "frame_index": frame_count,
                            "image": str(image_dest),
                            "label_file": str(label_dest),
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": confidence,
                            "xywhn": xywhn,
                            "source": str(getattr(result, "path", source_item)),
                        }
                    )

            label_dest.write_text(
                "\n".join(label_lines) + ("\n" if label_lines else ""),
                encoding="utf-8",
            )
            rows.extend(frame_events)

    events_path = dirs["bootstrap"] / "review" / "predictions.jsonl"
    with events_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    class_list_path = dirs["bootstrap"] / "review" / "classes.txt"
    class_list_path.write_text("\n".join(classes) + "\n", encoding="utf-8")

    summary = {
        "run_id": run_id,
        "source": source,
        "resolved_sources": expanded_sources,
        "source_count": source_count,
        "model": model_name,
        "classes": classes,
        "frame_count": frame_count,
        "detection_count": detection_count,
        "predictions": str(events_path),
        "review_images_dir": str(review_images),
        "review_labels_dir": str(review_labels),
    }

    summary_path = dirs["bootstrap"] / "review" / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    return summary
