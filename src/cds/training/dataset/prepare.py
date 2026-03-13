from __future__ import annotations

from pathlib import Path
from typing import Any

from cds.training.dataset.split import deterministic_split, time_aware_split
from cds.training.dataset.validate import validate_yolo_dataset
from cds.training.dataset.xml_to_yolo import convert_voc_xml_to_yolo
from cds.utils.config_io import write_json


def _write_split_paths(paths: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(str(p.resolve()) for p in paths) + "\n", encoding="utf-8")


def _safe_link_name(record: dict[str, Any]) -> tuple[str, str]:
    source_id = str(record.get("source_id", "")).strip()
    image_path = Path(record["image"])
    if source_id:
        image_name = f"{source_id}_{image_path.name}"
    else:
        image_name = image_path.name
    label_name = f"{Path(image_name).stem}.txt"
    return image_name, label_name


def _reset_split_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            _reset_split_dir(child)
            child.rmdir()
        else:
            child.unlink()


def _symlink_or_replace(target: Path, source: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source.resolve())


def _link_split(split_name: str, records: list[dict[str, Any]], output_root: Path) -> list[Path]:
    image_dest = output_root / "images" / split_name
    label_dest = output_root / "labels" / split_name
    _reset_split_dir(image_dest)
    _reset_split_dir(label_dest)

    linked_images: list[Path] = []
    for record in records:
        image_path = Path(record["image"])
        label_path = Path(record["label"])
        image_name, label_name = _safe_link_name(record)

        dest_image = image_dest / image_name
        dest_label = label_dest / label_name
        _symlink_or_replace(dest_image, image_path)
        if label_path.exists():
            _symlink_or_replace(dest_label, label_path)
        else:
            dest_label.write_text("", encoding="utf-8")
        linked_images.append(dest_image)
    return linked_images


def _write_data_yaml(output_root: Path, class_names: list[str]) -> Path:
    yaml_path = output_root / "data.yaml"
    lines = [
        f"path: {output_root.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for name in class_names:
        lines.append(f"  - {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def _write_classes_file(output_root: Path, class_names: list[str]) -> Path:
    path = output_root / "classes.txt"
    path.write_text("\n".join(class_names) + "\n", encoding="utf-8")
    return path


def prepare_dataset_pipeline(
    output_root: Path,
    xml_root: Path,
    image_root: Path | None,
    class_names: list[str],
    split_mode: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, Any]:
    class_names = [item.strip() for item in class_names if str(item).strip()]
    if not class_names:
        raise ValueError("prepare_dataset_pipeline requires non-empty class_names")

    conversion = convert_voc_xml_to_yolo(
        xml_root=xml_root,
        output_root=output_root,
        class_names=class_names,
        image_root=image_root,
        copy_images=True,
    )

    records = conversion["manifest"]
    image_paths = [Path(row["image"]) for row in records]
    record_by_image = {str(Path(row["image"]).resolve()): row for row in records}

    if split_mode == "time-aware":
        split_paths = time_aware_split(
            records=records,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
    else:
        split_paths = deterministic_split(
            image_paths=image_paths,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

    split_records: dict[str, list[dict[str, Any]]] = {}
    linked_split_paths: dict[str, list[Path]] = {}
    for split_name, split_image_paths in split_paths.items():
        rows: list[dict[str, Any]] = []
        for image_path in split_image_paths:
            row = record_by_image.get(str(Path(image_path).resolve()))
            if row is None:
                continue
            rows.append(row)
        split_records[split_name] = rows
        linked_split_paths[split_name] = _link_split(split_name, rows, output_root)
        _write_split_paths(linked_split_paths[split_name], output_root / "splits" / f"{split_name}.txt")

    data_yaml = _write_data_yaml(output_root, class_names)
    classes_file = _write_classes_file(output_root, class_names)

    health = validate_yolo_dataset(output_root, class_names)
    health_path = output_root / "reports" / "dataset_health.json"
    write_json(health_path, health)

    manifest_payload = {
        "conversion": conversion["stats"],
        "split_mode": split_mode,
        "split_sizes": {k: len(v) for k, v in split_records.items()},
        "class_names": class_names,
        "classes_file": str(classes_file),
        "data_yaml": str(data_yaml),
        "health_report": str(health_path),
    }
    if "size_repair" in conversion:
        manifest_payload["size_repair"] = conversion["size_repair"]
    write_json(output_root / "reports" / "dataset_manifest.json", manifest_payload)

    return {
        "manifest": manifest_payload,
        "health": health,
    }
