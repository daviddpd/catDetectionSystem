from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from cds.training.dataset.split import deterministic_split, time_aware_split
from cds.training.dataset.validate import validate_yolo_dataset
from cds.training.dataset.xml_to_yolo import convert_voc_xml_to_yolo
from cds.utils.config_io import write_json


def _write_split_paths(paths: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(str(p.resolve()) for p in paths) + "\n", encoding="utf-8")


def _copy_split(split_name: str, image_paths: list[Path], output_root: Path) -> None:
    image_dest = output_root / "images" / split_name
    label_dest = output_root / "labels" / split_name
    image_dest.mkdir(parents=True, exist_ok=True)
    label_dest.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        dest_image = image_dest / image_path.name
        if dest_image != image_path:
            shutil.copy2(image_path, dest_image)
        raw_label = output_root / "labels_raw" / f"{image_path.stem}.txt"
        if raw_label.exists():
            shutil.copy2(raw_label, label_dest / raw_label.name)
        else:
            (label_dest / f"{image_path.stem}.txt").write_text("", encoding="utf-8")


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

    if split_mode == "time-aware":
        splits = time_aware_split(
            records=records,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
    else:
        splits = deterministic_split(
            image_paths=image_paths,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

    for split_name, split_paths in splits.items():
        _copy_split(split_name, split_paths, output_root)
        _write_split_paths(split_paths, output_root / "splits" / f"{split_name}.txt")

    data_yaml = _write_data_yaml(output_root, class_names)
    classes_file = _write_classes_file(output_root, class_names)

    health = validate_yolo_dataset(output_root, class_names)
    health_path = output_root / "reports" / "dataset_health.json"
    write_json(health_path, health)

    manifest_payload = {
        "conversion": conversion["stats"],
        "split_mode": split_mode,
        "split_sizes": {k: len(v) for k, v in splits.items()},
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
