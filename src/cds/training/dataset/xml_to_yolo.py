from __future__ import annotations

import hashlib
import re
import shutil
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class ConversionStats:
    xml_files: int
    images_found: int
    labels_written: int
    skipped_unknown_class: int
    skipped_missing_image: int
    repaired_missing_size: int
    skipped_invalid_size: int


def _find_image(xml_path: Path, image_root: Path | None) -> Path | None:
    candidates: list[Path] = []
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    filename = root.findtext("filename")
    if filename:
        p = Path(filename)
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(xml_path.parent / filename)
            if image_root is not None:
                candidates.append(image_root / filename)

    for candidate in candidates:
        if candidate.exists() and candidate.suffix.lower() in _IMAGE_SUFFIXES:
            return candidate.resolve()

    stem = xml_path.stem
    search_dirs = [xml_path.parent]
    if image_root is not None:
        search_dirs.append(image_root)
    for directory in search_dirs:
        if not directory.exists():
            continue
        for suffix in sorted(_IMAGE_SUFFIXES):
            candidate = directory / f"{stem}{suffix}"
            if candidate.exists():
                return candidate.resolve()

    return None


def _read_image_size(root: ET.Element) -> tuple[int, int] | None:
    try:
        width = int(float(root.findtext("size/width", default="0")))
        height = int(float(root.findtext("size/height", default="0")))
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _upsert_xml_image_size(root: ET.Element, width: int, height: int) -> None:
    size_node = root.find("size")
    if size_node is None:
        size_node = ET.SubElement(root, "size")

    width_node = size_node.find("width")
    if width_node is None:
        width_node = ET.SubElement(size_node, "width")
    width_node.text = str(int(width))

    height_node = size_node.find("height")
    if height_node is None:
        height_node = ET.SubElement(size_node, "height")
    height_node.text = str(int(height))

    depth_node = size_node.find("depth")
    if depth_node is None:
        depth_node = ET.SubElement(size_node, "depth")
    if not (depth_node.text or "").strip():
        depth_node.text = "3"


def _probe_image_size_pillow(image_path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image
    except Exception:
        return None

    try:
        with Image.open(image_path) as image:
            width, height = image.size
        if width > 0 and height > 0:
            return int(width), int(height)
    except Exception:
        return None
    return None


def _probe_image_size_cv2(image_path: Path) -> tuple[int, int] | None:
    try:
        import cv2
    except Exception:
        return None

    try:
        image = cv2.imread(str(image_path))
    except Exception:
        return None
    if image is None or len(image.shape) < 2:
        return None
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        return None
    return int(width), int(height)


def _probe_image_size_headers(image_path: Path) -> tuple[int, int] | None:
    try:
        with image_path.open("rb") as handle:
            header = handle.read(64)
            if len(header) < 10:
                return None

            if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24 and header[12:16] == b"IHDR":
                width, height = struct.unpack(">II", header[16:24])
                if width > 0 and height > 0:
                    return int(width), int(height)

            if header[:2] == b"BM" and len(header) >= 26:
                width = struct.unpack("<i", header[18:22])[0]
                height = abs(struct.unpack("<i", header[22:26])[0])
                if width > 0 and height > 0:
                    return int(width), int(height)

            if header[:6] in {b"GIF87a", b"GIF89a"}:
                width, height = struct.unpack("<HH", header[6:10])
                if width > 0 and height > 0:
                    return int(width), int(height)

            if header[:4] == b"RIFF" and header[8:12] == b"WEBP" and len(header) >= 30:
                chunk = header[12:16]
                if chunk == b"VP8X":
                    width = 1 + int.from_bytes(header[24:27], "little")
                    height = 1 + int.from_bytes(header[27:30], "little")
                    if width > 0 and height > 0:
                        return width, height
                if chunk == b"VP8L" and len(header) >= 25:
                    b0, b1, b2, b3 = header[21:25]
                    width = 1 + (((b1 & 0x3F) << 8) | b0)
                    height = 1 + (((b3 & 0x0F) << 10) | (b2 << 2) | ((b1 & 0xC0) >> 6))
                    if width > 0 and height > 0:
                        return width, height

            if header[:2] == b"\xFF\xD8":
                data = header + handle.read()
                return _probe_jpeg_size(data)
    except Exception:
        return None

    return None


def _probe_jpeg_size(data: bytes) -> tuple[int, int] | None:
    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    if len(data) < 4 or data[:2] != b"\xFF\xD8":
        return None

    offset = 2
    while offset + 1 < len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break

        marker = data[offset]
        offset += 1

        if marker in {0xD8, 0xD9}:
            continue
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(data):
            break

        segment_length = struct.unpack(">H", data[offset : offset + 2])[0]
        if segment_length < 2 or offset + segment_length > len(data):
            break

        if marker in sof_markers:
            if segment_length < 7:
                break
            height = struct.unpack(">H", data[offset + 3 : offset + 5])[0]
            width = struct.unpack(">H", data[offset + 5 : offset + 7])[0]
            if width > 0 and height > 0:
                return int(width), int(height)

        offset += segment_length

    return None


def _probe_image_size(image_path: Path) -> tuple[int, int] | None:
    for probe in (_probe_image_size_pillow, _probe_image_size_cv2, _probe_image_size_headers):
        size = probe(image_path)
        if size is not None:
            return size
    return None


def _to_yolo_line(class_id: int, width: int, height: int, box: tuple[float, float, float, float]) -> str:
    xmin, ymin, xmax, ymax = box
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_w = (xmax - xmin) / width
    box_h = (ymax - ymin) / height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"


def _safe_source_id(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
    return f"src-{digest}"


def _extract_timestamp_from_name(path: Path) -> str | None:
    # Common pattern from camera filenames like 20211013205816 or 2021-10-13_20-58-16.
    stem = path.stem
    compact = re.search(r"(20\d{12})", stem)
    if compact:
        return compact.group(1)
    dashed = re.search(r"(20\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2})", stem)
    if dashed:
        return dashed.group(1)
    return None


def convert_voc_xml_to_yolo(
    xml_root: Path,
    output_root: Path,
    class_names: list[str] | None = None,
    image_root: Path | None = None,
    copy_images: bool = True,
) -> dict[str, Any]:
    class_names = [item.strip() for item in (class_names or []) if str(item).strip()]
    if not class_names:
        raise ValueError("convert_voc_xml_to_yolo requires non-empty class_names")
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    labels_dir = output_root / "labels_raw"
    images_dir = output_root / "images_raw"
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    stats = ConversionStats(
        xml_files=0,
        images_found=0,
        labels_written=0,
        skipped_unknown_class=0,
        skipped_missing_image=0,
        repaired_missing_size=0,
        skipped_invalid_size=0,
    )

    manifest_rows: list[dict[str, Any]] = []
    repaired_size_xml: list[str] = []
    skipped_invalid_size_xml: list[str] = []

    xml_paths = sorted(xml_root.rglob("*.xml"))
    for xml_path in xml_paths:
        stats.xml_files += 1
        tree = ET.parse(str(xml_path))
        root = tree.getroot()

        image_path = _find_image(xml_path, image_root)
        if image_path is None:
            stats.skipped_missing_image += 1
            continue
        stats.images_found += 1

        image_size = _read_image_size(root)
        if image_size is None:
            inferred_size = _probe_image_size(image_path)
            if inferred_size is None:
                stats.skipped_invalid_size += 1
                skipped_invalid_size_xml.append(str(xml_path))
                continue

            _upsert_xml_image_size(root, inferred_size[0], inferred_size[1])
            try:
                tree.write(str(xml_path), encoding="utf-8", xml_declaration=True)
            except Exception:
                # Continue with in-memory repaired XML when filesystem update fails.
                pass

            image_size = _read_image_size(root)
            if image_size is None:
                stats.skipped_invalid_size += 1
                skipped_invalid_size_xml.append(str(xml_path))
                continue

            stats.repaired_missing_size += 1
            repaired_size_xml.append(str(xml_path))

        width, height = image_size
        lines: list[str] = []

        for obj in root.findall("object"):
            class_name = (obj.findtext("name") or "").strip()
            if class_name not in class_to_id:
                stats.skipped_unknown_class += 1
                continue

            bnd = obj.find("bndbox")
            if bnd is None:
                continue

            xmin = float(bnd.findtext("xmin", default="0"))
            ymin = float(bnd.findtext("ymin", default="0"))
            xmax = float(bnd.findtext("xmax", default="0"))
            ymax = float(bnd.findtext("ymax", default="0"))

            xmin = max(0.0, min(xmin, float(width)))
            ymin = max(0.0, min(ymin, float(height)))
            xmax = max(0.0, min(xmax, float(width)))
            ymax = max(0.0, min(ymax, float(height)))
            if xmax <= xmin or ymax <= ymin:
                continue

            lines.append(_to_yolo_line(class_to_id[class_name], width, height, (xmin, ymin, xmax, ymax)))

        source_id = _safe_source_id(image_path)
        image_name = f"{source_id}_{image_path.name}"
        label_name = f"{Path(image_name).stem}.txt"
        image_dest = images_dir / image_name
        label_dest = labels_dir / label_name

        if copy_images:
            if image_dest != image_path:
                shutil.copy2(image_path, image_dest)
        else:
            image_dest = image_path

        label_dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats.labels_written += 1

        manifest_rows.append(
            {
                "xml": str(xml_path),
                "image": str(image_dest),
                "label": str(label_dest),
                "source_id": source_id,
                "timestamp": _extract_timestamp_from_name(image_path),
            }
        )

    return {
        "stats": {
            "xml_files": stats.xml_files,
            "images_found": stats.images_found,
            "labels_written": stats.labels_written,
            "skipped_unknown_class": stats.skipped_unknown_class,
            "skipped_missing_image": stats.skipped_missing_image,
            "repaired_missing_size": stats.repaired_missing_size,
            "skipped_invalid_size": stats.skipped_invalid_size,
        },
        "size_repair": {
            "repaired_xml": repaired_size_xml,
            "skipped_xml": skipped_invalid_size_xml,
        },
        "classes": class_names,
        "manifest": manifest_rows,
    }
