from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from cds.training.dataset.prepare import prepare_dataset_pipeline


def _write_test_bmp(path: Path, width: int, height: int) -> None:
    row_size = (width * 3 + 3) & ~3
    pixel_size = row_size * height
    file_size = 14 + 40 + pixel_size

    file_header = struct.pack("<2sIHHI", b"BM", file_size, 0, 0, 54)
    info_header = struct.pack(
        "<IIIHHIIIIII",
        40,
        width,
        height,
        1,
        24,
        0,
        pixel_size,
        2835,
        2835,
        0,
        0,
    )
    pixels = b"\x00" * pixel_size
    path.write_bytes(file_header + info_header + pixels)


class DatasetPrepareTests(unittest.TestCase):
    def test_prepare_writes_labels_in_place_and_symlinks_split_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xml_root = root / "xml"
            image_root = root / "images"
            output_root = root / "dataset"
            xml_root.mkdir(parents=True, exist_ok=True)
            image_root.mkdir(parents=True, exist_ok=True)

            image_path = image_root / "sample.bmp"
            _write_test_bmp(image_path, width=8, height=6)

            xml_path = xml_root / "sample.xml"
            xml_path.write_text(
                """
<annotation>
  <filename>sample.bmp</filename>
  <size>
    <width>8</width>
    <height>6</height>
    <depth>3</depth>
  </size>
  <object>
    <name>cat</name>
    <bndbox>
      <xmin>1</xmin>
      <ymin>1</ymin>
      <xmax>7</xmax>
      <ymax>5</ymax>
    </bndbox>
  </object>
</annotation>
""".strip(),
                encoding="utf-8",
            )

            result = prepare_dataset_pipeline(
                output_root=output_root,
                xml_root=xml_root,
                image_root=image_root,
                class_names=["cat"],
                split_mode="deterministic",
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1,
            )

            self.assertFalse((output_root / "images_raw").exists())
            self.assertFalse((output_root / "labels_raw").exists())
            self.assertEqual(result["health"]["status"], "pass")

            xml_label = xml_path.with_suffix(".txt")
            image_label = image_path.with_suffix(".txt")
            self.assertTrue(xml_label.exists())
            self.assertTrue(image_label.exists())
            self.assertEqual(
                xml_label.read_text(encoding="utf-8"),
                image_label.read_text(encoding="utf-8"),
            )

            linked_images = list((output_root / "images").rglob("*.bmp"))
            linked_labels = list((output_root / "labels").rglob("*.txt"))
            self.assertEqual(len(linked_images), 1)
            self.assertEqual(len(linked_labels), 1)
            self.assertTrue(linked_images[0].is_symlink())
            self.assertTrue(linked_labels[0].is_symlink())
            self.assertEqual(linked_images[0].resolve(), image_path.resolve())
            self.assertEqual(linked_labels[0].resolve(), xml_label.resolve())


if __name__ == "__main__":
    unittest.main()
