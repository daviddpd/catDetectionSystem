from __future__ import annotations

import struct
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from cds.training.dataset.xml_to_yolo import convert_voc_xml_to_yolo


def _write_test_bmp(path: Path, width: int, height: int) -> None:
    row_size = (width * 3 + 3) & ~3
    pixel_size = row_size * height
    file_size = 14 + 40 + pixel_size

    file_header = struct.pack("<2sIHHI", b"BM", file_size, 0, 0, 54)
    info_header = struct.pack("<IIIHHIIIIII", 40, width, height, 1, 24, 0, pixel_size, 2835, 2835, 0, 0)
    pixels = b"\x00" * pixel_size
    path.write_bytes(file_header + info_header + pixels)


class XmlToYoloConversionTests(unittest.TestCase):
    def test_repairs_missing_xml_size_from_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xml_root = root / "xml"
            output_root = root / "out"
            xml_root.mkdir(parents=True, exist_ok=True)

            image_path = xml_root / "sample.bmp"
            _write_test_bmp(image_path, width=4, height=3)

            xml_path = xml_root / "sample.xml"
            xml_path.write_text(
                """
<annotation>
  <filename>sample.bmp</filename>
  <object>
    <name>cat</name>
    <bndbox>
      <xmin>1</xmin>
      <ymin>1</ymin>
      <xmax>3</xmax>
      <ymax>2</ymax>
    </bndbox>
  </object>
</annotation>
""".strip(),
                encoding="utf-8",
            )

            result = convert_voc_xml_to_yolo(
                xml_root=xml_root,
                output_root=output_root,
                class_names=["cat"],
                image_root=xml_root,
                copy_images=False,
            )

            self.assertEqual(result["stats"]["labels_written"], 1)
            self.assertEqual(result["stats"]["repaired_missing_size"], 1)
            self.assertEqual(result["stats"]["skipped_invalid_size"], 0)
            self.assertIn(str(xml_path), result["size_repair"]["repaired_xml"])

            tree = ET.parse(str(xml_path))
            parsed = tree.getroot()
            self.assertEqual(parsed.findtext("size/width"), "4")
            self.assertEqual(parsed.findtext("size/height"), "3")

            self.assertEqual(len(result["manifest"]), 1)
            label_path = Path(result["manifest"][0]["label"])
            self.assertTrue(label_path.exists())
            self.assertTrue(label_path.read_text(encoding="utf-8").strip().startswith("0 "))

    def test_skips_pair_when_size_cannot_be_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xml_root = root / "xml"
            output_root = root / "out"
            xml_root.mkdir(parents=True, exist_ok=True)

            image_path = xml_root / "broken.jpg"
            image_path.write_bytes(b"not-an-image")

            xml_path = xml_root / "broken.xml"
            xml_path.write_text(
                """
<annotation>
  <filename>broken.jpg</filename>
  <object>
    <name>cat</name>
    <bndbox>
      <xmin>0</xmin>
      <ymin>0</ymin>
      <xmax>1</xmax>
      <ymax>1</ymax>
    </bndbox>
  </object>
</annotation>
""".strip(),
                encoding="utf-8",
            )

            result = convert_voc_xml_to_yolo(
                xml_root=xml_root,
                output_root=output_root,
                class_names=["cat"],
                image_root=xml_root,
                copy_images=False,
            )

            self.assertEqual(result["stats"]["labels_written"], 0)
            self.assertEqual(result["stats"]["repaired_missing_size"], 0)
            self.assertEqual(result["stats"]["skipped_invalid_size"], 1)
            self.assertIn(str(xml_path), result["size_repair"]["skipped_xml"])


if __name__ == "__main__":
    unittest.main()
