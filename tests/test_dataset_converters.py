"""Tests for the dataset annotation converters in backend/datasets.py.

The converters are pure functions (no disk or network), so they are tested
with small synthetic inputs. A full VOC / COCO / WIDER / VisDrone import was
NOT run here — that would download hundreds of MB to GBs.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, BACKEND / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


datasets = _load("datasets_under_test", "datasets.py")
coco_import = _load("coco_import_under_test", "coco_import.py")

VOC_XML = """<annotation>
  <size><width>640</width><height>480</height></size>
  <object><name>person</name><difficult>0</difficult>
    <bndbox><xmin>64</xmin><ymin>48</ymin><xmax>192</xmax><ymax>144</ymax></bndbox></object>
  <object><name>dog</name><difficult>1</difficult>
    <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>10</xmax><ymax>10</ymax></bndbox></object>
  <object><name>spaceship</name><difficult>0</difficult>
    <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>10</xmax><ymax>10</ymax></bndbox></object>
</annotation>"""

COCO_DOC = {
    "images": [
        {"id": 1, "file_name": "a.jpg", "width": 100, "height": 200},
        {"id": 2, "file_name": "b.jpg", "width": 50, "height": 50},
    ],
    "categories": [
        {"id": 2, "name": "dog"},
        {"id": 5, "name": "cat"},
        {"id": 1, "name": "person"},
    ],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 5, "bbox": [10, 20, 30, 40]},
        {"id": 2, "image_id": 1, "category_id": 5, "bbox": [90, 190, 50, 50]},
        {"id": 3, "image_id": 1, "category_id": 2, "bbox": [0, 0, 10, 10], "iscrowd": 1},
        {"id": 4, "image_id": 2, "category_id": 1, "bbox": [0, 0, 50, 50]},
    ],
}

# LVIS v1 style: no file_name, coco_url instead; categories are 1-based.
LVIS_DOC = {
    "images": [
        {"id": 397133, "width": 640, "height": 427,
         "coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg"},
    ],
    "categories": [{"id": 1, "name": "aerosol_can"}, {"id": 2, "name": "air_conditioner"}],
    "annotations": [
        {"id": 1, "image_id": 397133, "category_id": 2, "bbox": [0, 0, 64, 43]},
    ],
}

WIDER_TXT = """0--Parade/0_Parade_1.jpg
2
10 20 30 40 0 0 0 0 0 0
50 60 10 10 0 0 0 1 0 0
1--Handshaking/1_Handshaking_1.jpg
1
0 0 5 5 0 0 0 0 0 0
"""

VISDRONE_TXT = """10,20,100,50,0,1,0,0
10,20,100,50,1,2,0,0
900,400,200,200,0,10,0,0
"""


class YoloLineTests(unittest.TestCase):
    def test_clips_to_image(self):
        line = datasets._yolo_line(3, 90, 190, 150, 250, 100, 200)
        self.assertEqual(line, "3 0.950000 0.975000 0.100000 0.050000")

    def test_degenerate_is_none(self):
        self.assertIsNone(datasets._yolo_line(0, 10, 10, 10, 20, 100, 100))
        self.assertIsNone(datasets._yolo_line(0, -5, -5, 0, 0, 100, 100))


class VocTests(unittest.TestCase):
    def test_valid_object_only(self):
        lines = datasets.voc_xml_to_yolo(VOC_XML, datasets.VOC20)
        # person only: difficult dog skipped, unknown "spaceship" skipped.
        self.assertEqual(lines, ["14 0.200000 0.200000 0.200000 0.200000"])


class CocoTests(unittest.TestCase):
    def test_classes_sorted_by_id_and_crowd_skipped(self):
        names, labels = datasets.coco_json_to_yolo(COCO_DOC)
        self.assertEqual(names, ["person", "dog", "cat"])
        self.assertEqual(labels["a.jpg"], [
            "2 0.250000 0.200000 0.300000 0.200000",
            "2 0.950000 0.975000 0.100000 0.050000",
        ])
        self.assertEqual(labels["b.jpg"], ["0 0.500000 0.500000 1.000000 1.000000"])

    def test_lvis_style_coco_url_name(self):
        names, labels = datasets.coco_json_to_yolo(LVIS_DOC)
        self.assertEqual(names, ["aerosol_can", "air_conditioner"])
        # 64x43 box in a 640x427 image: cx 32/640, cy 21.5/427, w 64/640, h 43/427.
        self.assertEqual(list(labels), ["000000397133.jpg"])
        self.assertEqual(labels["000000397133.jpg"],
                         ["1 0.050000 0.050351 0.100000 0.100703"])

    def test_match_image_strips_coco_prefix(self):
        by_stem = {"000000000139": Path("/tmp/000000000139.jpg")}
        self.assertEqual(
            datasets._match_image("COCO_val2017_000000000139", by_stem),
            Path("/tmp/000000000139.jpg"))
        self.assertEqual(
            datasets._match_image("000000000139", by_stem),
            Path("/tmp/000000000139.jpg"))
        self.assertIsNone(datasets._match_image("nope", by_stem))


class WiderTests(unittest.TestCase):
    def test_parses_and_skips_invalid_faces(self):
        entries = datasets.wider_txt_to_yolo(WIDER_TXT)
        self.assertEqual(entries, {
            "0--Parade/0_Parade_1.jpg": [(10, 20, 40, 60)],
            "1--Handshaking/1_Handshaking_1.jpg": [(0, 0, 5, 5)],
        })


class VisdroneTests(unittest.TestCase):
    def test_ignored_regions_skipped_and_class_is_1_based(self):
        lines = datasets.visdrone_txt_to_yolo(VISDRONE_TXT, 1000, 500, datasets.VISDRONE10)
        self.assertEqual(lines, [
            "0 0.060000 0.090000 0.100000 0.100000",
            "9 0.950000 0.900000 0.100000 0.200000",
        ])


class CocoImportNameTests(unittest.TestCase):
    def test_image_file_name_falls_back_to_coco_url(self):
        self.assertEqual(
            coco_import.image_file_name({"id": 397133, "coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg"}),
            "000000397133.jpg")
        self.assertEqual(
            coco_import.image_file_name({"id": 1, "file_name": "a.jpg"}), "a.jpg")


class CatalogTests(unittest.TestCase):
    def test_every_entry_is_complete(self):
        self.assertEqual(len(datasets.CATALOG), 17)
        for entry in datasets.CATALOG:
            for field in ("id", "name", "description", "images", "classes",
                          "size_mb", "format", "url", "license"):
                self.assertIn(field, entry, entry["id"])
            self.assertIn(entry["format"],
                          {"sample", "yolo", "voc", "coco", "wider", "visdrone", "external"})
            if entry["format"] == "external":
                self.assertIn("external_url", entry, entry["id"])
                self.assertIn("note", entry, entry["id"])
            else:
                self.assertTrue(entry["url"].startswith(("http://", "https://")), entry["id"])

    def test_external_download_refuses(self):
        with self.assertRaises(ValueError):
            datasets.start_download("open-images-v7")
        with self.assertRaises(KeyError):
            datasets.start_download("no-such-dataset")


if __name__ == "__main__":
    unittest.main()
