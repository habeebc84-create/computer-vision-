"""Regression tests for annotation integrity and train/validation coverage."""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import dataset_manager as dataset


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "DS", tmp_path)
    monkeypatch.setattr(dataset, "IMG_ALL", tmp_path / "images" / "all")
    monkeypatch.setattr(dataset, "LBL_ALL", tmp_path / "labels" / "all")
    monkeypatch.setattr(dataset, "CLASSES_FILE", tmp_path / "classes.json")
    dataset.ensure_dirs()
    dataset.set_classes(["pen", "comb", "phone case"])
    return tmp_path


def image(name, color, class_id=None):
    cv2.imwrite(str(dataset.IMG_ALL / name), np.full((20, 20, 3), color, dtype=np.uint8))
    if class_id is not None:
        dataset.save_annotation(name, [
            {"class_id": class_id, "x_min": 3, "y_min": 4, "x_max": 10, "y_max": 12},
        ])


def test_split_excludes_unreviewed_and_duplicates_and_protects_rare_class(workspace):
    image("pen-a.png", 50, 0)
    image("pen-b.png", 60, 0)
    image("one-comb.png", 70, 1)
    image("reviewed-background.png", 80)
    dataset.save_annotation("reviewed-background.png", [])
    image("unreviewed.png", 90)
    shutil.copyfile(dataset.IMG_ALL / "pen-a.png", dataset.IMG_ALL / "pen-a-copy.png")
    dataset.save_annotation("pen-a-copy.png", dataset.get_annotations("pen-a.png")["boxes"])

    result = dataset.split_dataset(50)
    assert result["unannotated_images"] == ["unreviewed.png"]
    # The deterministic name order retains one of the identical files.
    assert len(result["duplicate_images"]) == 1
    assert result["classes_without_boxes"] == ["phone case"]
    assert result["classes_without_validation"] == ["comb"]
    assert (workspace / "images" / "train" / "one-comb.png").is_file()
    assert set(p.name for p in (workspace / "images" / "val").glob("*.png")) - {"reviewed-background.png"}
    dataset.require_current_split()

    dataset.save_annotation("pen-a.png", [])
    with pytest.raises(ValueError, match="out of date"):
        dataset.require_current_split()
    with pytest.raises(ValueError, match="identical images with different boxes"):
        dataset.split_dataset(50)


def test_class_order_and_annotation_coordinates_are_checked(workspace):
    image("pen.png", 50, 0)
    with pytest.raises(ValueError, match="original order"):
        dataset.set_classes(["comb", "pen", "phone case"])
    with pytest.raises(ValueError, match="unique"):
        dataset.set_classes(["pen", "comb", "phone case", "Pen"])
    with pytest.raises(ValueError, match="positive width"):
        dataset.save_annotation("pen.png", [
            {"class_id": 0, "x_min": 12, "y_min": 4, "x_max": 10, "y_max": 12},
        ])
    with pytest.raises(ValueError, match="defined class"):
        dataset.save_annotation("pen.png", [
            {"class_id": 3, "x_min": 3, "y_min": 4, "x_max": 10, "y_max": 12},
        ])
    assert dataset.get_annotations("pen.png")["boxes"][0]["class_id"] == 0
