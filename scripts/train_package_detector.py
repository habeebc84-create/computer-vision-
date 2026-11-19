"""Reproduce the packaged YOLOv8n detector. See models/package-model.md."""

from pathlib import Path
import json
import random
import shutil

import torch
from ultralytics import YOLO


def convert_labels(src, dest):
    boxes = []
    for line in src.read_text().splitlines():
        category, *vertices = line.split()
        coords = [float(p) for p in vertices]
        xs, ys = coords[::2], coords[1::2]
        left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
        box = ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)
        boxes.append(category + " " + " ".join(f"{v:.6f}" for v in box))
    dest.write_text("\n".join(boxes) + "\n")


def prepare_dataset(root, seed):
    rng = random.Random(seed)
    sample = root / "package-boxes"
    sample.mkdir()  # Start with an empty split on every run.
    for split, limit in [("train", 256), ("val", 72), ("test", None)]:
        groups = {}
        for image in sorted((root / "images" / split).glob("*.jpg")):
            groups.setdefault(image.name.split(".rf.")[0], []).append(image)
        photos = [rng.choice(group) for group in groups.values()]
        if limit is not None:
            photos = rng.sample(photos, min(limit, len(photos)))
        image_dir = sample / "images" / split
        label_dir = sample / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for photo in photos:
            shutil.copyfile(photo, image_dir / photo.name)
            label = photo.with_suffix(".txt").name
            convert_labels(root / "labels" / split / label, label_dir / label)

    config = sample / "data.yaml"
    # JSON is also valid YAML. Quoting a resolved path handles spaces on Windows.
    config.write_text(json.dumps({
        "path": str(sample), "train": "images/train", "val": "images/val",
        "test": "images/test", "names": ["package"],
    }))
    return config


def main():
    root = Path.cwd()
    seed = 17
    config = prepare_dataset(root, seed)
    torch.set_num_threads(2)
    run = root / "fit"
    model = YOLO(root / "yolov8n.pt")
    model.train(
        data=str(config), epochs=5, imgsz=320, batch=8, device="cpu",
        workers=0, freeze=10, optimizer="AdamW", lr0=0.001, warmup_epochs=1.0,
        close_mosaic=0, plots=False, amp=False, seed=seed,
        project=str(run), name="package", exist_ok=True, verbose=False,
    )
    best = YOLO(run / "package" / "weights" / "best.pt")
    metrics = best.val(
        data=str(config), split="test", imgsz=320, rect=False, device="cpu",
        batch=8, workers=0, plots=False, project=str(run), name="test",
    )
    (root / "metrics.json").write_text(json.dumps(metrics.results_dict, indent=2))
    best.export(format="onnx", imgsz=320, batch=1, device="cpu", simplify=False, opset=17)


if __name__ == "__main__":
    main()
