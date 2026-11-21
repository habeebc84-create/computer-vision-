"""Import a labeled COCO-format archive, including Objects365."""
import json
import math
from pathlib import PurePosixPath
from zipfile import ZipFile

from dataset_manager import MAX_UPLOAD_BYTES, ensure_dirs, get_classes, insert_image, set_classes

MAX_ARCHIVE_BYTES = 100 * 1024 * 1024


def import_coco(archive):
    from dataset_manager import LBL_ALL

    with ZipFile(archive) as bundle:
        entries = [entry for entry in bundle.infolist() if not entry.is_dir()]
        if sum(entry.file_size for entry in entries) > 4 * MAX_ARCHIVE_BYTES:
            raise ValueError("The uncompressed dataset exceeds 400 MB. Import a smaller subset.")
        metadata = [entry for entry in entries if entry.filename.lower().endswith(".json")]
        if len(metadata) != 1:
            raise ValueError("Include one COCO annotations JSON file and its images in the ZIP.")
        doc = json.loads(bundle.read(metadata[0]))
        images = {image["id"]: image for image in doc["images"]}
        categories = {item["id"]: item["name"].strip() for item in doc["categories"]}
        if not images or not categories or not all(categories.values()):
            raise ValueError("The dataset must contain images and named categories.")

        classes = get_classes()
        class_ids = {}
        for i, name in enumerate(classes):
            class_ids.setdefault(name.casefold(), i)
        indices = {}
        for category, name in categories.items():
            key = name.casefold()
            if key not in class_ids:
                class_ids[key] = len(classes)
                classes.append(name)
            indices[category] = class_ids[key]
        labels = {image_id: [] for image_id in images}
        for annotation in doc["annotations"]:
            image = images[annotation["image_id"]]
            width, height = float(image["width"]), float(image["height"])
            x, y, w, h = map(float, annotation["bbox"])
            if min(width, height) <= 0 or not all(map(math.isfinite, (x, y, w, h, width, height))):
                raise ValueError("The dataset contains invalid image dimensions or boxes.")
            left, top = max(0, x), max(0, y)
            right, bottom = min(width, x + w), min(height, y + h)
            if right <= left or bottom <= top or annotation.get("iscrowd"):
                continue
            coords = ((left + right) / (2 * width), (top + bottom) / (2 * height),
                      (right - left) / width, (bottom - top) / height)
            cls = indices[annotation["category_id"]]
            labels[annotation["image_id"]].append(f"{cls} " + " ".join(f"{v:.6f}" for v in coords))

        files = {}
        for entry in entries:
            name = PurePosixPath(entry.filename).name
            if name in files:
                files[name] = None
            else:
                files[name] = entry

        ensure_dirs()
        saved, skipped = [], []
        set_classes(classes)
        for image_id, image in images.items():
            name = PurePosixPath(image["file_name"]).name
            entry = files.get(name)
            if entry is None or entry.file_size > MAX_UPLOAD_BYTES:
                skipped.append(name)
                continue
            try:
                stored = insert_image(name, bundle.read(entry))
            except ValueError:
                skipped.append(name)
                continue
            (LBL_ALL / (PurePosixPath(stored).stem + ".txt")).write_text("\n".join(labels[image_id]))
            saved.append(stored)
    return {"saved": saved, "skipped": skipped, "classes": len(categories)}
