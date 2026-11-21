"""Dataset preparation: upload, annotation (YOLO txt), split, data.yaml, stats, export.
Layout: dataset/images/{train,val}, dataset/labels/{train,val}, dataset/data.yaml
Annotations are stored per-image as YOLO txt (class cx cy w h, normalised)
alongside the image in dataset/labels/all/ until split, plus a JSON sidecar
for the annotation editor."""
import hashlib, json, math, random, shutil, zipfile
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import cv2

from image_processing import decode_image

BASE = Path(__file__).resolve().parent.parent
DS = BASE / "dataset"
IMG_ALL = DS / "images" / "all"
LBL_ALL = DS / "labels" / "all"
CLASSES_FILE = DS / "classes.json"
ALLOWED = {".jpg", ".jpeg", ".png"}

def ensure_dirs():
    for d in [IMG_ALL, LBL_ALL, DS/"images"/"train", DS/"images"/"val",
              DS/"labels"/"train", DS/"labels"/"val"]:
        d.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

def insert_image(filename, data):
    """Store a validated image with its own YOLO label stem."""
    img = decode_image(data)
    original = Path(filename.replace("\\", "/")).name
    stem = Path(original).stem[:120] or "image"
    # Normalize to PNG so training and annotation read the same pixels and orientation.
    ok, encoded = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Could not encode the image.")

    with NamedTemporaryFile(dir=IMG_ALL, suffix=".tmp", delete=False) as temp:
        staged = Path(temp.name)
    try:
        staged.write_bytes(encoded.tobytes())
        existing_stems = {path.stem.casefold() for path in IMG_ALL.iterdir()
                          if path.suffix.lower() in ALLOWED}
        existing_stems.update(path.stem.casefold() for path in LBL_ALL.glob("*.txt"))
        candidate = stem
        if candidate.casefold() in existing_stems:
            candidate = f"{stem}_{uuid4().hex[:12]}"
        while True:
            destination = IMG_ALL / f"{candidate}.png"
            try:
                destination.hardlink_to(staged)
                return destination.name
            except FileExistsError:
                candidate = f"{stem}_{uuid4().hex[:12]}"
    finally:
        staged.unlink(missing_ok=True)

def get_classes():
    if CLASSES_FILE.exists():
        try: return json.loads(CLASSES_FILE.read_text())
        except Exception: return []
    return []

def set_classes(classes):
    classes = [str(c).strip() for c in classes if str(c).strip()]
    existing = get_classes()
    if classes[:len(existing)] != existing:
        raise ValueError("Keep existing classes in their original order so saved boxes retain their labels.")
    seen = {name.casefold() for name in existing}
    for name in classes[len(existing):]:
        key = name.casefold()
        if key in seen:
            raise ValueError("Class names must be unique.")
        seen.add(key)
    CLASSES_FILE.write_text(json.dumps(classes, indent=2))
    return classes

def list_images():
    ensure_dirs()
    out = []
    for p in sorted(IMG_ALL.glob("*")):
        if p.suffix.lower() in ALLOWED:
            img = cv2.imread(str(p))
            h, w = (img.shape[:2] if img is not None else (0, 0))
            lab = LBL_ALL / (p.stem + ".txt")
            n = 0
            if lab.exists():
                n = len([l for l in lab.read_text().splitlines() if l.strip()])
            out.append({"name": p.name, "width": w, "height": h,
                        "annotations": n, "size_kb": round(p.stat().st_size/1024, 1)})
    return out

def save_annotation(image_name, boxes):
    """boxes: [{class_id, x_min,y_min,x_max,y_max}] in pixels. Writes YOLO txt."""
    ensure_dirs()
    img = cv2.imread(str(IMG_ALL / image_name))
    if img is None: raise FileNotFoundError(f"Image not found: {image_name}")
    h, w = img.shape[:2]
    lines = []
    num_classes = len(get_classes())
    for b in boxes:
        x1, y1, x2, y2 = float(b["x_min"]), float(b["y_min"]), float(b["x_max"]), float(b["y_max"])
        cls = b["class_id"]
        if type(cls) is not int or not 0 <= cls < num_classes:
            raise ValueError("Choose a defined class for every box.")
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError("Draw each box inside the image with positive width and height.")
        cx, cy = ((x1+x2)/2)/w, ((y1+y2)/2)/h
        bw, bh = (x2-x1)/w, (y2-y1)/h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    (LBL_ALL / (Path(image_name).stem + ".txt")).write_text("\n".join(lines))
    return len(lines)

def get_annotations(image_name):
    img = cv2.imread(str(IMG_ALL / image_name))
    if img is None: raise FileNotFoundError(image_name)
    h, w = img.shape[:2]
    lab = LBL_ALL / (Path(image_name).stem + ".txt")
    boxes = []
    if lab.exists():
        for line in lab.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5: continue
            cid, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
            boxes.append({"class_id": cid, "x_min": round((cx-bw/2)*w, 1),
                          "y_min": round((cy-bh/2)*h, 1), "x_max": round((cx+bw/2)*w, 1),
                          "y_max": round((cy+bh/2)*h, 1)})
    return {"image": image_name, "width": w, "height": h, "boxes": boxes}

def stats():
    ensure_dirs()
    imgs = list_images()
    classes = get_classes()
    per_class = {c: 0 for c in classes}
    total_ann = 0
    for im in imgs:
        lab = LBL_ALL / (Path(im["name"]).stem + ".txt")
        if lab.exists():
            for line in lab.read_text().splitlines():
                if line.strip():
                    try:
                        cid = int(line.split()[0])
                    except ValueError:
                        continue
                    if 0 <= cid < len(classes):
                        total_ann += 1
                        per_class[classes[cid]] += 1
    train_n = len(list((DS/"images"/"train").glob("*")))
    val_n = len(list((DS/"images"/"val").glob("*")))
    return {"total_images": len(imgs), "total_annotations": total_ann,
            "num_classes": len(classes), "classes": classes,
            "per_class": per_class, "train_images": train_n, "val_images": val_n,
            "annotated_images": sum(1 for i in imgs if i["annotations"] > 0)}

def _source_images():
    return sorted(path for path in IMG_ALL.iterdir() if path.is_file() and path.suffix.lower() in ALLOWED)


def _dataset_version(images):
    digest = hashlib.sha256(json.dumps(get_classes()).encode())
    for image in images:
        info = image.stat()
        digest.update(json.dumps([image.name, info.st_size, info.st_mtime_ns]).encode())
        label = LBL_ALL / f"{image.stem}.txt"
        digest.update(b"+")
        if label.is_file():
            digest.update(b"label:")
            digest.update(label.read_bytes())
    return digest.hexdigest()


def require_current_split():
    prepared = DS / "data.yaml"
    if not prepared.is_file():
        raise FileNotFoundError("Prepare and split the annotated dataset before training.")
    version = f"# dataset-version: {_dataset_version(_source_images())}"
    if version not in prepared.read_text().splitlines():
        raise ValueError("The split is out of date. In Dataset, split the current images and labels again.")


def _training_examples(images, num_classes):
    examples = {}
    unreviewed = []
    duplicates = []
    for image in images:
        label = LBL_ALL / f"{image.stem}.txt"
        if not label.is_file():
            unreviewed.append(image.name)
            continue
        rows = set()
        for text in label.read_text().splitlines():
            if not text.strip():
                continue
            try:
                raw_class, *raw_box = text.split()
                cls = int(raw_class)
                cx, cy, w, h = map(float, raw_box)
                valid = (0 <= cls < num_classes and 0 < w <= 1 and 0 < h <= 1
                         and w / 2 - 1e-6 <= cx <= 1 - w / 2 + 1e-6
                         and h / 2 - 1e-6 <= cy <= 1 - h / 2 + 1e-6)
                if not valid:
                    raise ValueError
            except ValueError:
                raise ValueError(f"Invalid box in {label.name}. Review its class and coordinates.") from None
            rows.add((cls, cx, cy, w, h))
        digest = hashlib.sha256()
        with image.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        key = digest.hexdigest()
        if key in examples:
            previous, previous_rows = examples[key]
            if previous_rows != rows:
                raise ValueError(f"{previous.name} and {image.name} are identical images with different boxes. Review the annotations.")
            duplicates.append(image.name)
        else:
            examples[key] = (image, rows)
    return list(examples.values()), unreviewed, duplicates


def _choose_validation(groups, val_size):
    # Never move the final example of any class out of training.
    remaining = Counter(cls for group in groups for cls in group)
    candidates = {cls: [] for cls in remaining}
    for index, group in enumerate(groups):
        for cls in group:
            candidates[cls].append(index)
    validation = set()
    covered = set()

    def move(index):
        if (len(validation) == val_size or index in validation
                or any(remaining[cls] < 2 for cls in groups[index])):
            return False
        validation.add(index)
        for cls in groups[index]:
            remaining[cls] -= 1
        covered.update(groups[index])
        return True

    # Prefer coverage of the classes with the fewest distinct images.
    for cls in sorted(remaining, key=lambda cls: (remaining[cls], cls)):
        if cls not in covered:
            for index in candidates[cls]:
                if move(index):
                    break
    for index in range(len(groups)):
        if len(validation) == val_size:
            break
        move(index)
    return validation


def split_dataset(train_pct=80):
    """Build a repeatable split from reviewed images, keeping every labeled class in training."""
    if not math.isfinite(train_pct) or not 0 < train_pct < 100:
        raise ValueError("The training percentage must be between 0 and 100.")
    ensure_dirs()
    images = _source_images()
    classes = get_classes()
    if not classes:
        raise ValueError("Add at least one class first.")
    version = _dataset_version(images)
    examples, unreviewed, duplicates = _training_examples(images, len(classes))
    if len(examples) < 2:
        raise ValueError("Review annotations for at least two distinct images before splitting. Save an empty box list for a confirmed background image.")
    random.Random(42).shuffle(examples)
    groups = [{row[0] for row in rows} for _, rows in examples]
    labeled = set().union(*groups)
    if not labeled:
        raise ValueError("Add bounding boxes for at least one class before splitting.")
    n_train = max(1, min(len(examples) - 1, round(len(examples) * train_pct / 100)))
    validation = _choose_validation(groups, len(examples) - n_train)
    if not validation:
        raise ValueError("Each image contains a class with no other example. Add labeled examples to create a validation set.")
    val_classes = set().union(*(groups[i] for i in validation))
    if not val_classes:
        raise ValueError("Validation would contain only backgrounds. Add more labeled examples of the objects you want to measure.")
    if version != _dataset_version(_source_images()):
        raise ValueError("Images or labels changed during the split. Run the split again.")

    config = DS / "data.yaml"
    config.unlink(missing_ok=True)
    for sub in ("train", "val"):
        for folder in ("images", "labels"):
            for path in (DS / folder / sub).iterdir():
                if path.is_file():
                    path.unlink()
    for index, (image, rows) in enumerate(examples):
        group = "val" if index in validation else "train"
        shutil.copy2(image, DS / "images" / group / image.name)
        with (DS / "labels" / group / f"{image.stem}.txt").open("w") as out:
            for cls, *coords in sorted(rows):
                out.write(f"{cls} " + " ".join(f"{v:.9g}" for v in coords) + "\n")
    # Keep original class IDs for compatibility with the annotation editor.
    yaml = (f"# dataset-version: {version}\n"
            f"path: {json.dumps(str(DS))}\ntrain: images/train\nval: images/val\n"
            f"nc: {len(classes)}\nnames: {json.dumps(classes, ensure_ascii=False)}\n")
    config.write_text(yaml)
    return {
        "train": len(examples) - len(validation), "val": len(validation), "data_yaml": str(config),
        "unannotated_images": unreviewed, "duplicate_images": duplicates,
        "warning": f"{len(unreviewed)} unreviewed images excluded." if unreviewed else None,
        "classes_without_boxes": [classes[i] for i in range(len(classes)) if i not in labeled],
        "classes_without_validation": [classes[i] for i in sorted(labeled - val_classes)],
    }

def export_zip(dest: Path):
    ensure_dirs()
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for p in DS.rglob("*"):
            if p.is_file() and "all" not in p.parts and p.suffix != ".cache":
                z.write(p, p.relative_to(DS))
    return dest
