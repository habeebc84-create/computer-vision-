"""Dataset preparation: upload, annotation (YOLO txt), split, data.yaml, stats, export.
Layout: dataset/images/{train,val}, dataset/labels/{train,val}, dataset/data.yaml
Annotations are stored per-image as YOLO txt (class cx cy w h, normalised)
alongside the image in dataset/labels/all/ until split, plus a JSON sidecar
for the annotation editor."""
import json, random, shutil, zipfile
from pathlib import Path

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

def get_classes():
    if CLASSES_FILE.exists():
        try: return json.loads(CLASSES_FILE.read_text())
        except Exception: return []
    return []

def set_classes(classes):
    classes = [str(c).strip() for c in classes if str(c).strip()]
    CLASSES_FILE.write_text(json.dumps(classes, indent=2)); return classes

def list_images():
    ensure_dirs()
    out = []
    for p in sorted(IMG_ALL.glob("*")):
        if p.suffix.lower() in ALLOWED:
            import cv2
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
    import cv2
    img = cv2.imread(str(IMG_ALL / image_name))
    if img is None: raise FileNotFoundError(f"Image not found: {image_name}")
    h, w = img.shape[:2]
    lines = []
    for b in boxes:
        x1, y1, x2, y2 = float(b["x_min"]), float(b["y_min"]), float(b["x_max"]), float(b["y_max"])
        cx, cy = ((x1+x2)/2)/w, ((y1+y2)/2)/h
        bw, bh = (x2-x1)/w, (y2-y1)/h
        lines.append(f"{int(b['class_id'])} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    (LBL_ALL / (Path(image_name).stem + ".txt")).write_text("\n".join(lines))
    return len(lines)

def get_annotations(image_name):
    import cv2
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
                    total_ann += 1
                    cid = int(line.split()[0])
                    if 0 <= cid < len(classes): per_class[classes[cid]] += 1
    train_n = len(list((DS/"images"/"train").glob("*")))
    val_n = len(list((DS/"images"/"val").glob("*")))
    return {"total_images": len(imgs), "total_annotations": total_ann,
            "num_classes": len(classes), "classes": classes,
            "per_class": per_class, "train_images": train_n, "val_images": val_n,
            "annotated_images": sum(1 for i in imgs if i["annotations"] > 0)}

def split_dataset(train_pct=80):
    """Copy all/ images+labels into train/val, write data.yaml. Validates consistency."""
    ensure_dirs()
    imgs = list_images()
    if not imgs: raise ValueError("No images uploaded yet — upload images first.")
    if len(imgs) < 2: raise ValueError("A train/validation split needs at least two images.")
    classes = get_classes()
    if not classes: raise ValueError("No classes defined — add at least one class first.")
    missing = [i["name"] for i in imgs if i["annotations"] == 0]
    rng = random.Random(42); pool = imgs[:]; rng.shuffle(pool)
    n_train = max(1, min(len(pool) - 1, round(len(pool) * train_pct / 100)))
    train, val = pool[:n_train], pool[n_train:]
    for sub in ("train", "val"):
        for p in (DS/"images"/sub).glob("*"): p.unlink()
        for p in (DS/"labels"/sub).glob("*"): p.unlink()
    for group, items in (("train", train), ("val", val)):
        for im in items:
            shutil.copy2(IMG_ALL/im["name"], DS/"images"/group/im["name"])
            src = LBL_ALL/(Path(im["name"]).stem+".txt")
            if src.exists(): shutil.copy2(src, DS/"labels"/group/src.name)
            else: (DS/"labels"/group/(Path(im["name"]).stem+".txt")).write_text("")
    yaml = (f"# YOLO dataset — generated by the Dataset Preparation module\n"
            f"path: {DS}\ntrain: images/train\nval: images/val\n\n"
            f"nc: {len(classes)}\nnames: {classes}\n")
    (DS/"data.yaml").write_text(yaml)
    return {"train": len(train), "val": len(val), "data_yaml": str(DS/"data.yaml"),
            "unannotated_images": missing,
            "warning": (f"{len(missing)} image(s) have no annotations and were still split; "
                        "annotate them for a useful model.") if missing else None}

def export_zip(dest: Path):
    ensure_dirs()
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for p in DS.rglob("*"):
            if p.is_file() and "all" not in p.parts and p.suffix != ".cache":
                z.write(p, p.relative_to(DS))
    return dest
