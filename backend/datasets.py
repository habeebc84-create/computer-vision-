"""Curated datasets with real images and bounding-box annotations."""
import json, shutil, threading, urllib.request, zipfile
from pathlib import Path
from objects365_sample import CATEGORIES, IMAGE_IDS, SAMPLE_ID, SAMPLE_SOURCE, import_objects365_sample
import open_images_sample

BASE = Path(__file__).resolve().parent.parent
DL_DIR = BASE / "downloads"
DL_DIR.mkdir(parents=True, exist_ok=True)

# Standard COCO 80 class names (0-79), as used by yolov8n.pt and by the
# label files inside the COCO sample zips.
COCO80 = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
          "truck", "boat", "traffic light", "fire hydrant", "stop sign",
          "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
          "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
          "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
          "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
          "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
          "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
          "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
          "donut", "cake", "chair", "couch", "potted plant", "bed",
          "dining table", "toilet", "tv", "laptop", "mouse", "remote",
          "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
          "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
          "hair drier", "toothbrush"]

SAMPLE_IMPORTERS = {
    SAMPLE_ID: import_objects365_sample,
    open_images_sample.SAMPLE_ID: open_images_sample.import_open_images_sample,
    open_images_sample.ACCESSORIES_ID: open_images_sample.import_accessories_sample,
}

CATALOG = [
    {"id": open_images_sample.SAMPLE_ID,
     "name": "Open Images · Pens and phones",
     "description": "60 real photos with reviewed pen and cell-phone boxes, including small objects in everyday scenes.",
     "images": len(open_images_sample.SAMPLE_ROWS), "classes": len(open_images_sample.CATEGORIES), "size_mb": 30,
     "url": open_images_sample.SAMPLE_SOURCE,
     "license": "Open Images annotations (CC BY 4.0) and per-image CC BY 2.0 attribution"},
    {"id": open_images_sample.ACCESSORIES_ID,
     "name": "Wikimedia Commons · Combs and phone cases",
     "description": "12 photos with hand-labeled boxes for combs and phone cases.",
     "images": len(open_images_sample.ACCESSORIES_ROWS), "classes": 2, "size_mb": 2,
     "url": open_images_sample.ACCESSORIES_SOURCE,
     "license": "Per-image Creative Commons attribution in the repository manifest"},
    {"id": SAMPLE_ID,
     "name": "Objects365 sample",
     "description": "10 real Objects365 photos with bounding boxes. Use this small sample to check import, annotation, and training.",
     "images": len(IMAGE_IDS), "classes": len(CATEGORIES), "size_mb": 5.2,
     "url": SAMPLE_SOURCE,
     "license": "Objects365 terms and individual image licenses"},
    {"id": "coco128",
     "name": "COCO128",
     "description": "128 images from COCO train 2017 with real YOLO labels across the 80 COCO classes. Recommended for real training — big enough to learn from, small enough to train quickly on CPU.",
     "images": 128, "classes": 80, "size_mb": 6.7,
     "url": "https://ultralytics.com/assets/coco128.zip",
     "license": "COCO 2017, CC BY 4.0"},
    {"id": "coco8",
     "name": "COCO8",
     "description": "Just 8 images (4 train / 4 val) from COCO train 2017. A tiny smoke test — use it to check the whole download → split → train pipeline works before training on COCO128.",
     "images": 8, "classes": 80, "size_mb": 0.4,
     "url": "https://ultralytics.com/assets/coco8.zip",
     "license": "COCO 2017, CC BY 4.0"},
]

_lock = threading.RLock()
_import_lock = threading.Lock()
_state = {}  # id -> {status, progress, imported_images, imported_labels, error}

def _marker_path(ds_id):
    return DL_DIR / f"{ds_id}.imported.json"

def _load_state(ds_id):
    with _lock:
        s = _state.get(ds_id)
    if s:
        return dict(s)
    m = _marker_path(ds_id)
    if m.exists():
        try:
            d = json.loads(m.read_text())
            return {"status": "imported", "progress": 100,
                    "imported_images": d.get("images", 0),
                    "imported_labels": d.get("labels", 0), "error": None}
        except Exception:
            pass
    return {"status": "not-downloaded", "progress": 0,
            "imported_images": 0, "imported_labels": 0, "error": None}

def _set_state(ds_id, **kw):
    with _lock:
        s = _state.setdefault(ds_id, {"status": "not-downloaded", "progress": 0,
                                      "imported_images": 0, "imported_labels": 0,
                                      "error": None})
        s.update(kw)

def catalog():
    out = []
    for d in CATALOG:
        item = dict(d)
        item.update(_load_state(d["id"]))
        out.append(item)
    return out

def status(ds_id):
    d = next((c for c in CATALOG if c["id"] == ds_id), None)
    if not d:
        raise KeyError(ds_id)
    item = dict(d)
    item.update(_load_state(ds_id))
    return item

def start_download(ds_id, force=False):
    d = next((c for c in CATALOG if c["id"] == ds_id), None)
    if not d:
        raise KeyError(ds_id)
    with _lock:
        cur = _load_state(ds_id)
        if cur["status"] == "downloading":
            return status(ds_id)
        if cur["status"] == "imported" and (not force or ds_id in SAMPLE_IMPORTERS):
            return status(ds_id)
        _set_state(ds_id, status="downloading", progress=0, error=None)
        t = threading.Thread(target=_run_import, args=(d,), daemon=True)
        t.start()
    return status(ds_id)

def _run_import(dataset):
    with _import_lock:
        _download_and_import(dataset)


def _download_and_import(d):
    ds_id, url = d["id"], d["url"]
    zip_path = DL_DIR / f"{ds_id}.zip"
    tmp_dir = DL_DIR / f"{ds_id}_extract"
    try:
        if ds_id in SAMPLE_IMPORTERS:
            count = SAMPLE_IMPORTERS[ds_id](lambda progress: _set_state(ds_id, progress=progress))
            _marker_path(ds_id).write_text(json.dumps({"images": count, "labels": count, "source": url}))
            _set_state(ds_id, status="imported", progress=100, imported_images=count, imported_labels=count, error=None)
            return
        # --- download (follows redirects; streams so big files don't blow RAM)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            with open(zip_path, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        _set_state(ds_id, progress=round(done / total * 70, 1))
        if not zipfile.is_zipfile(zip_path):
            raise ValueError("Downloaded file is not a valid zip archive.")
        # --- extract to a temp dir
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp_dir)
        _set_state(ds_id, progress=75)
        # --- import: find every image, pair it with its YOLO txt label
        from dataset_manager import IMG_ALL, LBL_ALL, ensure_dirs, get_classes, set_classes
        ensure_dirs()
        label_by_stem = {}
        for p in tmp_dir.rglob("*.txt"):
            label_by_stem.setdefault(p.stem, p)
        images = [p for p in tmp_dir.rglob("*")
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if not images:
            raise ValueError("Zip contained no images.")
        # Append COCO classes without changing the meaning of existing project labels.
        classes = get_classes()
        class_ids = {}
        for i, name in enumerate(classes):
            class_ids.setdefault(name.casefold(), i)
        for name in COCO80:
            if name not in class_ids:
                class_ids[name] = len(classes)
                classes.append(name)
        coco_to_project = [class_ids[name] for name in COCO80]
        set_classes(classes)
        n_img = n_lbl = 0
        for img in sorted(images):
            dest = IMG_ALL / img.name
            shutil.copy2(img, dest)
            n_img += 1
            lab = label_by_stem.get(img.stem)
            if lab is not None:
                labels = []
                for line in lab.read_text().splitlines():
                    if line.strip():
                        class_id, coords = line.split(maxsplit=1)
                        labels.append(f"{coco_to_project[int(class_id)]} {coords}")
                (LBL_ALL / (img.stem + ".txt")).write_text("\n".join(labels))
                n_lbl += 1
        _marker_path(ds_id).write_text(json.dumps(
            {"images": n_img, "labels": n_lbl, "source": url}))
        _set_state(ds_id, status="imported", progress=100,
                   imported_images=n_img, imported_labels=n_lbl, error=None)
    except Exception as e:
        _set_state(ds_id, status="failed", error=str(e))
        # clean up partial files so a retry starts clean
        try:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
        except Exception:
            pass
    finally:
        try:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
        except Exception:
            pass
