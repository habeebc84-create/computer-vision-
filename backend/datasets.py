"""Curated datasets with real images and bounding-box annotations.

One-click entries download from a verified official (or official-mirror) URL
and import into dataset/images/all + dataset/labels/all, converting each
dataset's own annotation format to YOLO txt labels. Every entry carries its
own class list — labels are remapped through THAT dataset's classes, never
assumed to be COCO.

Formats:
  sample    - the 10-image Objects365 sample (special-cased, see below)
  yolo      - zip of images + YOLO txt labels (COCO128 / COCO8)
  voc       - tar/zip of JPEGImages + Pascal VOC XML annotations
  coco      - images zip + a COCO-format JSON annotation zip (also LVIS v1,
              which is COCO-format but has no file_name — names come from
              coco_url instead)
  wider     - WIDER FACE images zip + wider_face_split.zip (native txt)
  visdrone  - VisDrone2019-DET zip (images + native txt annotations)
  external  - too large, or registration-gated, for one-click download.
              Listed with the official download page instead; start_download
              refuses these with a clear message.
"""
import json, math, shutil, tarfile, threading, urllib.request, xml.etree.ElementTree as ET, zipfile
from pathlib import Path, PurePosixPath
from coco_import import image_file_name as _image_file_name
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

# Pascal VOC 2012/2007 detection classes, in the order used by the
# official devkit (note: diningtable / pottedplant / tvmonitor, no spaces).
VOC20 = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
         "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
         "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# VisDrone2019-DET classes, in the order used by the official annotations
# (file values are 1-based, so the converter subtracts 1).
VISDRONE10 = ["pedestrian", "people", "bicycle", "car", "van", "truck",
              "tricycle", "awning-tricycle", "bus", "motor"]

# Small curated samples, each with its own importer (they download photos
# one by one and build a COCO-format bundle in memory — see the modules).
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
     "format": "sample",
     "url": open_images_sample.SAMPLE_SOURCE,
     "license": "Open Images annotations (CC BY 4.0) and per-image CC BY 2.0 attribution"},
    {"id": open_images_sample.ACCESSORIES_ID,
     "name": "Wikimedia Commons · Combs and phone cases",
     "description": "12 photos with hand-labeled boxes for combs and phone cases.",
     "images": len(open_images_sample.ACCESSORIES_ROWS), "classes": 2, "size_mb": 2,
     "format": "sample",
     "url": open_images_sample.ACCESSORIES_SOURCE,
     "license": "Per-image Creative Commons attribution in the repository manifest"},
    {"id": SAMPLE_ID,
     "name": "Objects365 sample",
     "description": "10 real Objects365 photos with bounding boxes. Use this small sample to check import, annotation, and training.",
     "images": len(IMAGE_IDS), "classes": len(CATEGORIES), "size_mb": 5.2,
     "format": "sample",
     "url": SAMPLE_SOURCE,
     "license": "Objects365 terms and individual image licenses"},
    {"id": "coco128",
     "name": "COCO128",
     "description": "128 images from COCO train 2017 with real YOLO labels across the 80 COCO classes. Recommended for real training — big enough to learn from, small enough to train quickly on CPU.",
     "images": 128, "classes": 80, "size_mb": 6.7,
     "format": "yolo", "class_names": COCO80,
     "url": "https://ultralytics.com/assets/coco128.zip",
     "license": "COCO 2017, CC BY 4.0"},
    {"id": "coco8",
     "name": "COCO8",
     "description": "Just 8 images (4 train / 4 val) from COCO train 2017. A tiny smoke test — use it to check the whole download → split → train pipeline works before training on COCO128.",
     "images": 8, "classes": 80, "size_mb": 0.4,
     "format": "yolo", "class_names": COCO80,
     "url": "https://ultralytics.com/assets/coco8.zip",
     "license": "COCO 2017, CC BY 4.0"},
    {"id": "coco2017-val",
     "name": "COCO 2017 validation",
     "description": "The full COCO 2017 validation split: 5,000 real photos with COCO JSON annotations across the 80 classes. Two downloads (images + annotations) that the importer merges for you.",
     "images": 5000, "classes": 80, "size_mb": 1019,
     "format": "coco", "annotations_file": "instances_val2017.json",
     "url": "http://images.cocodataset.org/zips/val2017.zip",
     "annotations_url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
     "license": "COCO 2017, CC BY 4.0"},
    {"id": "voc2012",
     "name": "Pascal VOC 2012 trainval",
     "description": "11,530 images with Pascal VOC XML annotations across 20 everyday classes (person, car, dog, chair, …). The classic detection benchmark — a large download, so only start it on a fast connection.",
     "images": 11530, "classes": 20, "size_mb": 1907,
     "format": "voc", "class_names": VOC20,
     "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
     "license": "Free for research use (images © their Flickr owners)"},
    {"id": "voc2007",
     "name": "Pascal VOC 2007 trainval",
     "description": "5,011 images with Pascal VOC XML annotations across the same 20 classes as VOC 2012. The smaller VOC set — a good first VOC download.",
     "images": 5011, "classes": 20, "size_mb": 439,
     "format": "voc", "class_names": VOC20,
     "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
     "license": "Free for research use (images © their Flickr owners)"},
    {"id": "widerface-train",
     "name": "WIDER FACE train",
     "description": "12,880 photos with 159,424 labelled faces — the standard face-detection training set. One class (face). Two downloads (images + annotation file) merged by the importer.",
     "images": 12880, "classes": 1, "size_mb": 1401,
     "format": "wider", "class_names": ["face"],
     "annotations_file": "wider_face_train_bbx_gt.txt",
     "url": "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_train.zip",
     "annotations_url": "https://huggingface.co/datasets/wider_face/resolve/main/data/wider_face_split.zip",
     "license": "Non-commercial research use (mirror of the official WIDER FACE data)"},
    {"id": "visdrone-train",
     "name": "VisDrone2019-DET train",
     "description": "6,471 drone photos with over 540,000 boxes across 10 classes (pedestrian, car, van, bus, …). Aerial views — good for small-object detection. Ignored regions in the annotations are skipped.",
     "images": 6471, "classes": 10, "size_mb": 1475,
     "format": "visdrone", "class_names": VISDRONE10,
     "url": "https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip",
     "license": "Non-commercial research use (Tianjin University)"},
    {"id": "visdrone-val",
     "name": "VisDrone2019-DET val",
     "description": "548 drone photos from the VisDrone2019-DET validation split, same 10 classes as the train set. Small enough to try VisDrone quickly.",
     "images": 548, "classes": 10, "size_mb": 72,
     "format": "visdrone", "class_names": VISDRONE10,
     "url": "https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip",
     "license": "Non-commercial research use (Tianjin University)"},
    # --- too large or registration-gated for one-click download ---
    {"id": "coco2017-train",
     "name": "COCO 2017 train",
     "description": "The full COCO 2017 training split: 118,287 images across the 80 classes. 18 GB — too large for a one-click download, so get it from the official site instead.",
     "images": 118287, "classes": 80, "size_mb": 18432,
     "format": "external",
     "url": "http://images.cocodataset.org/zips/train2017.zip",
     "external_url": "http://images.cocodataset.org/zips/train2017.zip",
     "note": "18 GB download — too large for one-click. Download the images zip and the annotations zip, then use Dataset → Import annotated ZIP.",
     "license": "COCO 2017, CC BY 4.0"},
    {"id": "lvis-v1-train",
     "name": "LVIS v1 train",
     "description": "100,170 images with 1,203 object categories — the large-vocabulary long-tail benchmark. Annotations are COCO-format; the images are COCO train2017 photos.",
     "images": 100170, "classes": 1203, "size_mb": 18766,
     "format": "external",
     "url": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip",
     "external_url": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip",
     "note": "Needs the 18 GB COCO train2017 images as well as the 334 MB annotation file. Download both, then use Dataset → Import annotated ZIP (LVIS JSON is COCO-format).",
     "license": "LVIS v1, CC BY 4.0"},
    {"id": "lvis-v1-val",
     "name": "LVIS v1 val",
     "description": "19,809 images with 1,203 object categories. The images span COCO train2017 (15,000) and val2017 (4,809), so a one-click download would need both COCO archives (~19 GB).",
     "images": 19809, "classes": 1203, "size_mb": 19029,
     "format": "external",
     "url": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip",
     "external_url": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip",
     "note": "Needs COCO train2017 + val2017 images (~19 GB) as well as the 61 MB annotation file. Download them, then use Dataset → Import annotated ZIP.",
     "license": "LVIS v1, CC BY 4.0"},
    {"id": "open-images-v7",
     "name": "Open Images V7",
     "description": "Google's 9-million-image dataset; the detection subset is 1,743,042 training images across 600 boxable classes. 561 GB in total — get a subset from the official download page instead.",
     "images": 1743042, "classes": 600, "size_mb": 574464,
     "format": "external",
     "url": "https://storage.googleapis.com/openimages/web/download_v7.html",
     "external_url": "https://storage.googleapis.com/openimages/web/download_v7.html",
     "note": "561 GB for the full detection subset. Use the official download page (or the FiftyOne zoo dataset) to fetch only the classes or split you need.",
     "license": "CC BY 2.0 (annotations); images under their own licences"},
    {"id": "bdd100k",
     "name": "BDD100K",
     "description": "100,000 dashcam images for driving-scene detection (79,863 labelled) across 10 classes: person, rider, car, truck, bus, train, motor, bike, traffic light, traffic sign.",
     "images": 100000, "classes": 10, "size_mb": 7270,
     "format": "external",
     "url": "https://bdd-data.berkeley.edu/",
     "external_url": "https://bdd-data.berkeley.edu/",
     "note": "Registration is required on the official Berkeley site before downloading. The site could not be reached from this environment when checked (21 Nov 2026).",
     "license": "Non-commercial / research use (BDD100K licence)"},
    {"id": "sku-110k",
     "name": "SKU-110K",
     "description": "11,762 densely packed retail-shelf photos with over 1.7 million boxes — about 147 products per image — all under a single class, “object”. The stress test for crowded small-object detection.",
     "images": 11762, "classes": 1, "size_mb": 13926,
     "format": "external",
     "url": "https://github.com/eg4000/SKU110K_CVPR19",
     "external_url": "https://github.com/eg4000/SKU110K_CVPR19",
     "note": "About 13.6 GB via the Ultralytics mirror; the original release is academic / non-commercial use. Annotations are CSV, not YOLO txt.",
     "license": "Academic / non-commercial use (Trax Retail)"},
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
    if d.get("format") == "external":
        raise ValueError(
            f"{d['name']} cannot be downloaded with one click — "
            f"get it from the official site: {d.get('external_url') or d['url']}")
    cur = _load_state(ds_id)
    if cur["status"] == "downloading":
        return status(ds_id)
    if cur["status"] == "imported" and (not force or ds_id in SAMPLE_IMPORTERS):
        return status(ds_id)
    with _lock:
        _set_state(ds_id, status="downloading", progress=0, error=None)
        t = threading.Thread(target=_run_import, args=(d,), daemon=True)
        t.start()
    return status(ds_id)

def _run_import(dataset):
    # Serialise imports so two downloads never write the same images/labels.
    with _import_lock:
        _download_and_import(dataset)

# ---------------------------------------------------------------------------
# Annotation converters (pure functions — unit-tested, no disk or network).
# Every converter returns YOLO txt lines: "class cx cy w h" (normalised).
# ---------------------------------------------------------------------------

def _yolo_line(class_id, x1, y1, x2, y2, width, height):
    """One pixel box -> a YOLO txt line, clipped to the image.
    Returns None for a degenerate box (zero/negative area after clipping)."""
    if width <= 0 or height <= 0:
        return None
    x1 = max(0.0, min(float(x1), width)); x2 = max(0.0, min(float(x2), width))
    y1 = max(0.0, min(float(y1), height)); y2 = max(0.0, min(float(y2), height))
    if x2 <= x1 or y2 <= y1:
        return None
    cx, cy = ((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height
    w, h = (x2 - x1) / width, (y2 - y1) / height
    return f"{int(class_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

def voc_xml_to_yolo(xml_text, class_names):
    """Parse one Pascal VOC Annotation XML -> YOLO lines.
    Skips objects marked difficult and names outside class_names."""
    root = ET.fromstring(xml_text)
    size = root.find("size")
    width = float(size.findtext("width")); height = float(size.findtext("height"))
    lines = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in class_names:
            continue
        if (obj.findtext("difficult") or "0").strip() == "1":
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        line = _yolo_line(class_names.index(name),
                          box.findtext("xmin"), box.findtext("ymin"),
                          box.findtext("xmax"), box.findtext("ymax"),
                          width, height)
        if line:
            lines.append(line)
    return lines

def coco_json_to_yolo(doc):
    """COCO-format annotation doc -> (class_names, {image file name: [YOLO lines]}).
    Works for COCO 2017 and for LVIS v1, which is COCO-format. Skips crowd
    boxes and degenerate boxes; class order follows sorted category id."""
    images = {img["id"]: img for img in doc.get("images", [])}
    categories = {c["id"]: (c.get("name") or "").strip()
                  for c in doc.get("categories", [])}
    categories = {cid: name for cid, name in categories.items() if name}
    if not images or not categories:
        raise ValueError("The annotation file has no images or named categories.")
    ordered = sorted(categories)
    class_names = [categories[cid] for cid in ordered]
    index = {cid: i for i, cid in enumerate(ordered)}
    labels = {PurePosixPath(_image_file_name(img)).name: []
              for img in images.values()}
    for ann in doc.get("annotations", []):
        img = images.get(ann.get("image_id"))
        if img is None or ann.get("iscrowd"):
            continue
        if ann.get("category_id") not in index:
            continue
        width = float(img.get("width") or 0); height = float(img.get("height") or 0)
        x, y, w, h = map(float, ann["bbox"])
        line = _yolo_line(index[ann["category_id"]], x, y, x + w, y + h,
                          width, height)
        if line:
            labels[PurePosixPath(_image_file_name(img)).name].append(line)
    return class_names, labels

def wider_txt_to_yolo(text):
    """Parse a WIDER FACE *_bbx_gt.txt file -> {relative image path: [(x1, y1, x2, y2), ...]}.
    Boxes are in pixels; the image size is not in the file, so the importer
    reads it from the image itself. Faces flagged invalid (8th field == 1)
    are skipped."""
    entries = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        path = lines[i].strip(); i += 1
        if not path:
            continue
        if i >= len(lines):
            raise ValueError("Truncated WIDER FACE annotation file (missing face count).")
        try:
            count = int(lines[i].strip())
        except ValueError:
            raise ValueError("Malformed WIDER FACE annotation file (face count is not a number).")
        i += 1
        boxes = []
        for _ in range(count):
            if i >= len(lines):
                raise ValueError("Truncated WIDER FACE annotation file (missing face box).")
            parts = lines[i].split(); i += 1
            if len(parts) < 4:
                continue
            if len(parts) >= 8 and parts[7] == "1":
                continue  # invalid face
            x1, y1, w, h = map(int, parts[:4])
            if w > 0 and h > 0:
                boxes.append((x1, y1, x1 + w, y1 + h))
        entries[path] = boxes
    return entries

def visdrone_txt_to_yolo(text, width, height, class_names):
    """Parse one VisDrone annotation txt -> YOLO lines. Each line is
    x,y,w,h,score,class,truncation,occlusion. Rows with score != 0 are
    ignored regions; class is 1-based, so subtract 1 (Ultralytics
    visdrone2yolo does the same)."""
    lines = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        row = raw.split(",")
        if len(row) < 6:
            continue
        if row[4].strip() != "0":
            continue  # ignored region
        try:
            cls = int(row[5]) - 1
            x, y, w, h = map(int, row[:4])
        except ValueError:
            continue
        if not 0 <= cls < len(class_names):
            continue
        line = _yolo_line(cls, x, y, x + w, y + h, width, height)
        if line:
            lines.append(line)
    return lines

# ---------------------------------------------------------------------------
# Download + import
# ---------------------------------------------------------------------------

def _head_length(url):
    """Content-Length from a HEAD request, or 0 if the server won't say."""
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return int(resp.headers.get("Content-Length") or 0)
    except Exception:
        return 0

def _download(url, dest, on_fraction):
    """Stream one file to dest, calling on_fraction(0..1) as bytes arrive."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    on_fraction(done / total)
    return dest

def _extract(archive, tmp_dir):
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(tmp_dir)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as t:
            # filter="data" blocks absolute paths and symlinks in the archive.
            t.extractall(tmp_dir, filter="data")
    else:
        raise ValueError("Downloaded file is not a zip or tar archive.")

def _remap_classes(class_names):
    """Append a dataset's class names to the project classes (keeping
    existing indices) and return the dataset id -> project id map.
    Names are matched case-insensitively, so a dataset's "Person" never
    creates a second class next to an existing "person"."""
    from dataset_manager import get_classes, set_classes
    classes = get_classes()
    by_fold = {name.casefold(): i for i, name in enumerate(classes)}
    mapping = []
    for name in class_names:
        key = name.casefold()
        if key in by_fold:
            mapping.append(by_fold[key])
        else:
            by_fold[key] = len(classes)
            classes.append(name)
            mapping.append(len(classes) - 1)
    set_classes(classes)
    return mapping

def _image_dims(path):
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path.name}")
    h, w = img.shape[:2]
    return w, h

def _match_image(label_stem, by_stem):
    """Find an extracted image for a label stem. Exact match first; then
    strip a COCO_ split prefix (LVIS names images COCO_val2017_000000000139);
    then any image whose stem ends with the label stem."""
    if label_stem in by_stem:
        return by_stem[label_stem]
    for prefix in ("COCO_train2017_", "COCO_val2017_"):
        if label_stem.startswith(prefix):
            stripped = label_stem[len(prefix):]
            if stripped in by_stem:
                return by_stem[stripped]
    for stem, path in by_stem.items():
        if label_stem.endswith("_" + stem) or stem.endswith("_" + label_stem):
            return path
    return None

def _download_and_import(d):
    ds_id, url = d["id"], d["url"]
    tmp_dir = DL_DIR / f"{ds_id}_extract"
    try:
        if ds_id in SAMPLE_IMPORTERS:
            count = SAMPLE_IMPORTERS[ds_id](lambda progress: _set_state(ds_id, progress=progress))
            _marker_path(ds_id).write_text(json.dumps({"images": count, "labels": count, "source": url}))
            _set_state(ds_id, status="imported", progress=100, imported_images=count, imported_labels=count, error=None)
            return
        if d.get("format") == "external":
            raise ValueError("This dataset cannot be downloaded with one click — "
                             f"get it from the official site: {d.get('external_url') or url}")
        # --- download the images archive and, where needed, the annotations
        # archive. Progress 0-70% is shared across files, weighted by their
        # expected sizes when the servers report them, else split evenly.
        files = [(url, DL_DIR / f"{ds_id}.archive")]
        if d.get("annotations_url"):
            files.append((d["annotations_url"], DL_DIR / f"{ds_id}.annotations"))
        sizes = [_head_length(u) for u, _ in files]
        if sizes and all(s > 0 for s in sizes):
            total_expected = sum(sizes)
            done_bytes = 0
            for (dl_url, dest), expected in zip(files, sizes):
                start = done_bytes
                def on_fraction(frac, _start=start, _expected=expected):
                    _set_state(ds_id, progress=round((_start + frac * _expected) / total_expected * 70, 1))
                _download(dl_url, dest, on_fraction)
                done_bytes += expected
                _set_state(ds_id, progress=round(done_bytes / total_expected * 70, 1))
        else:
            for i, (dl_url, dest) in enumerate(files):
                def on_fraction(frac, _i=i):
                    _set_state(ds_id, progress=round((_i + frac) / len(files) * 70, 1))
                _download(dl_url, dest, on_fraction)
                _set_state(ds_id, progress=round((i + 1) / len(files) * 70, 1))
        # --- extract everything into one temp dir
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        for _, archive in files:
            _extract(archive, tmp_dir)
        _set_state(ds_id, progress=75)
        # --- import, by format
        from dataset_manager import IMG_ALL, LBL_ALL, ensure_dirs
        ensure_dirs()
        images = [p for p in tmp_dir.rglob("*")
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if not images:
            raise ValueError("Download contained no images.")
        by_stem = {}
        for p in images:
            by_stem.setdefault(p.stem, p)
        fmt = d.get("format")
        n_img = n_lbl = 0
        if fmt == "yolo":
            label_by_stem = {}
            for p in tmp_dir.rglob("*.txt"):
                label_by_stem.setdefault(p.stem, p)
            mapping = _remap_classes(d["class_names"])
            for img in sorted(images):
                shutil.copy2(img, IMG_ALL / img.name)
                n_img += 1
                lab = label_by_stem.get(img.stem)
                if lab is not None:
                    labels = []
                    for line in lab.read_text().splitlines():
                        if line.strip():
                            class_id, coords = line.split(maxsplit=1)
                            labels.append(f"{mapping[int(class_id)]} {coords}")
                    (LBL_ALL / (img.stem + ".txt")).write_text("\n".join(labels))
                    n_lbl += 1
        elif fmt == "voc":
            mapping = _remap_classes(d["class_names"])
            xml_by_stem = {}
            for p in tmp_dir.rglob("*.xml"):
                xml_by_stem.setdefault(p.stem, p)
            for img in sorted(images):
                shutil.copy2(img, IMG_ALL / img.name)
                n_img += 1
                xml = xml_by_stem.get(img.stem)
                if xml is not None:
                    raw = voc_xml_to_yolo(xml.read_text(), d["class_names"])
                    labels = [f"{mapping[int(line.split(maxsplit=1)[0])]} {line.split(maxsplit=1)[1]}"
                              for line in raw]
                    (LBL_ALL / (img.stem + ".txt")).write_text("\n".join(labels))
                    n_lbl += 1
        elif fmt == "coco":
            ann_name = d.get("annotations_file")
            json_path = None
            for p in tmp_dir.rglob("*.json"):
                if ann_name and p.name == ann_name:
                    json_path = p
                    break
            if json_path is None:
                raise ValueError(f"Annotations file {ann_name or '*.json'} not found in the download.")
            doc = json.loads(json_path.read_text())
            class_names, labels = coco_json_to_yolo(doc)
            mapping = _remap_classes(class_names)
            for label_name, raw_lines in sorted(labels.items()):
                img = _match_image(PurePosixPath(label_name).stem, by_stem)
                if img is None:
                    continue
                shutil.copy2(img, IMG_ALL / img.name)
                n_img += 1
                if raw_lines:
                    remapped = [f"{mapping[int(line.split(maxsplit=1)[0])]} {line.split(maxsplit=1)[1]}"
                                for line in raw_lines]
                    (LBL_ALL / (img.stem + ".txt")).write_text("\n".join(remapped))
                    n_lbl += 1
        elif fmt == "wider":
            ann_name = d.get("annotations_file")
            txt_path = None
            for p in tmp_dir.rglob("*.txt"):
                if ann_name and p.name == ann_name:
                    txt_path = p
                    break
            if txt_path is None:
                raise ValueError(f"Annotations file {ann_name or '*.txt'} not found in the download.")
            mapping = _remap_classes(d["class_names"])
            entries = wider_txt_to_yolo(txt_path.read_text())
            for rel_path, boxes in sorted(entries.items()):
                stem = PurePosixPath(rel_path).stem
                img = by_stem.get(stem)
                if img is None:
                    continue
                shutil.copy2(img, IMG_ALL / img.name)
                n_img += 1
                if boxes:
                    w, h = _image_dims(img)
                    lines = []
                    for (x1, y1, x2, y2) in boxes:
                        line = _yolo_line(mapping[0], x1, y1, x2, y2, w, h)
                        if line:
                            lines.append(line)
                    if lines:
                        (LBL_ALL / (img.stem + ".txt")).write_text("\n".join(lines))
                        n_lbl += 1
        elif fmt == "visdrone":
            mapping = _remap_classes(d["class_names"])
            ann_by_stem = {}
            for p in tmp_dir.rglob("*.txt"):
                if p.parent.name == "annotations":
                    ann_by_stem.setdefault(p.stem, p)
            for img in sorted(images):
                shutil.copy2(img, IMG_ALL / img.name)
                n_img += 1
                ann = ann_by_stem.get(img.stem)
                if ann is not None:
                    w, h = _image_dims(img)
                    raw = visdrone_txt_to_yolo(ann.read_text(), w, h, d["class_names"])
                    labels = [f"{mapping[int(line.split(maxsplit=1)[0])]} {line.split(maxsplit=1)[1]}"
                              for line in raw]
                    if labels:
                        (LBL_ALL / (img.stem + ".txt")).write_text("\n".join(labels))
                        n_lbl += 1
        else:
            raise ValueError(f"Unknown dataset format: {fmt!r}")
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
