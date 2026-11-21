"""Import small-object photos and their reviewed bounding boxes."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryFile
from urllib.request import Request, urlopen
from zipfile import ZipFile

import cv2

from coco_import import import_coco
from dataset_manager import MAX_UPLOAD_BYTES
from image_processing import decode_image

SAMPLE_ID = "open-images-pen-phone"
SAMPLE_SOURCE = "https://storage.googleapis.com/openimages/web/download_v7.html"
SAMPLE_ROWS = json.loads((Path(__file__).parent / "data" / f"{SAMPLE_ID}.json").read_text(encoding="utf-8"))
CATEGORIES = {"pen": 1, "cell phone": 2}
ACCESSORIES_ID = "commons-combs-phone-cases"
ACCESSORIES_SOURCE = "https://commons.wikimedia.org/"
ACCESSORIES_ROWS = json.loads((Path(__file__).parent / "data" / f"{ACCESSORIES_ID}.json").read_text(encoding="utf-8"))


def _download_image(row):
    filename = f"{row['id']}.jpg"
    url = row.get("url") or f"https://open-images-dataset.s3.amazonaws.com/validation/{filename}"
    request = Request(url, headers={"User-Agent": "VisionDataset/1.0 (github.com/habeebc84-create/computer-vision-)"})
    with urlopen(request, timeout=30) as response:
        data = response.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"photo {filename} exceeds the 20 MB image limit.")
    image = decode_image(data)
    h, w = image.shape[:2]
    if max(w, h) > 768:
        scale = 768 / max(w, h)
        w, h = max(1, round(w * scale)), max(1, round(h * scale))
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError(f"Could not encode photo {filename}.")
    return encoded.tobytes(), w, h


def import_open_images_sample(progress):
    return _import_samples(SAMPLE_ROWS, CATEGORIES, "open-images", progress)


def import_accessories_sample(progress):
    return _import_samples(ACCESSORIES_ROWS, {"comb": 1, "phone case": 2}, "commons", progress)


def _import_samples(rows, categories, filename_prefix, progress):
    document = {"images": [], "annotations": [], "categories": [
        {"id": category, "name": label} for label, category in categories.items()
    ]}
    with TemporaryFile() as archive:
        with ZipFile(archive, "w") as bundle, ThreadPoolExecutor(max_workers=6) as pool:
            for index, (row, (data, w, h)) in enumerate(
                    zip(rows, pool.map(_download_image, rows)), 1):
                name = f"{filename_prefix}-{row['id']}.jpg"
                bundle.writestr(name, data)
                document["images"].append({"id": index, "width": w, "height": h, "file_name": name})
                for label, x1, y1, x2, y2 in row["boxes"]:
                    document["annotations"].append({
                        "image_id": index, "category_id": categories[label],
                        "bbox": [x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h],
                    })
                progress(index * 90 // len(rows))
            bundle.writestr("annotations.json", json.dumps(document))
        result = import_coco(archive)
    if result["skipped"]:
        raise ValueError("Some of the sample photos could not be imported.")
    return len(result["saved"])
