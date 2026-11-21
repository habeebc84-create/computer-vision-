"""Import a public ten-image Objects365 sample without redistributing its images."""
import json
from tempfile import TemporaryFile
from urllib.parse import urlsplit
from urllib.request import urlopen
from zipfile import ZipFile

from coco_import import import_coco
from dataset_manager import MAX_UPLOAD_BYTES

SAMPLE_ID = "objects365-sample"
SAMPLE_SOURCE = "https://huggingface.co/datasets/ohjoonhee/Objects365-10BBoxes-Filtered"
SAMPLE_ROWS = ("https://datasets-server.huggingface.co/rows?dataset=ohjoonhee%2F"
               "Objects365-10BBoxes-Filtered&config=default&split=train&offset=0&length=10")
SAMPLE_REVISION = "8c17f317010e44968cf28e7b4861b5ec488a0664"
# Objects365 V2 one-based category IDs used by this sample.
# Source: open-mmlab/mmdetection, mmdet/datasets/objects365.py.
CATEGORIES = {
    1: 'person',
    2: 'sneakers',
    3: 'chair',
    5: 'hat',
    6: 'car',
    7: 'lamp',
    8: 'glasses',
    9: 'bottle',
    11: 'cup',
    12: 'street lights',
    13: 'cabinet/shelf',
    14: 'handbag/satchel',
    15: 'bracelet',
    16: 'plate',
    18: 'helmet',
    19: 'book',
    20: 'gloves',
    22: 'boat',
    25: 'bench',
    32: 'microphone',
    35: 'suv',
    42: 'speaker',
    50: 'van',
    53: 'bakset',
    54: 'drum',
    56: 'bus',
    58: 'high heels',
    60: 'guitar',
    68: 'cymbal',
    69: 'lifesaver',
    73: 'sailboat',
    76: 'bed',
    84: 'air conditioner',
    86: 'hockey stick',
    95: 'clock',
    96: 'pot',
    110: 'machinery vehicle',
    134: 'refrigerator',
    141: 'jug',
    143: 'piano',
    163: 'chopsticks',
    200: 'heavy truck',
    209: 'mask',
    214: 'coffee machine',
    222: 'chicken',
    296: 'bear',
}
IMAGE_IDS = [420917, 900002, 900003, 900005, 85860, 711668, 689486, 900007, 900008, 900009]


def import_objects365_sample(progress):
    with urlopen(SAMPLE_ROWS, timeout=30) as response:
        rows = [entry["row"] for entry in json.loads(response.read(2 * 1024 * 1024))["rows"]]
    if [row["image_id"] for row in rows] != IMAGE_IDS:
        raise ValueError("The sample source changed. Import an official annotated ZIP instead.")

    document = {"images": [], "annotations": [], "categories": [
        {"id": category, "name": label} for category, label in CATEGORIES.items()
    ]}
    with TemporaryFile() as archive:
        with ZipFile(archive, "w") as bundle:
            for index, row in enumerate(rows, 1):
                image = row["image"]
                url = urlsplit(image["src"])
                if (url.scheme != "https" or url.netloc != "datasets-server.huggingface.co"
                        or SAMPLE_REVISION not in url.path.split("/")):
                    raise ValueError("The sample image URL did not match the expected source.")
                if (row["width"], row["height"]) != (image["width"], image["height"]):
                    raise ValueError("Sample image dimensions did not match its annotations.")
                with urlopen(image["src"], timeout=30) as response:
                    data = response.read(MAX_UPLOAD_BYTES + 1)
                if len(data) > MAX_UPLOAD_BYTES:
                    raise ValueError("A sample image exceeds the 20 MB import limit.")
                name = f"objects365-{row['image_id']}.jpg"
                bundle.writestr(name, data)
                document["images"].append({
                    "id": row["image_id"], "file_name": name, "width": row["width"], "height": row["height"],
                })
                document["annotations"].extend(
                    box for box in row["objects"] if not box["iscrowd"] and not box.get("isfake")
                )
                progress(int(index / len(rows) * 90))
            bundle.writestr("annotations.json", json.dumps(document))
        imported = import_coco(archive)
    if imported["skipped"]:
        raise ValueError("Some sample images could not be imported.")
    return len(imported["saved"])
