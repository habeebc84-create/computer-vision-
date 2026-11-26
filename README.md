# Real-Time Object Detection System — YOLOv8 & Deep Learning

**ITA0523 — Computer Vision capstone project**

A full-stack, production-style web application that detects objects in real time
from a webcam and in uploaded images — using real YOLOv8
inference (Ultralytics, PyTorch, OpenCV). No fake detections, no simulated
results: every bounding box, label and confidence score comes from the model.

| Image mode | Objects it can find | Where it runs |
| --- | --- | --- |
| Everyday objects (365) | YOLOE with the Objects365 vocabulary, including categories such as glasses and sneakers | Browser, with weights bundled in the frontend |
| Expanded objects (4,585) | A larger prompt-free YOLOE vocabulary | Browser, with a 125 MB first download |
| Packages (trained) | Parcels, using a YOLOv8n detector fine-tuned on factory package photos | Browser, with weights bundled in the frontend |
| COCO / selected backend model | The original 80 COCO categories or the labels learned by your selected checkpoint | Python backend, with a browser fallback for COCO |
| Search by name | Object names you enter as prompts for YOLO-World | Python backend or Grounding DINO in the browser |

Everyday objects is the default on a fresh install. The first frontend build or development launch downloads and verifies the 38 MB Objects365 weights. For a category outside its vocabulary, try Search by name. Small, occluded or unfamiliar objects can still be missed. See the [everyday model details](frontend/public/models/objects365.md) and [package training and held-out results](models/package-model.md).

The expanded option uses the [YOLOE-11l prompt-free ONNX export and matching label list](https://huggingface.co/talmago/yoloe.onnx/tree/4ea526e5f1962d756e5cc7e13360ba5008a0d672) at a pinned revision. See the [original YOLOE project](https://github.com/THU-MIG/yoloe) and [AGPL-3.0 license](frontend/public/models/AGPL-3.0.txt). Choose it in Image Detection for more category coverage. The 365-category model remains the default for faster processing. Larger vocabularies can still miss or mislabel objects.

## Modules (exactly as specified)

1. **Module 1 — Dataset Preparation** — upload images, annotate with a
   LabelImg-compatible YOLO-txt editor, define classes, split train/val,
   generate `data.yaml`, export the dataset as a zip.
2. **Module 2 — Model Training** — configure a pretrained YOLOv8 variant
   (n/s/m), epochs, image size, batch; start/stop real Ultralytics training in
   a background thread; live epoch / loss / precision / recall / mAP50 /
   mAP50-95 from `results.csv`; `best.pt` saved, downloadable and loadable.
3. **Module 3 — Inference & Visualization** — live webcam (browser camera,
   frames sent to the backend) and image detection with annotated output.
   Uploaded videos can be sampled in the browser using the 365-category detector,
   or processed frame by frame on the backend and downloaded as annotated MP4.

Plus: Overview Dashboard, Detection History (browser storage and SQLite), Analytics & Reports
(CSV + PDF), Settings (model, conf/IoU thresholds, image size, device).

## Architecture

```
object-detection-system/
├── run.py                      # single entry point -> uvicorn :8000
├── requirements.txt
├── frontend/                   # React + Vite + TypeScript + Tailwind + Recharts + lucide-react
│   └── src/{pages,components,lib}
├── backend/
│   ├── app.py                  # FastAPI app, /api/* routes, serves built frontend
│   ├── detection.py            # ModelManager — real YOLOv8 inference + annotation
│   ├── image_processing.py     # decode/encode helpers (OpenCV)
│   ├── video_processing.py     # background frame-by-frame video jobs
│   ├── webcam.py               # webcam contract (browser getUserMedia -> /api/detect/frame)
│   ├── model_training.py       # background Ultralytics training with real callbacks
│   ├── dataset_manager.py      # uploads, YOLO-txt annotations, split, data.yaml, export
│   ├── analytics.py            # SQLite history + analytics (real records only)
│   └── report_generator.py     # CSV + PDF (reportlab) reports
├── models/                     # yolov8n.pt auto-downloaded on first load; best.pt after training
├── dataset/images/{train,val}  # dataset/labels/{train,val}, dataset/data.yaml
├── outputs/{images,videos,reports}
└── database/detections.db      # SQLite detection history
```

The backend serves the built frontend statically, so one process (`run.py`)
serves the whole app at `http://localhost:8000`.

## Windows — install & launch

Prerequisites: **Python 3.10+** and **Node.js 18+** installed and on PATH.
(On Windows use PowerShell; `py` works in place of `python` if needed.)

```powershell
cd object-detection-system

# 1. Backend (CPU PyTorch — no GPU needed)
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

# 2. Frontend (only needed once, or after changing frontend code)
cd frontend
npm install
npm run build
cd ..

# 3. Launch (from the project root, venv active)
python run.py
```

Open **http://localhost:8000**. The first inference downloads `yolov8n.pt`
(~6 MB, COCO 80 classes) from Ultralytics into `models/` — internet is
required once for that download. The `.pt` file is git-ignored on purpose.

## Browser-only deployment on Vercel

Image detection, live webcam detection, and sampled video detection run in the
browser on the Vercel site. Detection History shows image runs, completed video
sessions, and saved webcam snapshots from this browser. Records saved in this
browser stay in its site storage. Connect a backend for training, dataset
management, full-frame video export, and server reports.

To connect just this browser, open **Settings → Backend connection**, enter your Python server URL, and choose **Connect backend**. The app verifies its health endpoint before saving the address on this device. **Use deployment default** clears that override.

To configure a backend for everyone on the Vercel deployment:

1. Host the backend on a persistent Python host — Render, Railway or Fly.io
   (a web service running `python run.py`, with the repo's `requirements.txt`;
   free tiers work for demos but are slow to wake and CPU-only).
2. In `frontend/`, set the backend URL before building:
   ```bash
   # frontend/.env  (see frontend/.env.example)
   VITE_API_URL=https://your-backend-host.example.com
   ```
   Or set `VITE_API_URL` as a Vercel environment variable for the project.
3. Rebuild and redeploy the frontend (`npm run build`, then deploy) — Vite
   bakes `VITE_API_URL` into the build at build time, so changing it requires
   a fresh build. All API calls and all output/download links
   (`/outputs/...`, `/dataset/...`) are then prefixed with that URL.
4. Keep the backend URL reachable over HTTPS (a Vercel page served over
   HTTPS cannot call an `http://` backend — the browser blocks mixed content).

Running locally (`python run.py`) needs none of this: the backend serves the
frontend itself on the same origin, and `VITE_API_URL` stays empty.

## In-browser detection (no backend)

**Update: image and live-webcam detection now work even when the backend is
offline.** When the frontend cannot reach the backend or the detection API
is unavailable, the Image and Live pages automatically fall back to running real YOLOv8 inference inside the
browser — the page shows a violet chip, "Running in your browser", so a browser result is never passed off as a backend result.

How it works:

- The browser downloads `yolov8n.onnx` (~12 MB, once only) from Habeeb's own
  Hugging Face model repo
  (`https://huggingface.co/habeebc12/yolov8n-onnx/resolve/main/yolov8n.onnx`)
  and caches it in Cache Storage. The `.onnx` file is **not** committed to
  the repo. (First version fetched it from Ultralytics' GitHub release
  download — that host sends no CORS headers, so browsers blocked the
  download with "Failed to fetch"; Hugging Face serves it with
  `Access-Control-Allow-Origin: *`, which is why the model lives there.)
- Inference runs with ONNX Runtime Web (`onnxruntime-web` 1.20.1, single-
  threaded wasm; its wasm files load from the jsDelivr CDN), using the same
  preprocessing and NMS as the backend: 640×640 letterbox (grey 114 padding,
  like Ultralytics), NCHW float32, YOLOv8 `84×8400` output parsing,
  confidence threshold + class-aware NMS, boxes rescaled to the original
  image. The confidence/IoU thresholds come from Settings (saved on the
  device in `localStorage`, so they work offline too).
- The ONNX code is lazy-loaded — the runtime only downloads when the backend
  is actually unreachable, so the normal (backend) path costs nothing extra.

### Open-vocabulary in the browser (Grounding DINO)

**“Any object — type its name” also works with no backend.** When the backend
is unreachable and the user picks open-vocab mode, the Image and Live pages
run real Grounding DINO inference in the browser instead of erroring:

- Model: `onnx-community/grounding-dino-tiny-ONNX` via
  `@huggingface/transformers` 4.3.2 (`zero-shot-object-detection` pipeline,
  `dtype: 'q8'`), loaded lazily only when open-vocab mode is used offline.
  The q8 weights (`model_quantized.onnx`, ~204 MB) + tokenizer (~1 MB) download
  once from Hugging Face (CORS-enabled) and are cached by Transformers.js in
  Cache Storage — about **~205 MB total, once only**. Nothing is committed to
  the repo. (The `Xenova/grounding-dino-tiny` repo named in older docs is no
  longer reachable — 401 — so the maintained `onnx-community` build is used.)
- Results use the same shape as the COCO path (detections with
  `class_name` = the matched prompt text, confidence, bbox in original-image
  pixels, annotated image, `source: 'browser'`, engine
  `In-browser (Grounding DINO)`). Pipeline boxes are already in original
  pixels; the code only clamps them into the image, and drops near-duplicate
  boxes for the same prompt (IoU > 0.5).
- The confidence threshold comes from Settings (saved in `localStorage`),
  default 0.25. Type specific phrases (“red bus” beats “thing”); up to 20
  prompts at a time. Preset chips on the Image/Live pages try objects that
  are **not** in the 80 COCO classes (microphone, sunglasses, headphones,
  traffic cone, whiteboard, coffee mug).

Limits — stated honestly:

- **Slow.** Grounding DINO tiny is a ~172M-parameter vision-language model,
  and the browser checks one object at a time: expect several seconds per
  prompt on a typical laptop CPU (9–20 s per prompt measured in Node on a
  throttled shared CPU; faster on a real laptop, still seconds). Photos are
  fine; live video is snapshots, not smooth video. The COCO YOLOv8 path stays
  the fast default for the 80 classes.
- **Experimental accuracy.** It can confuse lookalike objects and invents
  weak boxes for things that are not there if the threshold is set low; the
  Node verification below checks that a prompt with no match (“elephant” on
  the bus photo) is not invented.
- **Big first download.** ~205 MB once, versus ~12 MB for the COCO model.
  After that it is cached and loads offline.

Verified in Node (same library, same model, `dtype: 'q8'`, threshold 0.25,
Ultralytics bus photo 810×1080, one detector call per prompt):
`bus.` → bus 90.2%; `person.` → 4 people (83.9 / 80.1 / 78.9 / 59.8%);
`elephant.` → 0 detections — nothing invented. Total 5 detections in 32 s
for 3 prompts on a throttled shared CPU (9–12 s per prompt; faster on a
real laptop). Boxes are in original-image pixels (one was `xmin: -1`, which
the browser code clamps). Two things the test caught and the code handles:
prompts must be passed one call at a time (an array errors — the exported
text encoder has a fixed batch size of 1), and each phrase needs its trailing
period (without it the text encoder errors). A single joined query
(`"bus. person. elephant."`) was also tried and rejected: it found only the
bus and missed all 4 people, in both q8 and fp32.

Limits — stated honestly (COCO path):

- **COCO 80 classes only** for the fast path. Open-vocabulary now also works
  in the browser (see above) — the old “stays backend-only” note no longer applies.
- **Slower than the backend.** Measured ~180 ms/frame with native CPU
  inference in Node; in-browser wasm is slower (~2 fps effective for live
  video on a typical laptop). Photos are fine; live video is snapshots, not
  smooth video, until run on a faster machine.
- **Training needs the backend.** Browser detections and saved snapshots are
  available in Detection History on the device that ran them.
- **Tiny edge objects can differ from the backend.** Verified on the
  Ultralytics bus photo: the browser found the bus (84%) and all 4 people
  (88/88/87/43%), matching the backend's 6 detections except the tiny
  stop sign at the image edge (backend 26%, browser max score 4%) — a
  marginal object whose score depends on resize details.

## Objects365 training data

The [Objects365 overview](https://www.objects365.org/overview.html) lists
about 2 million images, 365 object categories, and 30 million bounding boxes.
The dataset covers everyday objects across varied scenes and is suited to
pretraining before fine-tuning on retail or custom objects. The
[Objects365 paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Shao_Objects365_A_Large-Scale_High-Quality_Dataset_for_Object_Detection_ICCV_2019_paper.html)
reports improved COCO detection scores over ImageNet pretraining in its experiments.

The browser detector already includes the Objects365 vocabulary. To train,
get the data from the [official download page](https://www.objects365.org/download.html).
In **Dataset → Objects365**, import a ZIP containing one COCO-format
annotation JSON and its matching images with the Python backend connected.
Use subsets up to 100 MB per archive and keep image basenames unique within
each ZIP. The importer converts boxes to YOLO labels and adds category names
while preserving existing label IDs. Split the imported data before training.
For conversion outside the app, see the
[Ultralytics COCO-to-YOLO reference](https://docs.ultralytics.com/reference/data/converter/#ultralytics.data.converter.convert_coco).

## Built-in datasets (one-click download)

**Objects365 sample** downloads ten real annotated images from the
[public Objects365 sample mirror](https://huggingface.co/datasets/ohjoonhee/Objects365-10BBoxes-Filtered).
Open Dataset and choose **Download & import** with the backend connected.
The importer converts boxes to YOLO format and keeps existing class IDs.
Use this sample to test the workflow. Use the official release for larger training runs.

Module 1 — Dataset has eleven one-click datasets. Click **Download &
import** and the backend downloads, verifies and imports them into
`dataset/images/all/` + `dataset/labels/all/`, with class names set to
that dataset's own classes. Then Split the dataset and train in Module 2
as usual. Every URL below was verified live (HTTP 200) on 21 Nov 2026.
Downloads run on the Python backend — they only work with the backend
running locally, not on the hosted site.

To fine-tune for small everyday objects, import **Open Images · Pens and
phones** (60 photos with original pen and cell-phone boxes from [Open
Images](https://storage.googleapis.com/openimages/web/download_v7.html)) and
**Wikimedia Commons · Combs and phone cases** (12 photos with hand-labeled boxes).
The repository records the [Open Images photo credits](backend/data/open-images-pen-phone.json)
and [Wikimedia Commons photo credits](backend/data/commons-combs-phone-cases.json).
The samples are a starting point. For accuracy on your camera, add labeled photos at different
distances, angles, and lighting, plus confirmed backgrounds.

Save boxes before splitting. A saved empty annotation marks an intentional background.
The split excludes unreviewed photos and exact duplicate files, and keeps examples of every
labeled class in training. It reports classes with no validation coverage so you know which
objects the metrics cover. Split again after changing the dataset.
After training, select **Load as active model**, then use **Backend model**
in Image or Live Detection. For uploaded videos, use **Backend · all frames**.

| Dataset | Images | Classes | Size | Format | Licence |
|---|---|---|---|---|---|
| Open Images · Pens and phones | 60 | 2 | ~30 MB | sample | Open Images annotations (CC BY 4.0) |
| Wikimedia Commons · Combs and phone cases | 12 | 2 | ~2 MB | sample | Per-image Creative Commons attribution |
| Objects365 sample | 10 | 46 | ~5 MB | sample | Objects365 terms |
| COCO128 | 128 | 80 (COCO) | ~6.7 MB | YOLO txt | COCO 2017, CC BY 4.0 |
| COCO8 | 8 (4 train / 4 val) | 80 (COCO) | ~0.4 MB | YOLO txt | COCO 2017, CC BY 4.0 |
| COCO 2017 validation | 5,000 | 80 (COCO) | ~1.0 GB | COCO JSON | COCO 2017, CC BY 4.0 |
| Pascal VOC 2012 trainval | 11,530 | 20 | ~1.9 GB | VOC XML | Research use |
| Pascal VOC 2007 trainval | 5,011 | 20 | ~439 MB | VOC XML | Research use |
| WIDER FACE train | 12,880 | 1 (face) | ~1.4 GB | WIDER txt | Non-commercial research |
| VisDrone2019-DET train | 6,471 | 10 | ~1.4 GB | VisDrone txt | Non-commercial research |
| VisDrone2019-DET val | 548 | 10 | ~72 MB | VisDrone txt | Non-commercial research |

- COCO 2017 validation — images `http://images.cocodataset.org/zips/val2017.zip`
  + annotations `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`
  (the importer uses `instances_val2017.json`).
- Pascal VOC 2012 — `http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar`
  (VOC 2007: `.../voc2007/VOCtrainval_06-Nov-2007.tar`). The importer reads
  the `Annotations/*.xml` files and skips objects marked difficult.
- WIDER FACE train — images from the Hugging Face mirror
  `https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_train.zip`
  + annotations `.../wider_face_split.zip` (`wider_face_train_bbx_gt.txt`).
  Faces flagged invalid are skipped.
- VisDrone2019-DET — `https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip`
  (and `-val.zip`). Annotation rows with a non-zero score are ignored
  regions and are skipped, as in the Ultralytics converter.
- Source: the zips contain real images and real labels — nothing is
  invented. Downloaded archives live in `downloads/` and imported files in
  `dataset/images/all/` + `dataset/labels/all/` — all git-ignored, so they
  are never committed. Re-downloading an imported dataset is skipped unless
  you choose Re-download.

### More datasets — download from the official site

These are too large, or need registration, for a one-click download, so the
Dataset page links to their official download pages instead of downloading
them. They are listed in the same catalog (`GET /api/datasets`).

| Dataset | Images | Classes | Size | How to get it |
|---|---|---|---|---|
| COCO 2017 train | 118,287 | 80 | ~18 GB | `http://images.cocodataset.org/zips/train2017.zip` (+ annotations zip) |
| LVIS v1 train | 100,170 | 1,203 | ~18 GB + 334 MB annotations | `https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip` + COCO train2017 images |
| LVIS v1 val | 19,809 | 1,203 | ~19 GB + 61 MB annotations | `https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip` + COCO train2017 + val2017 images |
| Open Images V7 | 1,743,042 (train) | 600 boxable | ~561 GB | `https://storage.googleapis.com/openimages/web/download_v7.html` (CC BY 2.0) |
| BDD100K | 100,000 | 10 | ~7.1 GB | `https://bdd-data.berkeley.edu/` — registration required |
| SKU-110K | 11,762 | 1 (object) | ~13.6 GB | `https://github.com/eg4000/SKU110K_CVPR19` — academic / non-commercial |
| Objects365 (full) | ~2 million | 365 | see official site | `https://www.objects365.org/download.html` — see “Objects365 training data” above |

- LVIS v1 annotations are COCO-format, so once the COCO images are
  downloaded, import them with Dataset → Import annotated ZIP. LVIS images
  have no `file_name` — the importer reads the name from `coco_url`.
  Verified on the real `lvis_v1_val.json` (21 Nov 2026): 1,203 classes,
  19,809 images, all 244,707 boxes convert to YOLO labels. LVIS val spans
  COCO train2017 (15,000 images) and val2017 (4,809), which is why it is
  not a one-click download.
- BDD100K's site could not be reached from this environment when checked
  (21 Nov 2026); the registration requirement is confirmed by the FiftyOne
  dataset docs.

For development with hot reload instead of the built frontend:

```powershell
# terminal 1 — backend (from project root)
.venv\Scripts\activate
python run.py
# terminal 2 — frontend dev server (proxies /api to :8000)
cd frontend
npm run dev
# open http://localhost:5173
```

Linux/macOS: same steps with `.venv/bin/activate` and `python3`.

## Open-vocabulary detection (search by object name)

The standard models (yolov8n/s/m/x) only know the 80 COCO classes — a
"person", "car", "dog" and 77 others. To look for objects beyond COCO, use
**YOLO-World** (`yolov8s-worldv2.pt`, Ultralytics): it is an
open-vocabulary detector, so you type the object names in plain English and
it searches for them — for example, "red bus", "traffic light", "laptop",
or "cricket bat". This mode works without a training dataset.

How to use it:

1. **Image Detection** or **Live Detection** — switch the toggle to
   **"Any object — type its name"** and type comma-separated object names
   (e.g. `red bus, traffic light, dog`, up to 20), then run detection as
   usual. Results show your prompted names, with the same boxes, confidence
   and history records as COCO detection (history type `open-vocab`).
2. Or pick **YOLOv8s-World v2** in **Settings → Model** — Image and Live
   then start in Any-object mode automatically.
3. API: `POST /api/detect/open-vocab/image` (multipart image + `classes`
   form field, JSON array or comma-separated text) and
   `POST /api/detect/open-vocab/frame` (live frames, no history record).
   `GET /api/models/available` lists all selectable models.

Notes and limits (honest):

- The world weights (~25 MB) download once from Ultralytics on first use
  into `models/` (git-ignored). Without internet the first open-vocab call
  returns a clear 503 error — it never invents boxes.
- Prompt wording matters: specific names ("red bus") work better than vague
  ones ("thing"). It is slower and less accurate than a COCO model on the
  classes that model was trained for, and it can occasionally mistake a
  similar-looking object — always check the confidence and the box.
- It does not "know every image in the world" — it matches your words
  against what it sees using a vision-language model, so unusual or very
  small objects may be missed. For a fixed set of objects you detect often,
  a custom-trained model (Module 2) is still the more accurate choice.

## Using the app

- **Overview Dashboard** — totals, class distribution and activity from
  recorded detections only. Empty states until the first real detection.
- **Live Detection** — choose **Enable camera names** or **Start webcam**
  and allow browser camera access. The camera list refreshes after permission,
  when the window regains focus, and when the browser reports a device change.
  Stop the stream before switching cameras. Frames are sampled up to ~6 fps;
  boxes/labels/confidence are drawn on a canvas overlay with measured FPS
  and inference ms. Auto-save stores the first detected frame and then up to one every 30 seconds in browser history. Save snapshot stores a result immediately. Small objects (expanded) checks cropped regions of larger frames and uses more processing time.
- **Image Detection** — upload JPG, PNG, WebP, GIF, BMP or AVIF (up to 20 MB),
  run detection, and download the annotated image. Animated images use the first
  frame on the backend. Convert HEIC photos to JPEG before uploading.
- **Dataset Preparation** — upload images, add classes, click an image and
  drag to draw boxes, Save annotations, set the train % and Split — this
  writes `dataset/data.yaml` and the `images/{train,val}` +
  `labels/{train,val}` folders. Export downloads a zip.
- **Model Training** — pick a variant/epochs/imgsz/batch, Start training.
  Progress and metrics are read live from the trainer's `results.csv`.
  Stop halts after the current epoch. `best.pt` is copied to `models/` and
  can be downloaded or loaded as the active model.
- **Settings** — model path, confidence, IoU, image size, device. All are
  applied to real inference calls.
- **Detection History** — search, filter, view, download, and delete saved
  image runs, video sessions, and live snapshots. The page includes browser
  records and any available backend records. Clearing the site's data removes
  browser records. Analytics and CSV/PDF reports use backend records.

## LabelImg notes

LabelImg is the classic desktop annotation tool (YOLO format). This app
includes an equivalent in-browser annotator (Module 1) that writes the same
YOLO txt format (`class cx cy w h`, normalised 0–1), so labels created here
— or in LabelImg — are interchangeable: copy LabelImg's `labels/*.txt` files
into `dataset/labels/all/` and the matching images into
`dataset/images/all/`, then Split in Module 1. Class order must match
`dataset/classes.json` / `data.yaml` `names:`.

## API (selected)

| Endpoint | Description |
|---|---|
| `GET /api/health` | Backend + model-loaded status |
| `GET /api/model/status` | Loaded model, device, classes, thresholds |
| `POST /api/model/load` `{path}` | Load a model file (e.g. `models/best.pt`) |
| `GET /api/models/available` | Selectable models: yolov8n/s/m/x (COCO) + yolov8s-worldv2 (open-vocab) |
| `POST /api/detect/image` | Multipart image → detections + annotated image URL |
| `POST /api/detect/open-vocab/image` | Multipart image + `classes` (JSON array or comma text) → YOLO-World detections |
| `POST /api/detect/open-vocab/frame` | Multipart JPEG frame + `classes` → YOLO-World detections (live, no history) |
| `POST /api/detect/frame` | Multipart JPEG frame → detections (live webcam) |
| `POST /api/detect/frame/save` | Same, plus records the frame in history |
| `POST /api/video/start` | Multipart video → job id |
| `GET /api/video/{id}` | Job progress / summary |
| `GET /api/video/{id}/download` | Processed MP4 |
| `GET /api/datasets` | Curated dataset catalog + download status |
| `POST /api/datasets/{id}/download` | Download + import a built-in dataset (background) |
| `GET /api/datasets/{id}/status` | Download/import progress |
| `POST /api/dataset/upload` | Multi-image upload |
| `POST /api/dataset/annotations/{name}` | Save YOLO boxes for an image |
| `POST /api/dataset/split` `{train_pct}` | Split + write `data.yaml` |
| `GET /api/dataset/export` | Dataset zip |
| `POST /api/training/start` | Start real training job |
| `GET /api/training/status` | Epoch, metrics (from `results.csv`), logs |
| `POST /api/training/stop` | Stop after current epoch |
| `GET /api/history` | Search/filter detection records |
| `GET /api/analytics/summary` | Totals from SQLite only |
| `POST /api/reports/generate` `{format: csv\|pdf}` | Report file URL |

## Troubleshooting

- **Model not loaded / first inference slow** — first load downloads
  `yolov8n.pt` (~6 MB). Check internet, then retry. `models/` must be writable.
- **Webcam won't start** — the browser must be allowed camera access, and
  the page must be served over `localhost` (or HTTPS). If denied, use Image
  detection instead; the error is shown in the UI.
- **Training fails: "dataset/data.yaml not found"** — complete Module 1
  (upload + classes + split) first. Training needs annotated images; images
  with no annotations produce a warning at split time.
- **Training is slow on CPU** — expected. Use `yolov8n`, `imgsz 320–640`,
  few epochs and a small dataset for a smoke test; a GPU (CUDA) makes it
  much faster.
- **Video processing slow** — also CPU-bound; progress is real frame count,
  so a long video takes a while. Short clips first.
- **Port 8000 in use** — stop the other process, or edit the port in `run.py`.
- **"Could not decode image"** — upload a valid JPG, PNG, WebP, GIF, BMP or AVIF under 20 MB.
- **Charts empty** — by design: analytics only show recorded detections.
  Run a detection first.

## Honesty notes

- Every detection result comes from `model.predict()` on the loaded YOLOv8
  weights. There is no demo mode and no simulated data anywhere in the app.
- FPS shown is measured on the machine running the backend; 20+ FPS is a
  target, not a guarantee — CPU inference on large frames is slower.
- Training metrics are read from the trainer's own `results.csv`; if a run
  fails, the real error is shown in the log and status, not a fake progress bar.

## Host the backend on Hugging Face (free)

The backend runs on Hugging Face Spaces (Docker SDK, free CPU tier) so the
live Vercel frontend can reach it. The repo already contains a `Dockerfile`
(two-stage: Node builds the frontend, Python 3.12 runs FastAPI on `$PORT`,
default 7860 on Spaces).

1. Create a free Hugging Face account, then a fine-grained access token with
   **write** access to Spaces (Settings → Access Tokens).
2. Create a Space with SDK **Docker** (e.g. `object-detection-backend`).
3. Push this repo's files into the Space repo (the `Dockerfile` must be at
   the root). The Space builds the image and starts the backend.
4. When the Space is running, copy its URL (e.g.
   `https://<user>-object-detection-backend.hf.space`) and set it as
   `VITE_API_URL` in the Vercel project, then redeploy the frontend.
   The site then shows "Backend connected" and detection, training
   and history all work live.

Notes: Space storage is ephemeral — detection history and uploads reset when
the Space restarts. Training on the free CPU tier is slow; use small
datasets (COCO8/COCO128) and few epochs. The YOLO-World model downloads
(~25 MB) on first open-vocab use.
