# Real-Time Object Detection System — YOLOv8 & Deep Learning

**ITA0523 — Computer Vision capstone project**

A full-stack, production-style web application that detects objects in real time
from a webcam, in uploaded images, and in uploaded videos — using real YOLOv8
inference (Ultralytics, PyTorch, OpenCV). No fake detections, no simulated
results: every bounding box, label and confidence score comes from the model.

## Modules (exactly as specified)

1. **Module 1 — Dataset Preparation** — upload images, annotate with a
   LabelImg-compatible YOLO-txt editor, define classes, split train/val,
   generate `data.yaml`, export the dataset as a zip.
2. **Module 2 — Model Training** — configure a pretrained YOLOv8 variant
   (n/s/m), epochs, image size, batch; start/stop real Ultralytics training in
   a background thread; live epoch / loss / precision / recall / mAP50 /
   mAP50-95 from `results.csv`; `best.pt` saved, downloadable and loadable.
3. **Module 3 — Inference & Visualization** — live webcam (browser camera,
   frames sent to the backend), image detection with annotated output, video
   detection frame-by-frame with progress and downloadable processed video.

Plus: Overview Dashboard, Detection History (SQLite), Analytics & Reports
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

## Why detection doesn't work on Vercel

**Short version: Vercel hosts only the static frontend. There is no Python
backend there, so every `/api/*` call fails.**

This app is two parts: a React frontend and a FastAPI + PyTorch + YOLOv8
backend. Vercel is a static/serverless host — it can serve the dashboard UI,
but it cannot run the backend: PyTorch + Ultralytics are hundreds of MB (far
over Vercel's serverless function size limit), training and video processing
run for minutes (over the request timeout), and the backend keeps a model in
memory and writes to SQLite/disk, which serverless functions don't do.

What you see on a Vercel deploy: the dashboard loads, the header shows an
amber **"Backend offline"** banner, and every detection/training/history
action fails with a clear message instead of a wall of HTML — the frontend
never pretends the backend is there.

**To make a Vercel frontend actually detect objects, point it at a backend:**

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

## Built-in datasets (one-click download)

Module 1 — Dataset has a **"Recommended datasets"** section with two official
Ultralytics COCO samples, verified live (HTTP 200) — click **Download &
import** and the backend downloads, verifies and imports them into
`dataset/images/all/` + `dataset/labels/all/`, with class names set to the
80 COCO classes. Then Split the dataset and train in Module 2 as usual.

| Dataset | Images | Classes | Size | Licence |
|---|---|---|---|---|
| COCO128 | 128 | 80 (COCO) | ~6.7 MB | COCO 2017, CC BY 4.0 |
| COCO8 | 8 (4 train / 4 val) | 80 (COCO) | ~0.4 MB | COCO 2017, CC BY 4.0 |

- COCO128 — `https://ultralytics.com/assets/coco128.zip` — recommended for
  real training: big enough to learn from, small enough to train on CPU.
- COCO8 — `https://ultralytics.com/assets/coco8.zip` — tiny smoke test for
  the download → split → train pipeline.
- Source: Ultralytics (redirects to GitHub releases). No labels or images are
  invented — the zips contain real COCO images and real YOLO txt labels.
- Downloaded zips live in `downloads/` and imported files in
  `dataset/images/all/` + `dataset/labels/all/` — all git-ignored, so they
  are never committed. Re-downloading an imported dataset is skipped unless
  you choose Re-download.

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

## Using the app

- **Overview Dashboard** — totals, class distribution and activity from
  recorded detections only. Empty states until the first real detection.
- **Live Detection** — pick a camera, Start webcam, allow browser camera
  access. Frames are captured at ~6 fps and POSTed to `/api/detect/frame`;
  boxes/labels/confidence are drawn on a canvas overlay with measured FPS
  and inference ms. Save snapshot stores that frame in history.
- **Image Detection** — upload JPG/PNG, Run detection, download the
  annotated image.
- **Video Detection** — upload MP4/AVI/MOV, watch real frame progress,
  download the processed MP4.
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
- **History / Analytics** — search, filter, view, download and delete
  records; charts from recorded data; CSV/PDF report export.

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
| `POST /api/detect/image` | Multipart image → detections + annotated image URL |
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
  or Video detection instead; the error is shown in the UI.
- **Training fails: "dataset/data.yaml not found"** — complete Module 1
  (upload + classes + split) first. Training needs annotated images; images
  with no annotations produce a warning at split time.
- **Training is slow on CPU** — expected. Use `yolov8n`, `imgsz 320–640`,
  few epochs and a small dataset for a smoke test; a GPU (CUDA) makes it
  much faster.
- **Video processing slow** — also CPU-bound; progress is real frame count,
  so a long video takes a while. Short clips first.
- **Port 8000 in use** — stop the other process, or edit the port in `run.py`.
- **"Could not decode image"** — upload a valid JPG/PNG under 20 MB.
- **Charts empty** — by design: analytics only show recorded detections.
  Run a detection first.

## Honesty notes

- Every detection result comes from `model.predict()` on the loaded YOLOv8
  weights. There is no demo mode and no simulated data anywhere in the app.
- FPS shown is measured on the machine running the backend; 20+ FPS is a
  target, not a guarantee — CPU inference on large frames is slower.
- Training metrics are read from the trainer's own `results.csv`; if a run
  fails, the real error is shown in the log and status, not a fake progress bar.
