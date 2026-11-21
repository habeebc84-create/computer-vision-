"""FastAPI backend — Real-Time Object Detection System (YOLOv8).
Serves the built React frontend statically and exposes /api/* endpoints.
Run:  python run.py   ->  http://localhost:8000
"""
import json, logging, shutil, sqlite3, time, uuid
from pathlib import Path
from threading import Lock
from zipfile import BadZipFile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from coco_import import MAX_ARCHIVE_BYTES, import_coco
from detection import (AVAILABLE_MODELS, load_settings, manager, save_settings,
                     validate_open_vocab_classes, world_manager)
from dataset_manager import (IMG_ALL, LBL_ALL, MAX_UPLOAD_BYTES, ensure_dirs, get_annotations,
                             insert_image,
                             get_classes, list_images, save_annotation, set_classes,
                             split_dataset, stats, export_zip)
from model_training import job as train_job
import datasets as builtin_datasets
from video_processing import VideoJob
from image_processing import decode_image, encode_jpeg
import analytics, report_generator

BASE = Path(__file__).resolve().parent.parent
_split_and_training_lock = Lock()
OUT_IMG = BASE / "outputs" / "images"; OUT_VID = BASE / "outputs" / "videos"
OUT_REP = BASE / "outputs" / "reports"; UP = BASE / "uploads"
# Defensive: a fresh clone has none of these directories (git-ignored/empty),
# and the /dataset static mount below crashes at import time if dataset/
# does not exist. run.py creates them too; this covers importing app.py
# directly (uvicorn, tests). dataset_manager.ensure_dirs() adds the
# dataset/images + dataset/labels subdirectories.
for d in (OUT_IMG, OUT_VID, OUT_REP, UP, BASE / "dataset", BASE / "models"):
    d.mkdir(parents=True, exist_ok=True)
ensure_dirs()

app = FastAPI(title="Real-Time Object Detection System", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
analytics.init_db()
video_jobs = {}
_frame_times = []

# ---------- Health / model / settings ----------
@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": manager.model is not None,
            "device": manager.device(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}

@app.get("/api/model/status")
def model_status():
    return manager.status()

@app.post("/api/model/load")
def model_load(payload: dict):
    path = payload.get("path", "")
    if not path: raise HTTPException(400, "Provide a model path, e.g. models/best.pt")
    try:
        manager.load(path)
    except Exception as e:
        raise HTTPException(400, f"Could not load model: {e}")
    save_settings({"model_path": path})
    return manager.status()

@app.get("/api/models/available")
def models_available():
    """Models the UI can choose: COCO (80 fixed classes) + YOLO-World
    (open-vocabulary — any typed object). Includes per-model download state."""
    out = []
    for m in AVAILABLE_MODELS:
        entry = dict(m)
        entry["downloaded"] = (BASE / m["path"]).exists()
        if m["kind"] == "trained" and not entry["downloaded"]:
            continue
        if m["kind"] == "world":
            entry["loaded"] = world_manager.model is not None
        out.append(entry)
    return out

@app.get("/api/settings")
def get_settings(): return load_settings()

@app.post("/api/settings")
def post_settings(payload: dict):
    allowed = {"model_path", "conf", "iou", "imgsz", "device"}
    patch = {k: v for k, v in payload.items() if k in allowed}
    if "conf" in patch: patch["conf"] = max(0.05, min(0.95, float(patch["conf"])))
    if "iou" in patch: patch["iou"] = max(0.1, min(0.95, float(patch["iou"])))
    if "imgsz" in patch: patch["imgsz"] = int(patch["imgsz"])
    if "model_path" in patch:
        try: manager.load(patch["model_path"])
        except Exception as e: raise HTTPException(400, f"Model failed to load: {e}")
    return save_settings(patch)

# ---------- Detection ----------
def _summary(dets, ms):
    return {"detections": dets, "count": len(dets),
            "unique_classes": sorted({d["class_name"] for d in dets}),
            "avg_confidence": round(sum(d["confidence"] for d in dets)/len(dets), 4) if dets else 0.0,
            "inference_ms": round(ms, 1),
            "fps": round(1000/max(ms, 0.01), 1)}

def _store_detection(kind, annotated, dets, ms, model):
    name = f"{kind}_{uuid.uuid4().hex}.jpg"
    path = OUT_IMG / name
    output_path = f"outputs/images/{name}"
    try:
        path.write_bytes(encode_jpeg(annotated))
        record_id = analytics.record(kind, dets, fps=1000/max(ms, .01), inference_ms=ms,
                                     output_path=output_path, model=model or "")
    except (OSError, sqlite3.Error) as exc:
        logging.getLogger(__name__).exception("Could not save a detection record")
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logging.getLogger(__name__).exception("Could not remove an incomplete detection output")
        if kind == "live":
            raise HTTPException(503, "Snapshot could not be saved to history. Check backend storage.") from exc
        return {"history_error": "Detection completed, but history storage is unavailable."}
    return {"record_id": record_id, "annotated_url": f"/{output_path}"}

@app.post("/api/detect/image")
async def detect_image(file: UploadFile = File(...)):
    data = await file.read(20*1024*1024 + 1)
    if len(data) > 20*1024*1024: raise HTTPException(400, "Image larger than 20 MB.")
    try: img = decode_image(data)
    except ValueError as e: raise HTTPException(400, str(e))
    try: dets, ms = manager.detect_array(img)
    except Exception as e: raise HTTPException(503, f"Model error: {e}")
    annotated = manager.annotate(img, dets)
    out = _summary(dets, ms)
    out.update(_store_detection("image", annotated, dets, ms, manager.model_path))
    out["original_name"] = file.filename
    return out

@app.post("/api/detect/frame")
async def detect_frame(frame: UploadFile = File(...)):
    """Live webcam frame (JPEG blob from the browser). Not stored in history
    unless the client calls /api/detect/frame/save — avoids spamming the DB."""
    data = await frame.read()
    try: img = decode_image(data)
    except ValueError as e: raise HTTPException(400, str(e))
    try: dets, ms = manager.detect_array(img)
    except Exception as e: raise HTTPException(503, f"Model error: {e}")
    now = time.time(); _frame_times.append(now)
    while _frame_times and now - _frame_times[0] > 2: _frame_times.pop(0)
    out = _summary(dets, ms); out["fps"] = round(len(_frame_times)/2, 1) if len(_frame_times) > 1 else out["fps"]
    return out

def _parse_classes(raw: str):
    """Form field `classes`: a JSON array ('["bus","dog"]') or comma-separated
    text ('bus, dog'). Returns the validated prompt list."""
    if raw is None:
        raise HTTPException(400, "Provide object names in the 'classes' field.")
    raw = raw.strip()
    if not raw:
        raise HTTPException(400, "Provide at least one object name to detect.")
    parsed = raw
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except Exception:
            raise HTTPException(400, "Could not parse 'classes' as JSON or comma-separated text.")
    try:
        return validate_open_vocab_classes(parsed)
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.post("/api/detect/open-vocab/image")
async def detect_open_vocab_image(file: UploadFile = File(...), classes: str = Form("")):
    """Open-vocabulary image detection (YOLO-World): detect any object named
    in `classes`. Real inference only — a missing/unreachable world model is
    a clear 503, never invented boxes."""
    prompts = _parse_classes(classes)
    data = await file.read(20*1024*1024 + 1)
    if len(data) > 20*1024*1024: raise HTTPException(400, "Image larger than 20 MB.")
    try: img = decode_image(data)
    except ValueError as e: raise HTTPException(400, str(e))
    try: dets, ms = world_manager.detect_array(img, prompts)
    except RuntimeError as e: raise HTTPException(503, str(e))
    except Exception as e: raise HTTPException(503, f"Model error: {e}")
    annotated = manager.annotate(img, dets)
    out = _summary(dets, ms)
    out.update(_store_detection("open-vocab", annotated, dets, ms, world_manager.model_path))
    out.update({"original_name": file.filename, "prompts": prompts})
    return out

@app.post("/api/detect/open-vocab/frame")
async def detect_open_vocab_frame(frame: UploadFile = File(...), classes: str = Form("")):
    """Open-vocabulary live webcam frame. No history record (same as /api/detect/frame)."""
    prompts = _parse_classes(classes)
    data = await frame.read()
    try: img = decode_image(data)
    except ValueError as e: raise HTTPException(400, str(e))
    try: dets, ms = world_manager.detect_array(img, prompts)
    except RuntimeError as e: raise HTTPException(503, str(e))
    except Exception as e: raise HTTPException(503, f"Model error: {e}")
    now = time.time(); _frame_times.append(now)
    while _frame_times and now - _frame_times[0] > 2: _frame_times.pop(0)
    out = _summary(dets, ms); out["fps"] = round(len(_frame_times)/2, 1) if len(_frame_times) > 1 else out["fps"]
    out["prompts"] = prompts
    return out

@app.post("/api/detect/frame/save")
async def save_frame(frame: UploadFile = File(...), classes: str | None = Form(None)):
    prompts = _parse_classes(classes) if classes is not None else None
    data = await frame.read()
    try: img = decode_image(data)
    except ValueError as e: raise HTTPException(400, str(e))
    try:
        if prompts is None:
            dets, ms = manager.detect_array(img)
            model_path = manager.model_path
        else:
            dets, ms = world_manager.detect_array(img, prompts)
            model_path = world_manager.model_path
    except Exception as e: raise HTTPException(503, f"Model error: {e}")
    out = _summary(dets, ms)
    out.update(_store_detection("live", manager.annotate(img, dets), dets, ms, model_path))
    return out

# ---------- Video ----------
@app.post("/api/video/start")
async def video_start(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".webm"}:
        raise HTTPException(400, "Upload an MP4, AVI, MOV or WebM video.")
    jid = uuid.uuid4().hex[:12]
    src = UP / f"{jid}{suffix}"
    if file.size == 0:
        raise HTTPException(400, "The video file is empty.")
    if file.size is not None and file.size > 100 * 1024 * 1024:
        raise HTTPException(413, "Video exceeds 100 MB.")
    try:
        with src.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                output.write(chunk)
        dst = OUT_VID / f"{jid}_detected.mp4"
        job = VideoJob(jid, src, dst, manager)
        video_jobs[jid] = job
        job.start()
    except Exception:
        src.unlink(missing_ok=True)
        video_jobs.pop(jid, None)
        raise
    return {"job_id": jid}

@app.get("/api/video/{jid}")
def video_status(jid: str):
    j = video_jobs.get(jid)
    if not j: raise HTTPException(404, "Video job not found.")
    s = j.snapshot()
    if s["status"] == "completed":
        with j.record_lock:
            if not j.recorded:
                summ = s["summary"]
                try:
                    analytics.record_summary("video", summ["objects"], list(summ["per_class"]),
                                             fps=s["fps"], output_path=f"outputs/videos/{jid}_detected.mp4",
                                             model=manager.model_path or "")
                    j.recorded = True
                except (OSError, sqlite3.Error):
                    logging.getLogger(__name__).exception("Could not save video history")
                    s["history_error"] = "Processed video is ready, but history storage is unavailable."
    return s

@app.get("/api/video/{jid}/download")
def video_download(jid: str):
    p = OUT_VID / f"{jid}_detected.mp4"
    job = video_jobs.get(jid)
    if (job and job.status != "completed") or not p.is_file():
        raise HTTPException(404, "Processed video not ready yet.")
    return FileResponse(p, media_type="video/mp4", filename=f"detected_{jid}.mp4")

# ---------- Dataset ----------
@app.get("/api/dataset/stats")
def ds_stats(): return stats()

@app.get("/api/dataset/images")
def ds_images(): return list_images()

@app.post("/api/dataset/import-coco")
async def ds_import_coco(file: UploadFile = File(...)):
    if file.size is not None and file.size > MAX_ARCHIVE_BYTES:
        raise HTTPException(413, "Dataset ZIP exceeds 100 MB. Import a smaller subset.")
    try:
        return await run_in_threadpool(import_coco, file.file)
    except (BadZipFile, KeyError, TypeError, AttributeError, ValueError) as exc:
        raise HTTPException(400, f"Invalid COCO dataset ZIP: {exc}") from exc

@app.post("/api/dataset/upload")
async def ds_upload(files: list[UploadFile] = File(...)):
    ensure_dirs()
    saved, skipped, errors = [], [], []
    for file in files:
        name = file.filename or "image"
        try:
            data = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                raise ValueError("Image exceeds 20 MB.")
            saved.append(await run_in_threadpool(insert_image, name, data))
        except ValueError as exc:
            skipped.append(name)
            errors.append({"file": name, "message": str(exc)})
        except OSError:
            logging.getLogger(__name__).exception("Could not store a dataset image")
            skipped.append(name)
            errors.append({"file": name, "message": "The backend could not write the image. Check available storage."})
    return {"saved": saved, "skipped": skipped, "errors": errors, "stats": stats()}

@app.delete("/api/dataset/images/{name}")
def ds_delete(name: str):
    for base, pat in ((IMG_ALL, name), (LBL_ALL, Path(name).stem + ".txt")):
        p = base / pat
        if p.exists(): p.unlink()
    return {"deleted": name, "stats": stats()}

@app.get("/api/dataset/classes")
def ds_classes(): return {"classes": get_classes()}

@app.post("/api/dataset/classes")
def ds_set_classes(payload: dict):
    try: return {"classes": set_classes(payload.get("classes", []))}
    except ValueError as exc: raise HTTPException(400, str(exc)) from exc

@app.get("/api/dataset/annotations/{name}")
def ds_get_ann(name: str):
    try: return get_annotations(name)
    except FileNotFoundError: raise HTTPException(404, "Image not found.")

@app.post("/api/dataset/annotations/{name}")
def ds_set_ann(name: str, payload: dict):
    try: n = save_annotation(name, payload.get("boxes", []))
    except FileNotFoundError: raise HTTPException(404, "Image not found.")
    except ValueError as exc: raise HTTPException(400, str(exc)) from exc
    except (TypeError, KeyError) as exc: raise HTTPException(400, "Invalid annotation boxes.") from exc
    return {"saved": n, "stats": stats()}

@app.post("/api/dataset/split")
def ds_split(payload: dict):
    with _split_and_training_lock:
        if train_job.status == "running":
            raise HTTPException(409, "Wait for training to finish before rebuilding the dataset split.")
        try: return split_dataset(float(payload.get("train_pct", 80)))
        except (TypeError, ValueError) as e: raise HTTPException(400, str(e)) from e

@app.get("/api/dataset/data-yaml")
def ds_yaml():
    p = BASE / "dataset" / "data.yaml"
    if not p.exists(): raise HTTPException(404, "data.yaml not generated yet — split the dataset first.")
    return {"content": p.read_text()}

@app.get("/api/dataset/export")
def ds_export():
    dest = OUT_REP / "dataset_export.zip"; export_zip(dest)
    return FileResponse(dest, media_type="application/zip", filename="dataset_export.zip")

# ---------- Curated built-in datasets (one-click download) ----------
@app.get("/api/datasets")
def ds_catalog():
    return builtin_datasets.catalog()

@app.post("/api/datasets/{ds_id}/download")
def ds_download(ds_id: str, force: bool = False):
    try:
        return builtin_datasets.start_download(ds_id, force=force)
    except KeyError:
        raise HTTPException(404, f"Unknown dataset: {ds_id}")

@app.get("/api/datasets/{ds_id}/status")
def ds_status(ds_id: str):
    try:
        return builtin_datasets.status(ds_id)
    except KeyError:
        raise HTTPException(404, f"Unknown dataset: {ds_id}")

# ---------- Training ----------
@app.post("/api/training/start")
def tr_start(payload: dict):
    cfg = {"model": payload.get("model", "yolov8n.pt"), "epochs": int(payload.get("epochs", 10)),
           "imgsz": int(payload.get("imgsz", 640)), "batch": int(payload.get("batch", 8)),
           "device": payload.get("device", "cpu")}
    with _split_and_training_lock:
        try: return train_job.start(cfg)
        except (FileNotFoundError, RuntimeError, ValueError) as e: raise HTTPException(400, str(e)) from e

@app.get("/api/training/status")
def tr_status(): return train_job.snapshot()

@app.post("/api/training/stop")
def tr_stop(): return train_job.stop()

@app.get("/api/training/model/download")
def tr_download():
    p = BASE / "models" / "best.pt"
    if not p.exists(): raise HTTPException(404, "No best.pt yet — finish a training run first.")
    return FileResponse(p, media_type="application/octet-stream", filename="best.pt")

@app.post("/api/training/load-best")
def tr_load_best():
    p = BASE / "models" / "best.pt"
    if not p.exists(): raise HTTPException(404, "No best.pt found.")
    try: manager.load(str(p))
    except Exception as e: raise HTTPException(400, str(e))
    save_settings({"model_path": "models/best.pt"})
    return manager.status()

# ---------- History / analytics / reports ----------
@app.get("/api/history")
def hist(search: str = "", type: str = "", date_from: str = "", date_to: str = ""):
    return analytics.list_history(search, type, date_from, date_to)

@app.get("/api/history/{rid}")
def hist_one(rid: int):
    r = analytics.get_one(rid)
    if not r: raise HTTPException(404, "Record not found.")
    return r

@app.delete("/api/history/{rid}")
def hist_del(rid: int):
    analytics.delete_one(rid); return {"deleted": rid}

@app.get("/api/history/{rid}/download")
def hist_download(rid: int):
    r = analytics.get_one(rid)
    if not r or not r.get("output_path"): raise HTTPException(404, "No output file for this record.")
    p = BASE / r["output_path"]
    if not p.exists(): raise HTTPException(404, "Output file no longer exists on disk.")
    return FileResponse(p, filename=p.name)

@app.get("/api/analytics/summary")
def an_summary(): return analytics.summary()

@app.get("/api/analytics/class-distribution")
def an_classes(): return analytics.class_distribution()

@app.get("/api/analytics/activity")
def an_activity(): return analytics.activity()

@app.get("/api/analytics/confidence")
def an_conf(): return analytics.confidence_buckets()

@app.post("/api/reports/generate")
def rep_generate(payload: dict):
    fmt = payload.get("format", "csv")
    ids = payload.get("ids")
    recs = analytics.list_history(limit=500)
    if ids: recs = [r for r in recs if r["id"] in ids]
    if not recs: raise HTTPException(400, "No detection records yet — run a detection first.")
    ts = time.strftime("%Y%m%d_%H%M%S")
    if fmt == "pdf":
        dest = OUT_REP / f"report_{ts}.pdf"; report_generator.build_pdf(recs, dest)
    else:
        dest = OUT_REP / f"report_{ts}.csv"; report_generator.build_csv(recs, dest)
    return {"file": dest.name, "url": f"/outputs/reports/{dest.name}", "records": len(recs)}

# ---------- Static frontend + outputs ----------
app.mount("/outputs", StaticFiles(directory=str(BASE / "outputs")), name="outputs")
app.mount("/dataset", StaticFiles(directory=str(BASE / "dataset")), name="dataset")
FRONT = BASE / "frontend" / "dist"
if FRONT.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONT / "assets")), name="assets")
    @app.get("/{path:path}")
    def spa(path: str):
        f = FRONT / path
        if f.is_file(): return FileResponse(f)
        return FileResponse(FRONT / "index.html")
