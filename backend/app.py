"""FastAPI backend — Real-Time Object Detection System (YOLOv8).
Serves the built React frontend statically and exposes /api/* endpoints.
Run:  python run.py   ->  http://localhost:8000
"""
import json, shutil, time, uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from detection import (AVAILABLE_MODELS, load_settings, manager, save_settings,
                     validate_open_vocab_classes, world_manager)
from dataset_manager import (ALLOWED, IMG_ALL, LBL_ALL, ensure_dirs, get_annotations,
                             get_classes, list_images, save_annotation, set_classes,
                             split_dataset, stats, export_zip)
from model_training import job as train_job
import datasets as builtin_datasets
from video_processing import VideoJob
from image_processing import decode_image, encode_jpeg
import analytics, report_generator

BASE = Path(__file__).resolve().parent.parent
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
    s = save_settings(patch)
    if "model_path" in patch:
        try: manager.load(patch["model_path"])
        except Exception as e: raise HTTPException(400, f"Settings saved but model failed to load: {e}")
    return s

# ---------- Detection ----------
def _summary(dets, ms):
    return {"detections": dets, "count": len(dets),
            "unique_classes": sorted({d["class_name"] for d in dets}),
            "avg_confidence": round(sum(d["confidence"] for d in dets)/len(dets), 4) if dets else 0.0,
            "inference_ms": round(ms, 1),
            "fps": round(1000/max(ms, 0.01), 1)}

@app.post("/api/detect/image")
async def detect_image(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(400, "Upload a JPG or PNG image.")
    data = await file.read()
    if len(data) > 20*1024*1024: raise HTTPException(400, "Image larger than 20 MB.")
    try: img = decode_image(data)
    except ValueError as e: raise HTTPException(400, str(e))
    try: dets, ms = manager.detect_array(img)
    except Exception as e: raise HTTPException(503, f"Model error: {e}")
    annotated = manager.annotate(img, dets)
    name = f"{int(time.time())}_{Path(file.filename or 'image').stem}.jpg"
    (OUT_IMG / name).write_bytes(encode_jpeg(annotated))
    rid = analytics.record("image", dets, fps=1000/max(ms, .01), inference_ms=ms,
                           output_path=f"outputs/images/{name}", model=manager.model_path or "")
    out = _summary(dets, ms)
    out.update({"record_id": rid, "annotated_url": f"/outputs/images/{name}",
                "original_name": file.filename})
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
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(400, "Upload a JPG or PNG image.")
    data = await file.read()
    if len(data) > 20*1024*1024: raise HTTPException(400, "Image larger than 20 MB.")
    try: img = decode_image(data)
    except ValueError as e: raise HTTPException(400, str(e))
    try: dets, ms = world_manager.detect_array(img, prompts)
    except RuntimeError as e: raise HTTPException(503, str(e))
    except Exception as e: raise HTTPException(503, f"Model error: {e}")
    annotated = manager.annotate(img, dets)
    name = f"{int(time.time())}_{Path(file.filename or 'image').stem}_world.jpg"
    (OUT_IMG / name).write_bytes(encode_jpeg(annotated))
    rid = analytics.record("open-vocab", dets, fps=1000/max(ms, .01), inference_ms=ms,
                           output_path=f"outputs/images/{name}",
                           model=world_manager.model_path or "models/yolov8s-worldv2.pt")
    out = _summary(dets, ms)
    out.update({"record_id": rid, "annotated_url": f"/outputs/images/{name}",
                "original_name": file.filename, "prompts": prompts})
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
async def save_frame(frame: UploadFile = File(...)):
    data = await frame.read()
    img = decode_image(data)
    dets, ms = manager.detect_array(img)
    name = f"live_{int(time.time())}.jpg"
    (OUT_IMG / name).write_bytes(encode_jpeg(manager.annotate(img, dets)))
    rid = analytics.record("live", dets, fps=1000/max(ms, .01), inference_ms=ms,
                           output_path=f"outputs/images/{name}", model=manager.model_path or "")
    out = _summary(dets, ms); out.update({"record_id": rid, "annotated_url": f"/outputs/images/{name}"})
    return out

# ---------- Video ----------
@app.post("/api/video/start")
async def video_start(file: UploadFile = File(...)):
    if file.content_type not in ("video/mp4", "video/avi", "video/quicktime", "video/x-msvideo"):
        raise HTTPException(400, "Upload an MP4, AVI or MOV video.")
    jid = uuid.uuid4().hex[:12]
    src = UP / f"{jid}_{Path(file.filename or 'video').name}"
    src.write_bytes(await file.read())
    dst = OUT_VID / f"{jid}_detected.mp4"
    job = VideoJob(jid, src, dst, manager); video_jobs[jid] = job; job.start()
    return {"job_id": jid}

@app.get("/api/video/{jid}")
def video_status(jid: str):
    j = video_jobs.get(jid)
    if not j: raise HTTPException(404, "Video job not found.")
    s = j.snapshot()
    if s["status"] == "completed" and not getattr(j, "_recorded", False):
        j._recorded = True
        summ = s.get("summary") or {}
        analytics.record_summary("video", summ.get("objects", 0),
                                 list((summ.get("per_class") or {}).keys()),
                                 fps=s["fps"], output_path=f"outputs/videos/{jid}_detected.mp4",
                                 model=manager.model_path or "")
    return s

@app.get("/api/video/{jid}/download")
def video_download(jid: str):
    p = OUT_VID / f"{jid}_detected.mp4"
    if not p.exists(): raise HTTPException(404, "Processed video not ready yet.")
    return FileResponse(p, media_type="video/mp4", filename=f"detected_{jid}.mp4")

# ---------- Dataset ----------
@app.get("/api/dataset/stats")
def ds_stats(): return stats()

@app.get("/api/dataset/images")
def ds_images(): return list_images()

@app.post("/api/dataset/upload")
async def ds_upload(files: list[UploadFile] = File(...)):
    ensure_dirs(); saved, skipped = [], []
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in ALLOWED: skipped.append(f.filename); continue
        dest = IMG_ALL / Path(f.filename).name
        dest.write_bytes(await f.read()); saved.append(f.filename)
    return {"saved": saved, "skipped": skipped, "stats": stats()}

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
    return {"classes": set_classes(payload.get("classes", []))}

@app.get("/api/dataset/annotations/{name}")
def ds_get_ann(name: str):
    try: return get_annotations(name)
    except FileNotFoundError: raise HTTPException(404, "Image not found.")

@app.post("/api/dataset/annotations/{name}")
def ds_set_ann(name: str, payload: dict):
    try: n = save_annotation(name, payload.get("boxes", []))
    except FileNotFoundError: raise HTTPException(404, "Image not found.")
    return {"saved": n, "stats": stats()}

@app.post("/api/dataset/split")
def ds_split(payload: dict):
    try: return split_dataset(float(payload.get("train_pct", 80)))
    except ValueError as e: raise HTTPException(400, str(e))

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
    try: return train_job.start(cfg)
    except (FileNotFoundError, RuntimeError) as e: raise HTTPException(400, str(e))

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
