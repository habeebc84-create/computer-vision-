"""Real YOLOv8 inference engine. No fake detections: every box comes from
the loaded Ultralytics model. If no model is available, calls raise."""
import json, os, threading, time
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent
SETTINGS_FILE = BASE / "settings.json"
DEFAULTS = {"model_path": "models/yolov8n.pt", "conf": 0.25, "iou": 0.45,
            "imgsz": 640, "device": "auto"}

def load_settings():
    s = dict(DEFAULTS)
    try:
        if SETTINGS_FILE.exists():
            s.update(json.loads(SETTINGS_FILE.read_text()))
    except Exception:
        pass
    return s

def save_settings(patch):
    s = load_settings(); s.update(patch)
    SETTINGS_FILE.write_text(json.dumps(s, indent=2)); return s

class ModelManager:
    def __init__(self):
        self._lock = threading.RLock()
        self.model = None
        self.model_path = None
        self.class_names = {}
        self.load_error = None
        self.loaded_at = None

    def device(self):
        s = load_settings()
        if s.get("device") not in (None, "auto"):
            return s["device"]
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def load(self, path=None):
        """Load (or reload) a YOLO model. Downloads yolov8n.pt on first use."""
        from ultralytics import YOLO
        s = load_settings()
        p = Path(path or s.get("model_path") or "models/yolov8n.pt")
        if not p.is_absolute():
            p = BASE / p
        with self._lock:
            self.load_error = None
            try:
                # Ultralytics downloads known pretrained weights to the requested
                # path, but cannot create a missing parent directory itself.
                if p == BASE / "models" / "yolov8n.pt":
                    p.parent.mkdir(parents=True, exist_ok=True)
                self.model = YOLO(str(p))
                self.model_path = str(p)
                self.class_names = dict(getattr(self.model, "names", {}) or {})
                self.loaded_at = time.strftime("%Y-%m-%d %H:%M:%S")
                return True
            except Exception as e:
                self.model = None
                if p == BASE / "models" / "yolov8n.pt" and not p.is_file():
                    self.load_error = ("Could not load the default model. Download the official "
                                       "yolov8n.pt weights into models/yolov8n.pt, or check "
                                       f"network access to GitHub releases. Details: {e}")
                    raise RuntimeError(self.load_error) from e
                self.load_error = str(e)
                raise

    def ensure_loaded(self):
        with self._lock:
            if self.model is None:
                self.load()
            return self.model

    def status(self):
        s = load_settings()
        return {"loaded": self.model is not None,
                "model_path": self.model_path or str(s.get("model_path")),
                "device": self.device(),
                "classes": len(self.class_names),
                "class_names": list(self.class_names.values()),
                "loaded_at": self.loaded_at, "error": self.load_error,
                "conf": s.get("conf"), "iou": s.get("iou"), "imgsz": s.get("imgsz")}

    def detect_array(self, bgr):
        """Run real inference on a BGR numpy image. Returns (detections, ms)."""
        model = self.ensure_loaded()
        s = load_settings()
        t0 = time.perf_counter()
        with self._lock:
            results = model.predict(source=bgr, conf=float(s["conf"]),
                                    iou=float(s["iou"]), imgsz=int(s["imgsz"]),
                                    device=self.device(), verbose=False)
        ms = (time.perf_counter() - t0) * 1000.0
        dets = []
        if results:
            r = results[0]
            if r.boxes is not None:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)
                for box, c, cl in zip(xyxy, confs, clss):
                    dets.append({"class_id": int(cl),
                                 "class_name": self.class_names.get(int(cl), str(int(cl))),
                                 "confidence": round(float(c), 4),
                                 "bbox": [round(float(v), 1) for v in box]})
        return dets, ms

    def annotate(self, bgr, dets):
        img = bgr.copy()
        palette = [(56,189,248),(34,211,238),(168,85,247),(52,211,153),
                   (251,146,60),(244,114,182),(250,204,21),(96,165,250)]
        for i, d in enumerate(dets):
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            color = palette[d["class_id"] % len(palette)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{d['class_name']} {d['confidence']*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(img, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (10, 15, 30), 2, cv2.LINE_AA)
        return img

manager = ModelManager()
