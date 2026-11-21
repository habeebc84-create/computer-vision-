"""Video processing: frame-by-frame YOLO inference in a background thread,
progress polling, processed MP4 output. Runs on real frames only."""
import threading, time
from pathlib import Path
import cv2

class VideoJob:
    def __init__(self, job_id, src, dst, manager):
        self.id = job_id; self.src = src; self.dst = dst; self.manager = manager
        self.status = "queued"; self.progress = 0.0; self.frames = 0; self.total_frames = 0
        self.objects = 0; self.fps = 0.0; self.error = None; self.detections = []
        self._t = threading.Thread(target=self._run, daemon=True)

    def start(self): self._t.start()

    def _run(self):
        try:
            self.status = "processing"
            cap = cv2.VideoCapture(str(self.src))
            if not cap.isOpened(): raise RuntimeError("Could not open uploaded video.")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = total
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(self.dst), fourcc, fps_in, (w, h))
            t0 = time.time(); seen = {}
            while True:
                ok, frame = cap.read()
                if not ok: break
                dets, _ms = self.manager.detect_array(frame)
                self.detections = dets
                for d in dets: seen[d["class_name"]] = seen.get(d["class_name"], 0) + 1
                self.objects += len(dets)
                out.write(self.manager.annotate(frame, dets))
                self.frames += 1
                if total: self.progress = round(self.frames/total*100, 1)
            cap.release(); out.release()
            el = max(time.time()-t0, 0.01)
            self.fps = round(self.frames/el, 2)
            self.status = "completed"
            self.summary = {"frames": self.frames, "objects": self.objects,
                            "fps": self.fps, "per_class": seen}
        except Exception as e:
            self.status = "failed"; self.error = str(e)

    def snapshot(self):
        return {"id": self.id, "status": self.status, "progress": self.progress,
                "frames": self.frames, "total_frames": self.total_frames,
                "objects": self.objects, "fps": self.fps, "error": self.error,
                "summary": getattr(self, "summary", None)}
