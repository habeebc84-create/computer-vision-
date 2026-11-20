"""Video processing: frame-by-frame YOLO inference in a background thread,
progress polling, processed MP4 output. Runs on real frames only."""
import math
import threading, time
import cv2

class VideoJob:
    def __init__(self, job_id, src, dst, manager):
        self.id = job_id; self.src = src; self.dst = dst; self.manager = manager
        self.status = "queued"; self.progress = 0.0; self.frames = 0; self.total_frames = 0
        self.objects = 0; self.fps = 0.0; self.error = None; self.detections = []
        self.record_lock = threading.Lock()
        self.recorded = False
        self._t = threading.Thread(target=self._run, daemon=True)

    def start(self): self._t.start()

    def _run(self):
        cap = out = None
        try:
            self.status = "processing"
            cap = cv2.VideoCapture(str(self.src))
            if not cap.isOpened(): raise RuntimeError("Could not open uploaded video.")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps_in = cap.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps_in) or fps_in <= 0: fps_in = 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if min(w, h) <= 0: raise RuntimeError("Video has invalid frame dimensions.")
            self.total_frames = total
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(self.dst), fourcc, fps_in, (w, h))
            if not out.isOpened(): raise RuntimeError("Could not create the processed video.")
            t0 = time.perf_counter(); seen = {}
            while True:
                ok, frame = cap.read()
                if not ok: break
                dets, _ms = self.manager.detect_array(frame)
                self.detections = dets
                for d in dets: seen[d["class_name"]] = seen.get(d["class_name"], 0) + 1
                self.objects += len(dets)
                out.write(self.manager.annotate(frame, dets))
                self.frames += 1
                if total: self.progress = min(99.9, round(self.frames/total*100, 1))
            if not self.frames: raise RuntimeError("The video contains no readable frames.")
            out.release(); out = None
            el = max(time.perf_counter()-t0, 0.01)
            self.fps = round(self.frames/el, 2)
            self.progress = 100.0
            self.summary = {"frames": self.frames, "objects": self.objects,
                            "fps": self.fps, "per_class": seen}
            self.status = "completed"
        except Exception as e:
            self.error = str(e)
            self.status = "failed"
        finally:
            if cap is not None: cap.release()
            if out is not None: out.release()
            self.src.unlink(missing_ok=True)
            if self.status == "failed": self.dst.unlink(missing_ok=True)

    def snapshot(self):
        return {"id": self.id, "status": self.status, "progress": self.progress,
                "frames": self.frames, "total_frames": self.total_frames,
                "objects": self.objects, "fps": self.fps, "error": self.error,
                "summary": getattr(self, "summary", None)}
