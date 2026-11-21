"""YOLOv8 training in a background thread with real Ultralytics callbacks.
Progress comes from results.csv / trainer state — never fabricated.
Stop: a callback at epoch end raises, aborting training cleanly."""
import csv, threading, time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

class TrainingJob:
    def __init__(self):
        self.status = "idle"          # idle|running|completed|failed|stopped
        self.config = {}
        self.epoch = 0; self.total_epochs = 0
        self.metrics = {}
        self.logs = []
        self.error = None
        self.best_path = None
        self.started_at = None; self.finished_at = None
        self._stop = threading.Event()
        self._thread = None

    def _log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self.logs.append(line); self.logs = self.logs[-500:]

    def _read_results_csv(self):
        run_dir = BASE / "runs" / "detect" / "train"
        # Ultralytics may create train, train2, ... — take the newest
        parent = BASE / "runs" / "detect"
        cands = sorted(parent.glob("train*"), key=lambda p: p.stat().st_mtime) if parent.exists() else []
        for d in reversed(cands):
            csvp = d / "results.csv"
            if csvp.exists():
                try:
                    with open(csvp) as f:
                        rows = list(csv.DictReader(f))
                    if rows:
                        r = rows[-1]
                        # results.csv epoch is 0-based; clamp to the configured total
                        self.epoch = min(int(float(r.get("epoch", 0))) + 1, self.total_epochs or 999)
                        self.metrics = {
                            "box_loss": float(r.get("train/box_loss", 0)),
                            "cls_loss": float(r.get("train/cls_loss", 0)),
                            "val_loss": float(r.get("val/box_loss", 0)),
                            "precision": float(r.get("metrics/precision(B)", 0)),
                            "recall": float(r.get("metrics/recall(B)", 0)),
                            "mAP50": float(r.get("metrics/mAP50(B)", 0)),
                            "mAP50_95": float(r.get("metrics/mAP50-95(B)", 0)),
                        }
                        self.best_path = str(d / "weights" / "best.pt") if (d/"weights"/"best.pt").exists() else self.best_path
                        return
                except Exception:
                    pass

    def start(self, config):
        if self.status == "running":
            raise RuntimeError("Training is already running.")
        data_yaml = BASE / "dataset" / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError("dataset/data.yaml not found — prepare and split a dataset in Module 1 first.")
        self.config = config; self.total_epochs = int(config.get("epochs", 10))
        self.epoch = 0; self.metrics = {}; self.error = None; self.best_path = None
        self.logs = []; self.status = "running"; self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return {"status": "running", "config": config}

    def _run(self):
        try:
            try:
                from ultralytics import YOLO
            except ImportError:
                raise RuntimeError(
                    "The 'ultralytics' package is not installed, so training cannot start. "
                    "Install the backend dependencies with: pip install -r requirements.txt"
                ) from None
            cfg = self.config
            model = YOLO(cfg.get("model", "yolov8n.pt"))
            job = self
            def _on_epoch_end(trainer):
                job._read_results_csv()
                job._log(f"Epoch {job.epoch}/{job.total_epochs} — mAP50 {job.metrics.get('mAP50', 0):.3f}")
                if job._stop.is_set():
                    raise RuntimeError("Training stopped by user.")
            model.add_callback("on_train_epoch_end", _on_epoch_end)
            self._log(f"Starting YOLOv8 training: model={cfg.get('model')} epochs={cfg.get('epochs')} imgsz={cfg.get('imgsz')} batch={cfg.get('batch')}")
            model.train(data=str(BASE/"dataset"/"data.yaml"), epochs=int(cfg.get("epochs", 10)),
                        imgsz=int(cfg.get("imgsz", 640)), batch=int(cfg.get("batch", 8)),
                        device=cfg.get("device", "cpu"), project=str(BASE/"runs"/"detect"),
                        name="train", exist_ok=True, verbose=False)
            self._read_results_csv()
            self.status = "completed" if not self._stop.is_set() else "stopped"
            self.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
            self._log("Training finished." + (f" Best weights: {self.best_path}" if self.best_path else ""))
            if self.best_path:
                import shutil
                shutil.copy2(self.best_path, BASE/"models"/"best.pt")
                self.best_path = str(BASE/"models"/"best.pt")
        except RuntimeError as e:
            if "stopped by user" in str(e):
                self.status = "stopped"; self._log("Training stopped by user.")
            else:
                self.status = "failed"; self.error = str(e); self._log(f"ERROR: {e}")
            self.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            self.status = "failed"; self.error = str(e); self._log(f"ERROR: {e}")
            self.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")

    def stop(self):
        if self.status != "running": return {"status": self.status}
        self._stop.set(); self._log("Stop requested — will halt after the current epoch.")
        return {"status": "stopping"}

    def snapshot(self):
        if self.status == "running": self._read_results_csv()
        best = None
        if self.best_path and Path(self.best_path).exists():
            p = Path(self.best_path)
            best = {"path": str(p), "size_mb": round(p.stat().st_size/1048576, 2)}
        elif (BASE/"models"/"best.pt").exists():
            p = BASE/"models"/"best.pt"
            best = {"path": str(p), "size_mb": round(p.stat().st_size/1048576, 2)}
        return {"status": self.status, "config": self.config, "epoch": min(self.epoch, self.total_epochs) if self.total_epochs else self.epoch,
                "total_epochs": self.total_epochs, "progress": min(100.0, round(self.epoch/max(self.total_epochs,1)*100,1)),
                "metrics": self.metrics, "logs": self.logs[-80:], "error": self.error,
                "best": best, "started_at": self.started_at, "finished_at": self.finished_at}

job = TrainingJob()
