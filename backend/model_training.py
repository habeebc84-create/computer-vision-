"""YOLOv8 training in a background thread with real Ultralytics callbacks.
Progress comes from results.csv / trainer state — never fabricated.
Stop: a callback at epoch end raises, aborting training cleanly."""
import csv, shutil, threading, time
from uuid import uuid4
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

    def _read_results_csv(self, run_dir):
        with (run_dir / "results.csv").open() as f:
            rows = list(csv.DictReader(f, skipinitialspace=True))
        if rows:
            row = rows[-1]
            self.epoch = min(int(float(row["epoch"])), self.total_epochs)
            self.metrics = {
                key: float(row[column]) for key, column in (
                    ("box_loss", "train/box_loss"), ("cls_loss", "train/cls_loss"),
                    ("val_loss", "val/box_loss"), ("precision", "metrics/precision(B)"),
                    ("recall", "metrics/recall(B)"), ("mAP50", "metrics/mAP50(B)"),
                    ("mAP50_95", "metrics/mAP50-95(B)"),
                )
            }

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
            weights = cfg.get("model", "yolov8n.pt")
            local_weights = BASE / "models" / weights
            model = YOLO(str(local_weights) if local_weights.is_file() else weights)
            job = self
            def _on_epoch_end(trainer):
                if not (trainer.save_dir / "results.csv").is_file():
                    return
                job._read_results_csv(trainer.save_dir)
                job._log(f"Epoch {job.epoch}/{job.total_epochs} — mAP50 {job.metrics.get('mAP50', 0):.3f}")
                if job._stop.is_set():
                    raise RuntimeError("Training stopped by user.")
            model.add_callback("on_fit_epoch_end", _on_epoch_end)
            self._log(f"Starting YOLOv8 training: model={cfg.get('model')} epochs={cfg.get('epochs')} imgsz={cfg.get('imgsz')} batch={cfg.get('batch')}")
            model.train(data=str(BASE/"dataset"/"data.yaml"), epochs=int(cfg.get("epochs", 10)),
                        imgsz=int(cfg.get("imgsz", 640)), batch=int(cfg.get("batch", 8)),
                        device=cfg.get("device", "cpu"), project=str(BASE/"runs"/"detect"),
                        name=f"train-{uuid4().hex}", verbose=False)
            run_dir = Path(model.trainer.save_dir)
            self._read_results_csv(run_dir)
            best = BASE / "models" / "best.pt"
            best.parent.mkdir(exist_ok=True)
            pending = best.with_suffix(".pt.tmp")
            shutil.copy2(run_dir / "weights" / "best.pt", pending)
            pending.replace(best)
            self.best_path = str(best)
            self.status = "completed" if not self._stop.is_set() else "stopped"
            self.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
            self._log("Training finished." + (f" Best weights: {self.best_path}" if self.best_path else ""))
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
