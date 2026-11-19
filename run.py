import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))
import uvicorn

ROOT = Path(__file__).resolve().parent

def ensure_runtime_dirs():
    """Create every directory the backend needs, so a fresh clone runs with
    no manual folder creation. A fresh clone has none of these (they are
    git-ignored or empty, so git never tracks them) — the /dataset static
    mount in backend/app.py used to crash the server at import time with
    "RuntimeError: Directory 'dataset' does not exist" before this existed."""
    for d in (ROOT / "dataset", ROOT / "models", ROOT / "outputs" / "images",
              ROOT / "outputs" / "videos", ROOT / "outputs" / "reports",
              ROOT / "uploads", ROOT / "database", ROOT / "downloads"):
        d.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ensure_runtime_dirs()
    # Hugging Face Spaces sets PORT=7860; locally it defaults to 8000.
    port = int(os.environ.get("PORT", "8000"))
    print(f"Real-Time Object Detection System — http://localhost:{port}")
    print("First run downloads yolov8n.pt (~6 MB) into models/ on first inference.")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
