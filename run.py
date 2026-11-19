import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))
import uvicorn

if __name__ == "__main__":
    # Hugging Face Spaces sets PORT=7860; locally it defaults to 8000.
    port = int(os.environ.get("PORT", "8000"))
    print(f"Real-Time Object Detection System — http://localhost:{port}")
    print("First run downloads yolov8n.pt (~6 MB) into models/ on first inference.")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
