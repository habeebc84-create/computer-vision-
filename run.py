import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))
import uvicorn

if __name__ == "__main__":
    print("Real-Time Object Detection System — http://localhost:8000")
    print("First run downloads yolov8n.pt (~6 MB) into models/ on first inference.")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
