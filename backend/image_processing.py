"""Image + video processing helpers (real OpenCV / YOLO inference)."""
import cv2
import numpy as np

def decode_image(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image — use a valid JPG/PNG file.")
    return img

def encode_jpeg(bgr, quality=90):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok: raise RuntimeError("Failed to encode annotated image.")
    return buf.tobytes()
