"""Image + video processing helpers (real OpenCV / YOLO inference)."""
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps

def decode_image(data: bytes):
    try:
        with Image.open(BytesIO(data), formats=["JPEG", "PNG", "WEBP", "GIF", "BMP", "AVIF"]) as image:
            return cv2.cvtColor(np.asarray(ImageOps.exif_transpose(image).convert("RGB")), cv2.COLOR_RGB2BGR)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError("Could not decode image. Use a valid JPG, PNG, WebP, GIF, BMP or AVIF file.") from exc

def encode_jpeg(bgr, quality=90):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok: raise RuntimeError("Failed to encode annotated image.")
    return buf.tobytes()
