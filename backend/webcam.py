"""Webcam notes: live capture runs in the BROWSER (getUserMedia) — the server
never opens a camera device. The frontend grabs frames, POSTs them to
/api/detect/frame, and draws the returned boxes on a canvas overlay.
This module only documents that contract and decodes frames server-side."""
# Contract:
#   POST /api/detect/frame  multipart field "frame" (JPEG/PNG blob)
#   -> {"detections": [...], "inference_ms": float, "fps": float, "count": int}
# Frame rate is throttled client-side (~5-8 fps) so the CPU backend keeps up.
# No webcam stream is ever faked: if getUserMedia is denied, the UI shows the
# browser error and offers Image / Video detection instead.
