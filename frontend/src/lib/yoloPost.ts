// Pure YOLOv8 post-processing — no browser or ONNX imports, so the same code
// can be tested in Node (see /tmp/ods-onnx-test) and run in the browser.
// Verified against the 80 COCO class names the backend reports (ultralytics).

export const COCO_NAMES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
  'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
  'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
  'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
  'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
  'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
  'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
  'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
  'toothbrush',
]

export const MODEL_INPUT_SIZE = 640

// Detection thresholds, shared by backend and in-browser inference.
// Stored in localStorage so the browser fallback can use them when the
// backend (the normal home of Settings) is unreachable.
export interface LocalSettings { conf: number; iou: number; imgsz: number }
const DEFAULT_SETTINGS: LocalSettings = { conf: 0.25, iou: 0.45, imgsz: 640 }

export function getLocalSettings(): LocalSettings {
  try {
    const raw = localStorage.getItem('ods-settings')
    if (!raw) return DEFAULT_SETTINGS
    const s = JSON.parse(raw)
    return {
      conf: typeof s.conf === 'number' ? s.conf : DEFAULT_SETTINGS.conf,
      iou: typeof s.iou === 'number' ? s.iou : DEFAULT_SETTINGS.iou,
      imgsz: typeof s.imgsz === 'number' ? s.imgsz : DEFAULT_SETTINGS.imgsz,
    }
  } catch { return DEFAULT_SETTINGS }
}

export function saveLocalSettings(s: Partial<LocalSettings>) {
  try { localStorage.setItem('ods-settings', JSON.stringify({ ...getLocalSettings(), ...s })) } catch { /* private mode */ }
}

export interface Detection {
  class_id: number
  class_name: string
  confidence: number
  bbox: number[] // [x1, y1, x2, y2] in original-image pixels
}

/** Letterbox scale/padding for fitting a w×h image into a size×size square. */
export function letterboxParams(w: number, h: number, size = MODEL_INPUT_SIZE) {
  const scale = Math.min(size / w, size / h)
  const nw = Math.round(w * scale)
  const nh = Math.round(h * scale)
  return { scale, padX: Math.floor((size - nw) / 2), padY: Math.floor((size - nh) / 2) }
}

/**
 * Convert an already-letterboxed size×size RGBA buffer (e.g. from a canvas
 * or from PIL in the Node test) into an NCHW float32 tensor, values in [0,1].
 */
export function rgbaToTensor(rgba: Uint8ClampedArray | Uint8Array, size = MODEL_INPUT_SIZE): Float32Array {
  const n = size * size
  const out = new Float32Array(3 * n)
  for (let i = 0; i < n; i++) {
    out[i] = rgba[i * 4] / 255
    out[n + i] = rgba[i * 4 + 1] / 255
    out[2 * n + i] = rgba[i * 4 + 2] / 255
  }
  return out
}

function iou(a: number[], b: number[]): number {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1])
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3])
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  if (inter <= 0) return 0
  const areaA = (a[2] - a[0]) * (a[3] - a[1])
  const areaB = (b[2] - b[0]) * (b[3] - b[1])
  return inter / (areaA + areaB - inter)
}

/** Decode [batch, 4 + classes, anchors] YOLO outputs and undo the letterbox transform. */
export function parseDetections(
  output: Float32Array,
  origW: number,
  origH: number,
  scale: number,
  padX: number,
  padY: number,
  confThreshold: number,
  iouThreshold: number,
  classNames: readonly string[],
  dimensions: readonly number[],
  duplicateBoxIou?: number,
): Detection[] {
  const [batch, channels, N] = dimensions
  if (dimensions.length !== 3 || batch !== 1 || channels !== 4 + classNames.length || output.length !== channels * N) {
    throw new Error('The detection model output does not match its class labels.')
  }
  const cands: Detection[] = []
  for (let j = 0; j < N; j++) {
    let bestClass = -1
    let bestScore = 0
    for (let k = 0; k < classNames.length; k++) {
      const s = output[(4 + k) * N + j]
      if (s > bestScore) { bestScore = s; bestClass = k }
    }
    if (bestClass < 0 || bestScore < confThreshold) continue
    const cx = output[0 * N + j], cy = output[1 * N + j]
    const w = output[2 * N + j], h = output[3 * N + j]
    // 640-space box -> original pixels (undo letterbox).
    const x1 = Math.max(0, (cx - w / 2 - padX) / scale)
    const y1 = Math.max(0, (cy - h / 2 - padY) / scale)
    const x2 = Math.min(origW, (cx + w / 2 - padX) / scale)
    const y2 = Math.min(origH, (cy + h / 2 - padY) / scale)
    if (x2 <= x1 || y2 <= y1) continue
    cands.push({
      class_id: bestClass,
      class_name: classNames[bestClass],
      confidence: bestScore,
      bbox: [Math.round(x1 * 10) / 10, Math.round(y1 * 10) / 10, Math.round(x2 * 10) / 10, Math.round(y2 * 10) / 10],
    })
  }
  // Class-aware NMS, highest confidence first.
  cands.sort((a, b) => b.confidence - a.confidence)
  const kept: Detection[] = []
  for (const d of cands) {
    // Broad vocabularies can assign near-identical boxes competing labels
    // such as sneaker and boot. Keep the highest confidence label for those.
    const duplicate = kept.some(k => {
      const threshold = k.class_id === d.class_id ? iouThreshold : duplicateBoxIou
      return threshold !== undefined && iou(k.bbox, d.bbox) > threshold
    })
    if (duplicate) continue
    kept.push(d)
  }
  return kept
}

/** Summarise detections into the same shape the backend returns. */
export function summarise(dets: Detection[], inferenceMs: number) {
  return {
    detections: dets,
    count: dets.length,
    unique_classes: [...new Set(dets.map(d => d.class_name))].sort(),
    avg_confidence: dets.length ? Math.round((dets.reduce((s, d) => s + d.confidence, 0) / dets.length) * 10000) / 10000 : 0,
    inference_ms: Math.round(inferenceMs * 10) / 10,
    fps: Math.round((1000 / Math.max(inferenceMs, 0.01)) * 10) / 10,
  }
}
