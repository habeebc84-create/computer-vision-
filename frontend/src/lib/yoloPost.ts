// Pure YOLO post-processing — no browser or ONNX imports, so the same code
// can be unit-tested in Node and shipped to the browser.
//
// Every decoder here maps exported rows/anchors back to *original image*
// pixel coordinates. A model whose output layout does not match the labels it
// is paired with is rejected with a clear message instead of silently
// returning zero detections — that silent mismatch was the "nothing is
// detected" bug: a wrong layout scored every box below the confidence
// threshold, so the UI reported "No objects detected" with no error at all.

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

/**
 * How a source image was fitted into the model's square input.
 *  - scale:  pixels-per-source-pixel used for both axes
 *  - padX/Y: grey/black padding added on each axis (letterbox)
 *  - scaleX/scaleY: per-axis scale (non-uniform squash, e.g. a 320 model
 *    trained on stretched frames)
 */
export interface Fit {
  scaleX: number
  scaleY: number
  scale: number
  padX: number
  padY: number
}

/** Letterbox scale/padding for fitting a w×h image into a size×size square. */
export function letterboxParams(w: number, h: number, size = MODEL_INPUT_SIZE): Fit {
  const scale = Math.min(size / w, size / h)
  const nw = Math.max(1, Math.round(w * scale))
  const nh = Math.max(1, Math.round(h * scale))
  const padX = Math.floor((size - nw) / 2)
  const padY = Math.floor((size - nh) / 2)
  return { scale, scaleX: scale, scaleY: scale, padX, padY }
}

/** Stretch scale for fitting a w×h image into a size×size square (no padding). */
export function squashParams(w: number, h: number, size = MODEL_INPUT_SIZE): Fit {
  return { scaleX: size / w, scaleY: size / h, scale: Math.min(size / w, size / h), padX: 0, padY: 0 }
}

export type ResizeMode = 'letterbox' | 'squash'

export function fitParams(w: number, h: number, size: number, mode: ResizeMode): Fit {
  return mode === 'squash' ? squashParams(w, h, size) : letterboxParams(w, h, size)
}

/** Map a box expressed in model-input pixels back to original-image pixels. */
export function unmapBox(
  x1: number, y1: number, x2: number, y2: number,
  fit: Fit, origW: number, origH: number,
): number[] | null {
  const left = (x1 - fit.padX) / fit.scaleX
  const top = (y1 - fit.padY) / fit.scaleY
  const right = (x2 - fit.padX) / fit.scaleX
  const bottom = (y2 - fit.padY) / fit.scaleY
  const cx1 = Math.max(0, Math.min(origW, left))
  const cy1 = Math.max(0, Math.min(origH, top))
  const cx2 = Math.max(0, Math.min(origW, right))
  const cy2 = Math.max(0, Math.min(origH, bottom))
  if (!(cx2 > cx1 && cy2 > cy1)) return null
  return [round1(cx1), round1(cy1), round1(cx2), round1(cy2)]
}

function round1(v: number) { return Math.round(v * 10) / 10 }

/**
 * Highest class score in a YOLO tensor, ignoring geometry. Used to explain a
 * zero-result run honestly: "the best candidate scored 0.11, your threshold is
 * 0.25" is actionable, while "no objects detected" is not.
 */
export function peakScore(output: Float32Array, dimensions: readonly number[], classCount: number): number {
  if (dimensions.length !== 3) return 0
  const [, c1, c2] = dimensions
  const rows = c1 === 4 + classCount ? c2 : c1
  const stride = c1 === 4 + classCount ? c1 : c2
  const transposed = c1 !== 4 + classCount
  let peak = 0
  for (let j = 0; j < rows; j++) {
    for (let k = 4; k < Math.min(4 + classCount, stride); k++) {
      const v = transposed ? output[j * stride + k] : output[k * rows + j]
      if (v > peak && Number.isFinite(v)) peak = v
    }
  }
  return peak
}

/**
 * Convert an already-letterboxed size×size RGBA buffer (e.g. from a canvas)
 * into an NCHW float32 tensor, values in [0,1].
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

/**
 * Decode a transposed Ultralytics YOLO detection tensor of shape
 * [1, 4 + classes, anchors] (centre-x, centre-y, width, height in model-input
 * pixels, then one score column per class) and undo the letterbox transform.
 *
 * Returns an empty array when nothing passes the threshold — that is a real
 * result. Throws only when the tensor cannot possibly match these labels.
 */
export function parseDetections(
  output: Float32Array,
  origW: number,
  origH: number,
  fit: Fit,
  confThreshold: number,
  iouThreshold: number,
  classNames: readonly string[],
  dimensions: readonly number[],
  duplicateBoxIou?: number,
): Detection[] {
  if (dimensions.length !== 3) {
    throw new Error(`The detection model returned ${dimensions.length} output dimensions; a YOLO box tensor has 3.`)
  }
  const [, c1, c2] = dimensions
  const expected = 4 + classNames.length
  // Ultralytics exports are [1, 4 + classes, anchors]. Accept the transposed
  // [1, anchors, 4 + classes] layout too (older/some custom exports) so a
  // model is never silently paired with the wrong axis.
  let transposed: boolean
  if (c1 === expected && output.length === c1 * c2) transposed = false
  else if (c2 === expected && output.length === c1 * c2) transposed = true
  else {
    throw new Error(
      `The detection model output [${dimensions.join(', ')}] does not match its ${classNames.length} class labels ` +
      `(expected [1, ${expected}, anchors]). Check that the model file and label list belong together.`,
    )
  }
  const rows = transposed ? c1 : c2
  const stride = transposed ? c2 : c1
  // Row-major [1, channels, anchors]: channel `k` of anchor `j` lives at
  // (k * anchors + j) — i.e. at(k, j). The transposed layout is
  // [1, anchors, channels]: at(k, j) = output[j * stride + k].
  const at = transposed
    ? (row: number, col: number) => output[col * stride + row]
    : (row: number, col: number) => output[row * rows + col]

  const cands: Detection[] = []
  for (let j = 0; j < rows; j++) {
    let bestClass = -1
    let bestScore = 0
    for (let k = 0; k < classNames.length; k++) {
      const s = at(4 + k, j)
      if (s > bestScore) { bestScore = s; bestClass = k }
    }
    if (bestClass < 0 || !Number.isFinite(bestScore) || bestScore < confThreshold) continue
    const cx = at(0, j), cy = at(1, j), w = at(2, j), h = at(3, j)
    if (![cx, cy, w, h].every(Number.isFinite) || w <= 0 || h <= 0) continue
    const bbox = unmapBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, fit, origW, origH)
    if (!bbox) continue
    cands.push({ class_id: bestClass, class_name: classNames[bestClass], confidence: bestScore, bbox })
  }
  return suppressDuplicates(cands, iouThreshold, duplicateBoxIou)
}

/**
 * Decode an NMS-free exported tensor where each row is
 * [x1, y1, x2, y2, confidence, classId, ...extra] in model-input pixels —
 * the layout used by the YOLOE prompt-free / Objects365 exports.
 */
export function parseRowDetections(
  output: Float32Array,
  dimensions: readonly number[],
  classNames: readonly string[],
  origW: number,
  origH: number,
  fit: Fit,
  confThreshold: number,
  iouThreshold: number,
  duplicateBoxIou?: number,
): Detection[] {
  if (dimensions.length !== 3) {
    throw new Error(`The detection model returned ${dimensions.length} output dimensions; expected 3.`)
  }
  const [, rows, stride] = dimensions
  if (stride < 6 || output.length !== rows * stride) {
    throw new Error(
      `The detection model output [${dimensions.join(', ')}] does not contain rows of boxes ` +
      `(expected [1, anchors, 6+]); check that this model file matches its labels.`,
    )
  }
  const detections: Detection[] = []
  for (let i = 0; i < rows; i++) {
    const o = i * stride
    const confidence = output[o + 4]
    const classId = output[o + 5]
    if (!Number.isFinite(confidence) || confidence < confThreshold) continue
    if (!Number.isInteger(classId) || classId < 0 || classId >= classNames.length) continue
    const bbox = unmapBox(output[o], output[o + 1], output[o + 2], output[o + 3], fit, origW, origH)
    if (!bbox) continue
    detections.push({ class_id: classId, class_name: classNames[classId], confidence, bbox })
  }
  return suppressDuplicates(detections, iouThreshold, duplicateBoxIou)
}

export function suppressDuplicates(candidates: Detection[], iouThreshold: number, duplicateBoxIou?: number): Detection[] {
  // Class-aware NMS, highest confidence first.
  const sorted = [...candidates].sort((a, b) => b.confidence - a.confidence)
  const kept: Detection[] = []
  for (const d of sorted) {
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
