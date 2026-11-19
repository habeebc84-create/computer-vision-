// In-browser YOLOv8 detection (ONNX Runtime Web) — used ONLY when the Python
// backend is unreachable. Real YOLOv8 inference running in this browser, not a
// simulation: the yolov8n.onnx weights are downloaded once from Habeeb's own
// Hugging Face model repo (habeebc12/yolov8n-onnx, which serves the file with
// CORS enabled — GitHub release downloads do not, and browsers block them)
// and cached in Cache Storage. COCO 80 classes only —
// open-vocabulary (YOLO-World) stays backend-only.
import * as ort from 'onnxruntime-web'
import { COCO_NAMES, MODEL_INPUT_SIZE, letterboxParams, rgbaToTensor, parseDetections, summarise, Detection, getLocalSettings } from './yoloPost'

export { COCO_NAMES, getLocalSettings }

export const ONNX_MODEL_URL =
  'https://huggingface.co/habeebc12/yolov8n-onnx/resolve/main/yolov8n.onnx'
const ONNX_MODEL_SIZE_MB = 12.3
const CACHE_NAME = 'ods-yolo-model-v2'
const ORT_VERSION = '1.20.1'

export type ModelState = 'not-downloaded' | 'downloading' | 'ready' | 'error'
let sessionPromise: Promise<ort.InferenceSession> | null = null
let modelState: ModelState = 'not-downloaded'
let modelError = ''

export function getBrowserModelState(): { state: ModelState; error: string } {
  return { state: modelState, error: modelError }
}

async function fetchModelBytes(onProgress?: (loaded: number, total: number) => void): Promise<ArrayBuffer> {
  // Cache Storage first — the ~12 MB model downloads once, then loads offline.
  try {
    const cache = await caches.open(CACHE_NAME)
    const hit = await cache.match(ONNX_MODEL_URL)
    if (hit) return await hit.arrayBuffer()
  } catch { /* Cache Storage unavailable — fall through to a plain fetch */ }
  const r = await fetch(ONNX_MODEL_URL)
  if (!r.ok || !r.body) throw new Error(`Could not download the detection model (HTTP ${r.status}). Check your internet connection and try again.`)
  const total = Number(r.headers.get('content-length') || ONNX_MODEL_SIZE_MB * 1024 * 1024)
  const reader = r.body.getReader()
  const chunks: Uint8Array[] = []
  let loaded = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    loaded += value.length
    onProgress?.(loaded, total)
  }
  const buf = new Uint8Array(loaded)
  let off = 0
  for (const c of chunks) { buf.set(c, off); off += c.length }
  try {
    const cache = await caches.open(CACHE_NAME)
    await cache.put(ONNX_MODEL_URL, new Response(buf.slice().buffer, { headers: { 'content-length': String(loaded) } }))
  } catch { /* caching is best-effort */ }
  return buf.buffer
}

/** Load (downloading + caching on first use) the ONNX session. Shared by all callers. */
export function ensureBrowserModel(onProgress?: (loaded: number, total: number) => void): Promise<ort.InferenceSession> {
  if (!sessionPromise) {
    modelState = 'downloading'
    // Single-threaded wasm: no SharedArrayBuffer / COOP-COEP requirements,
    // works on plain HTTPS pages such as Vercel.
    ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`
    ort.env.wasm.numThreads = 1
    sessionPromise = fetchModelBytes(onProgress)
      .then(bytes => ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] }))
      .then(s => { modelState = 'ready'; return s })
      .catch(e => {
        modelState = 'error'
        modelError = e?.message || String(e)
        sessionPromise = null
        throw e
      })
  }
  return sessionPromise
}

function letterboxToCanvas(source: CanvasImageSource, w: number, h: number): { canvas: HTMLCanvasElement; scale: number; padX: number; padY: number } {
  const { scale, padX, padY } = letterboxParams(w, h)
  const canvas = document.createElement('canvas')
  canvas.width = MODEL_INPUT_SIZE
  canvas.height = MODEL_INPUT_SIZE
  const ctx = canvas.getContext('2d')!
  // Grey (114,114,114) padding — the same letterbox colour Ultralytics uses,
  // so the browser path sees the same input as the backend.
  ctx.fillStyle = '#727272'
  ctx.fillRect(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
  ctx.drawImage(source, padX, padY, Math.round(w * scale), Math.round(h * scale))
  return { canvas, scale, padX, padY }
}

export interface BrowserDetectResult {
  detections: Detection[]
  count: number
  unique_classes: string[]
  avg_confidence: number
  inference_ms: number
  fps: number
  annotatedDataUrl: string
  source: 'browser'
}

/** Draw boxes + labels over the original image and return a JPEG data URL. */
export function annotate(source: CanvasImageSource, w: number, h: number, dets: Detection[]): string {
  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')!
  ctx.drawImage(source, 0, 0, w, h)
  const colors = ['#38bdf8', '#22d3ee', '#a78bfa', '#34d399', '#fb923c', '#f472b6', '#facc15']
  for (const d of dets) {
    const [x1, y1, x2, y2] = d.bbox
    const color = colors[d.class_id % colors.length]
    ctx.strokeStyle = color
    ctx.lineWidth = Math.max(2, w / 400)
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
    const label = `${d.class_name} ${(d.confidence * 100).toFixed(0)}%`
    ctx.font = `600 ${Math.max(13, Math.round(w / 55))}px Inter, sans-serif`
    const tw = ctx.measureText(label).width
    const fh = Math.max(20, Math.round(w / 40))
    ctx.fillStyle = color
    ctx.fillRect(x1, Math.max(0, y1 - fh), tw + 10, fh)
    ctx.fillStyle = '#050914'
    ctx.fillText(label, x1 + 5, Math.max(fh - 6, y1 - 6))
  }
  return canvas.toDataURL('image/jpeg', 0.9)
}

/**
 * Run real YOLOv8 inference in this browser on an image/video/canvas source.
 * Returns the same DetectResult shape as the backend, plus an annotated
 * data URL (the backend instead returns an /outputs URL).
 */
export async function detectInBrowser(
  source: CanvasImageSource,
  w: number,
  h: number,
  onProgress?: (loaded: number, total: number) => void,
): Promise<BrowserDetectResult> {
  const session = await ensureBrowserModel(onProgress)
  const { conf, iou } = getLocalSettings()
  const { canvas, scale, padX, padY } = letterboxToCanvas(source, w, h)
  const rgba = canvas.getContext('2d')!.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE).data
  const tensor = new ort.Tensor('float32', rgbaToTensor(rgba), [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE])
  const t0 = performance.now()
  const feeds: Record<string, ort.Tensor> = {}
  feeds[session.inputNames[0]] = tensor
  const out = await session.run(feeds)
  const output = out[session.outputNames[0]].data as Float32Array
  const inferenceMs = performance.now() - t0
  const dets = parseDetections(output, w, h, scale, padX, padY, conf, iou)
  return {
    ...summarise(dets, inferenceMs),
    annotatedDataUrl: annotate(source, w, h, dets),
    source: 'browser',
  }
}

/** Convenience wrapper for an uploaded image file. */
export async function detectImageFileInBrowser(
  file: File,
  onProgress?: (loaded: number, total: number) => void,
): Promise<BrowserDetectResult> {
  const bmp = await createImageBitmap(file)
  try {
    return await detectInBrowser(bmp, bmp.width, bmp.height, onProgress)
  } finally {
    bmp.close()
  }
}
