import * as ort from 'onnxruntime-web/wasm'
import wasmModuleUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url'
import wasmBinaryUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url'
import { COCO_NAMES, MODEL_INPUT_SIZE, letterboxParams, rgbaToTensor, parseDetections, suppressDuplicates, summarise, Detection, getLocalSettings } from './yoloPost'

import { OBJECTS365_NAMES } from './objects365'
import { EXPANDED_MODEL_URL, expandedLabels, parseExpandedDetections } from './expandedYolo'

export { COCO_NAMES, getLocalSettings }

export const ONNX_MODEL_URL =
  'https://huggingface.co/habeebc12/yolov8n-onnx/resolve/main/yolov8n.onnx'
const CACHE_NAME = 'ods-yolo-model-v2'
const models = {
  coco: { url: ONNX_MODEL_URL, labels: COCO_NAMES, size: MODEL_INPUT_SIZE },
  broad: { url: import.meta.env.BASE_URL + 'models/objects365.onnx', labels: OBJECTS365_NAMES, size: MODEL_INPUT_SIZE },
  expanded: { url: EXPANDED_MODEL_URL, labels: [], size: MODEL_INPUT_SIZE },
  package: { url: import.meta.env.BASE_URL + 'models/package.onnx', labels: ['package'], size: 320 },
}
export type BrowserModel = keyof typeof models
type Progress = (loaded: number, total: number) => void
const sessions = new Map<BrowserModel, Promise<ort.InferenceSession>>()

export type ModelState = 'not-downloaded' | 'downloading' | 'ready' | 'error'
let modelState: ModelState = 'not-downloaded'
let modelError = ''

export function getBrowserModelState(): { state: ModelState; error: string } {
  return { state: modelState, error: modelError }
}

type BrowserModelListener = (s: { state: ModelState; error: string }) => void
const browserModelListeners = new Set<BrowserModelListener>()
function notifyBrowserModelState() {
  const s = getBrowserModelState()
  browserModelListeners.forEach(cb => { try { cb(s) } catch { /* listener errors must not break detection */ } })
}

/** Subscribe to browser-model state changes. Returns an unsubscribe function. */
export function subscribeBrowserModelState(cb: BrowserModelListener): () => void {
  browserModelListeners.add(cb)
  return () => { browserModelListeners.delete(cb) }
}

async function fetchModelBytes(model: BrowserModel, onProgress?: Progress): Promise<ArrayBuffer> {
  // Remote weights use Cache Storage. Bundled models use the
  // server's HTTP caching policy so a deployment can replace the file.
  const { url } = models[model]
  let cache: Cache | undefined
  if (model === 'coco' || model === 'expanded') {
    try {
      cache = await caches.open(CACHE_NAME)
      const hit = await cache.match(url)
      if (hit) return await hit.arrayBuffer()
    } catch { /* Cache Storage may be disabled */ }
  }
  const r = await fetch(url)
  if (!r.ok || !r.body || r.headers.get('content-type')?.includes('text/html')) {
    throw new Error(`Could not load the ${model} detection model (HTTP ${r.status}).`)
  }
  const total = Number(r.headers.get('content-length'))
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
    await cache?.put(url, new Response(buf))
  } catch { /* caching is best-effort */ }
  return buf.buffer
}

export function ensureBrowserModel(onProgress?: Progress, model: BrowserModel = 'coco'): Promise<ort.InferenceSession> {
  const existing = sessions.get(model)
  if (existing) return existing

  if (model === 'coco') {
    modelState = 'downloading'
    modelError = ''
    notifyBrowserModelState()
  }
  ort.env.wasm.wasmPaths = { mjs: wasmModuleUrl, wasm: wasmBinaryUrl }
  ort.env.wasm.numThreads = 1
  const session = fetchModelBytes(model, onProgress)
    .then(bytes => ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] }))
    .then(session => {
      if (model === 'coco') {
        modelState = 'ready'
        notifyBrowserModelState()
      }
      return session
    })
    .catch(error => {
      if (model === 'coco') {
        modelState = 'error'
        modelError = error instanceof Error ? error.message : String(error)
        notifyBrowserModelState()
      }
      sessions.delete(model)
      throw error
    })
  sessions.set(model, session)
  return session
}

function letterboxToCanvas(source: CanvasImageSource, w: number, h: number, size: number, x = 0, y = 0) {
  const { scale, padX, padY } = letterboxParams(w, h, size)
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')!
  // Grey (114,114,114) padding — the same letterbox colour Ultralytics uses,
  // so the browser path sees the same input as the backend.
  ctx.fillStyle = '#727272'
  ctx.fillRect(0, 0, size, size)
  ctx.drawImage(source, x, y, w, h, padX, padY, Math.round(w * scale), Math.round(h * scale))
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
  engine?: string
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
 * Run the selected detector on an image/video/canvas source.
 * Returns the same DetectResult shape as the backend, plus an annotated
 * data URL (the backend instead returns an /outputs URL).
 */
export async function detectInBrowser(
  source: CanvasImageSource,
  w: number,
  h: number,
  onProgress?: Progress,
  model: BrowserModel = 'coco',
  scanSmallObjects = false,
): Promise<BrowserDetectResult> {
  const [session, labels] = await Promise.all([
    ensureBrowserModel(onProgress, model),
    model === 'expanded' ? expandedLabels() : models[model].labels,
  ])
  const { size } = models[model]
  const { conf, iou } = getLocalSettings()
  const duplicateBoxIou = model === 'broad' ? 0.9 : undefined
  let inferenceMs = 0
  const predict = async (width: number, height: number, x = 0, y = 0) => {
    const { canvas, scale, padX, padY } = letterboxToCanvas(source, width, height, size, x, y)
    const rgba = canvas.getContext('2d')!.getImageData(0, 0, size, size).data
    const tensor = new ort.Tensor('float32', rgbaToTensor(rgba, size), [1, 3, size, size])
    const start = performance.now()
    try {
      const outputs = await session.run({ [session.inputNames[0]]: tensor })
      try {
        const output = outputs[session.outputNames[0]]
        inferenceMs += performance.now() - start
        if (model === 'expanded') {
          return parseExpandedDetections(output.data as Float32Array, output.dims, labels, width, height, scale, padX, padY, conf, iou)
        }
        return parseDetections(output.data as Float32Array, width, height, scale, padX, padY, conf, iou, labels, output.dims, duplicateBoxIou)
      } finally {
        Object.values(outputs).forEach(output => output.dispose())
      }
    } finally {
      tensor.dispose()
    }
  }
  let dets = await predict(w, h)
  if (scanSmallObjects && Math.max(w, h) > size) {
    // Overlap adjacent crops so targets near the middle of the photo are
    // visible in full in at least one crop.
    const cropWidth = Math.ceil(w * 0.6)
    const cropHeight = Math.ceil(h * 0.6)
    for (const y of new Set([0, h - cropHeight])) {
      for (const x of new Set([0, w - cropWidth])) {
        const matches = await predict(cropWidth, cropHeight, x, y)
        for (const match of matches) {
          const [left, top, right, bottom] = match.bbox
          if ((x > 0 && left <= 1) || (y > 0 && top <= 1) ||
              (x + cropWidth < w && right >= cropWidth - 1) ||
              (y + cropHeight < h && bottom >= cropHeight - 1)) continue
          dets.push({ ...match, bbox: [left + x, top + y, right + x, bottom + y] })
        }
      }
    }
    dets = suppressDuplicates(dets, iou, duplicateBoxIou)
  }
  return {
    ...summarise(dets, inferenceMs),
    annotatedDataUrl: annotate(source, w, h, dets),
    engine: ({ broad: 'YOLOE · Objects365', expanded: 'YOLOE · Expanded', package: 'Package detector', coco: 'YOLOv8n · COCO' })[model],
    source: 'browser',
  }
}

/** Convenience wrapper for an uploaded image file. */
export async function detectImageFileInBrowser(
  file: File,
  onProgress?: Progress,
  model: BrowserModel = 'coco',
  scanSmallObjects = false,
): Promise<BrowserDetectResult> {
  const bmp = await createImageBitmap(file).catch(() => {
    throw new Error("Could not read this image. Upload a valid image or export it as JPG or PNG and try again.")
  })
  try {
    return await detectInBrowser(bmp, bmp.width, bmp.height, onProgress, model, scanSmallObjects)
  } finally {
    bmp.close()
  }
}
