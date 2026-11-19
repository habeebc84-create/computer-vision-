import * as ort from 'onnxruntime-web/wasm'
import wasmModuleUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url'
import wasmBinaryUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url'
import { COCO_NAMES, MODEL_INPUT_SIZE, letterboxParams, rgbaToTensor, parseDetections, summarise, Detection, getLocalSettings } from './yoloPost'

import { OBJECTS365_NAMES } from './objects365'

export { COCO_NAMES, getLocalSettings }

export const ONNX_MODEL_URL =
  'https://huggingface.co/habeebc12/yolov8n-onnx/resolve/main/yolov8n.onnx'
const CACHE_NAME = 'ods-yolo-model-v2'
const models = {
  coco: { url: ONNX_MODEL_URL, labels: COCO_NAMES, size: MODEL_INPUT_SIZE },
  broad: { url: import.meta.env.BASE_URL + 'models/objects365.onnx', labels: OBJECTS365_NAMES, size: MODEL_INPUT_SIZE },
  package: { url: import.meta.env.BASE_URL + 'models/package.onnx', labels: ['package'], size: 320 },
}
export type BrowserModel = keyof typeof models
type Progress = (loaded: number, total: number) => void
const sessions = new Map<BrowserModel, Promise<ort.InferenceSession>>()

async function fetchModelBytes(model: BrowserModel, onProgress?: Progress): Promise<ArrayBuffer> {
  // Remote COCO weights use Cache Storage. Bundled models use the
  // server's HTTP caching policy so a deployment can replace the file.
  const { url } = models[model]
  let cache: Cache | undefined
  if (model === 'coco') {
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

function ensureBrowserModel(model: BrowserModel, onProgress?: Progress): Promise<ort.InferenceSession> {
  const existing = sessions.get(model)
  if (existing) return existing

  ort.env.wasm.wasmPaths = { mjs: wasmModuleUrl, wasm: wasmBinaryUrl }
  ort.env.wasm.numThreads = 1
  const session = fetchModelBytes(model, onProgress)
    .then(bytes => ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] }))
    .catch(error => {
      sessions.delete(model)
      throw error
    })
  sessions.set(model, session)
  return session
}

function letterboxToCanvas(source: CanvasImageSource, w: number, h: number, size: number) {
  const { scale, padX, padY } = letterboxParams(w, h, size)
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')!
  // Grey (114,114,114) padding — the same letterbox colour Ultralytics uses,
  // so the browser path sees the same input as the backend.
  ctx.fillStyle = '#727272'
  ctx.fillRect(0, 0, size, size)
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
): Promise<BrowserDetectResult> {
  const session = await ensureBrowserModel(model, onProgress)
  const { size, labels } = models[model]
  const { conf, iou } = getLocalSettings()
  const { canvas, scale, padX, padY } = letterboxToCanvas(source, w, h, size)
  const rgba = canvas.getContext('2d')!.getImageData(0, 0, size, size).data
  const tensor = new ort.Tensor('float32', rgbaToTensor(rgba, size), [1, 3, size, size])
  const t0 = performance.now()
  const feeds: Record<string, ort.Tensor> = {}
  feeds[session.inputNames[0]] = tensor
  const out = await session.run(feeds)
  const output = out[session.outputNames[0]]
  const inferenceMs = performance.now() - t0
  const dets = parseDetections(output.data as Float32Array, w, h, scale, padX, padY, conf, iou, labels, output.dims, model === 'broad' ? 0.9 : undefined)
  return {
    ...summarise(dets, inferenceMs),
    annotatedDataUrl: annotate(source, w, h, dets),
    source: 'browser',
  }
}

/** Convenience wrapper for an uploaded image file. */
export async function detectImageFileInBrowser(
  file: File,
  onProgress?: Progress,
  model: BrowserModel = 'coco',
): Promise<BrowserDetectResult> {
  const bmp = await createImageBitmap(file)
  try {
    return await detectInBrowser(bmp, bmp.width, bmp.height, onProgress, model)
  } finally {
    bmp.close()
  }
}
