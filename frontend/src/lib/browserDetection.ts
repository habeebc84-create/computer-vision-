import * as ort from 'onnxruntime-web/wasm'
import wasmModuleUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url'
import wasmBinaryUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url'
import {
  COCO_NAMES, MODEL_INPUT_SIZE, letterboxParams, fitParams, rgbaToTensor, parseDetections,
  parseRowDetections, peakScore, suppressDuplicates, summarise, isRowOutput,
  type Detection, type Fit, type ResizeMode, getLocalSettings,
} from './yoloPost'

import { OBJECTS365_NAMES } from './objects365'
import { EXPANDED_MODEL_URL, expandedLabels } from './expandedYolo'

export { COCO_NAMES, getLocalSettings }

/** A local COCO model is shipped with the app so live detection does not
 * depend on a third-party model host being reachable from the browser. */
export const ONNX_MODEL_URL = import.meta.env.BASE_URL + 'models/yolov8n.onnx'
const ONNX_MODEL_MIRROR =
  'https://huggingface.co/salim4n/yolov8n-detect-onnx/resolve/main/yolov8n-onnx-web/yolov8n.onnx'

const CACHE_NAME = 'ods-yolo-model-v3'

export type OutputLayout = 'yolo' | 'rows'
export interface ModelSpec {
  url: string
  mirror?: string
  labels: readonly string[]
  labelsUrl?: string
  size: number
  /** 'yolo' = [1, 4+classes, anchors]; 'rows' = [1, anchors, 6+] */
  layout: OutputLayout
  resize: ResizeMode
  confFloor?: number
  engine: string
  bytes?: number
}

const models: Record<'coco' | 'broad' | 'expanded' | 'package', ModelSpec> = {
  coco: {
    url: ONNX_MODEL_URL, mirror: ONNX_MODEL_MIRROR, labels: COCO_NAMES,
    size: MODEL_INPUT_SIZE, layout: 'yolo', resize: 'letterbox', engine: 'YOLOv8n · COCO (80 classes)',
  },
  broad: {
    // Built by scripts/prepare-image-model.mjs from the pinned OpenAI/PP-YOLOE
    // Objects365 export (1x3x640x640 → 1x369x8400, 365 classes + 4 box rows).
    url: import.meta.env.BASE_URL + 'models/objects365.onnx', labels: OBJECTS365_NAMES,
    size: MODEL_INPUT_SIZE, layout: 'yolo', resize: 'letterbox', engine: 'YOLOE · Objects365 (365 classes)',
  },
  expanded: {
    url: EXPANDED_MODEL_URL, labels: [], labelsUrl: 'expanded',
    size: MODEL_INPUT_SIZE, layout: 'rows', resize: 'letterbox', engine: 'YOLOE · Expanded (4,585 classes)',
  },
  package: {
    // Fine-tuned YOLOv8n, exported at 320x320 with a *stretched* (non-letterbox)
    // resize — verified against the exported ONNX: letterboxing the same photo
    // scores 0.20-0.27 while stretching it scores 0.68-0.31, and only the
    // stretched geometry places boxes on the parcels.
    url: import.meta.env.BASE_URL + 'models/package.onnx', labels: ['package'],
    size: 320, layout: 'yolo', resize: 'squash', engine: 'Package detector (trained)',
  },
}
export type BrowserModel = keyof typeof models

export function modelLabel(model: BrowserModel) { return models[model].engine }

const availability = new Map<BrowserModel, boolean>()

/**
 * Whether this model's weights are actually reachable from the deployment.
 * Bundled models (package, objects365) are only present if the build fetched
 * them, and a missing file on a static host can answer 200 with the app's
 * index.html — both are checked here, so the UI can offer a mode that works
 * instead of failing on a dead link.
 */
export async function isModelAvailable(model: BrowserModel): Promise<boolean> {
  const known = availability.get(model)
  if (known !== undefined) return known
  const spec = models[model]
  let ok = false
  try {
    // A ranged GET is cheap even on a 12-38 MB file and works where HEAD is
    // not allowed by the host.
    const response = await fetch(spec.url, { headers: { Range: 'bytes=0-15' } })
    const type = response.headers.get('content-type') || ''
    if (response.ok && !type.includes('text/html')) {
      const head = new Uint8Array(await response.arrayBuffer())
      // ONNX protobufs never start with '<' (an HTML page) or '{'.
      ok = head.length > 0 && head[0] !== 0x3c && head[0] !== 0x7b
    }
  } catch { ok = false }
  availability.set(model, ok)
  return ok
}

/** Forget cached availability probes (used after a retry). */
export function resetModelAvailability() { availability.clear() }

type Progress = (loaded: number, total: number) => void
const sessions = new Map<BrowserModel, Promise<ort.InferenceSession>>()

export interface ModelStateInfo { state: ModelState; error: string }
export type ModelState = 'not-downloaded' | 'downloading' | 'ready' | 'error'

// Per-model state so the UI can report the model it is actually using.
const states = new Map<BrowserModel, ModelStateInfo>()
function stateKey(m: BrowserModel) { return states.get(m) ?? { state: 'not-downloaded', error: '' } }

export function getBrowserModelState(model: BrowserModel = 'coco'): ModelStateInfo {
  return { ...stateKey(model) }
}

type BrowserModelListener = (s: ModelStateInfo) => void
const browserModelListeners = new Set<{ model: BrowserModel; cb: BrowserModelListener }>()

function notifyBrowserModelState(model: BrowserModel) {
  const s = getBrowserModelState(model)
  browserModelListeners.forEach(listener => {
    if (listener.model !== model) return
    try { listener.cb(s) } catch { /* listener errors must not break detection */ }
  })
}

/** Subscribe to one browser model's state changes. Returns an unsubscribe function. */
export function subscribeBrowserModelState(cb: BrowserModelListener, model: BrowserModel = 'coco'): () => void {
  const listener = { model, cb }
  browserModelListeners.add(listener)
  cb(getBrowserModelState(model))
  return () => { browserModelListeners.delete(listener) }
}

function setState(model: BrowserModel, state: ModelState, error = '') {
  states.set(model, { state, error })
  notifyBrowserModelState(model)
}

/** Clear cached weights so the next run re-downloads (used by a retry button). */
export async function resetBrowserModel(model: BrowserModel = 'coco') {
  sessions.delete(model)
  setState(model, 'not-downloaded')
  try {
    const cache = await caches.open(CACHE_NAME)
    for (const request of await cache.keys()) {
      if (request.url.includes(resolveModelUrl(models[model]).split('/').pop() || '\u0000')) await cache.delete(request)
    }
  } catch { /* Cache Storage may be disabled */ }
}

function resolveModelUrl(spec: ModelSpec) {
  return spec.url
}

async function download(url: string, onProgress?: Progress): Promise<ArrayBuffer> {
  const response = await fetch(url, { mode: 'cors' })
  if (!response.ok || !response.body) {
    throw new Error(`HTTP ${response.status}`)
  }
  // A static host serving index.html for a missing file looks like a 200 but
  // is never a valid ONNX graph — reject it early with a useful message.
  const type = response.headers.get('content-type') || ''
  if (type.includes('text/html')) throw new Error('the server returned a web page instead of model weights')
  const total = Number(response.headers.get('content-length')) || 0
  const reader = response.body.getReader()
  const chunks: Uint8Array[] = []
  let loaded = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    loaded += value.length
    onProgress?.(loaded, total)
  }
  if (!loaded) throw new Error('the download was empty')
  const bytes = new Uint8Array(loaded)
  let offset = 0
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length }
  return bytes.buffer
}

async function fetchModelBytes(model: BrowserModel, onProgress?: Progress): Promise<ArrayBuffer> {
  const spec = models[model]
  // Bundled models (package, objects365) are served by the same deployment.
  // Remote weights go through Cache Storage so a reload is instant.
  const remote = /^https?:/i.test(spec.url)
  let cache: Cache | undefined
  if (remote) {
    try {
      cache = await caches.open(CACHE_NAME)
      const hit = await cache.match(spec.url)
      if (hit) return await hit.arrayBuffer()
    } catch { /* Cache Storage may be disabled */ }
  }
  const attempts = [spec.url, ...(spec.mirror ? [spec.mirror] : [])]
  const problems: string[] = []
  for (const url of attempts) {
    try {
      const bytes = await download(url, onProgress)
      try { await cache?.put(spec.url, new Response(bytes.slice(0))) } catch { /* caching is best-effort */ }
      return bytes
    } catch (error) {
      const host = remote ? new URL(url).host : url.replace(/^\//, '')
      problems.push(`${host}: ${error instanceof Error ? error.message : String(error)}`)
    }
  }
  // A missing *bundled* file means the deploy never shipped these weights (the
  // build-time download is best-effort). Say that plainly and point at a mode
  // that does work, instead of a bare HTTP failure.
  const detail = problems.join('; ')
  if (!remote) {
    throw new Error(
      `The ${spec.engine} weights are not available on this deployment (${detail}). ` +
      'Use "COCO classes" or "Search by name" instead, or redeploy so the build can fetch the model.',
    )
  }
  throw new Error(`Could not download the ${spec.engine} weights (${detail}).`)
}

export function ensureBrowserModel(onProgress?: Progress, model: BrowserModel = 'coco'): Promise<ort.InferenceSession> {
  const existing = sessions.get(model)
  if (existing) return existing

  setState(model, 'downloading')
  ort.env.wasm.wasmPaths = { mjs: wasmModuleUrl, wasm: wasmBinaryUrl }
  // Single-threaded WASM works on every host. Multi-threaded WASM needs
  // cross-origin isolation (COOP/COEP headers) which a plain static Vercel
  // deployment does not send — asking for more threads there throws at init.
  ort.env.wasm.numThreads = 1
  ort.env.wasm.simd = true
  ort.env.logLevel = 'error'

  const session = fetchModelBytes(model, onProgress)
    .then(bytes => ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' }))
    .then(result => {
      setState(model, 'ready')
      return result
    })
    .catch(error => {
      setState(model, 'error', error instanceof Error ? error.message : String(error))
      sessions.delete(model)
      throw error
    })
  sessions.set(model, session)
  return session
}

/**
 * ONNX Runtime tells us the input size it wants when a tensor has the wrong
 * shape ("... index: 2 Got: 640 Expected: 320"). Use that to correct the
 * model's declared size once, instead of assuming every export is 640×640 —
 * a hard-coded size makes every box land in the wrong place.
 */
function resizeFromError(spec: ModelSpec, message: string): boolean {
  const expected = /Expected:\s*(\d+)/.exec(message)
  if (!expected) return false
  const size = Number(expected[1])
  if (!Number.isFinite(size) || size < 32 || size > 2048 || size === spec.size) return false
  spec.size = size
  return true
}

/** Draw `source` into a size×size canvas using the model's resize mode. */
function prepareInput(
  source: CanvasImageSource, w: number, h: number, size: number, resize: ResizeMode, sx = 0, sy = 0,
): { canvas: HTMLCanvasElement; fit: Fit } {
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'
  if (resize === 'squash') {
    // Stretch the whole frame — this is what the package model was exported
    // with, and it is the only resize that scores those photos correctly.
    ctx.drawImage(source, sx, sy, w, h, 0, 0, size, size)
  } else {
    const { scale, padX, padY } = letterboxParams(w, h, size)
    // Grey (114,114,114) padding — the same letterbox colour Ultralytics uses,
    // so the browser path sees the same input as the backend.
    ctx.fillStyle = '#727272'
    ctx.fillRect(0, 0, size, size)
    ctx.drawImage(source, sx, sy, w, h, padX, padY, Math.max(1, Math.round(w * scale)), Math.max(1, Math.round(h * scale)))
  }
  return { canvas, fit: fitParams(w, h, size, resize) }
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
  /** Highest candidate score seen, even if below the threshold. */
  peak_confidence?: number
  /** Threshold that was applied. */
  conf_threshold?: number
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

/** Decode model output into detections using the spec's layout. */
function decode(
  spec: ModelSpec,
  output: ort.Tensor,
  labels: readonly string[],
  w: number,
  h: number,
  fit: Fit,
  conf: number,
  iou: number,
): { detections: Detection[]; peak: number } {
  const data = output.data as Float32Array
  const dims = output.dims as number[]
  // Three export layouts exist in the wild, so decide from the tensor itself
  // rather than trusting a hard-coded assumption — decoding along the wrong
  // axis produces boxes at [0,0,…] or scores far above 1.0.
  //   A. [1, 4+classes, anchors]  Ultralytics YOLOv8/YOLOE detection export
  //   B. [1, anchors, 4+classes]  the same tensor transposed
  //   C. [1, anchors, 6+]         NMS-free rows: x1,y1,x2,y2,score,classId,…
  const rowShape = isRowOutput(dims, labels.length)
  if (spec.layout === 'rows' || rowShape) {
    return { detections: parseRowDetections(data, dims, labels, w, h, fit, conf, iou, 0.9), peak: 0 }
  }
  return {
    detections: parseDetections(data, w, h, fit, conf, iou, labels, dims),
    peak: peakScore(data, dims, labels.length),
  }
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
  const spec = models[model]
  const [session, labels] = await Promise.all([
    ensureBrowserModel(onProgress, model),
    model === 'expanded' ? expandedLabels() : Promise.resolve(spec.labels),
  ])
  const { conf: configuredConf, iou } = getLocalSettings()
  const conf = Math.max(spec.confFloor ?? 0, configuredConf)
  let inferenceMs = 0
  let peak = 0
  const sessionInput = session.inputNames[0]
  const sessionOutput = session.outputNames[0]

  /**
   * Run the model on the source rectangle (x, y, cw, ch) and return boxes in
   * original-image pixels. Boxes are decoded in the crop's own coordinate
   * space and then shifted, so a crop never needs different maths.
   */
  const predict = async (cw: number, ch: number, x = 0, y = 0, retried = false): Promise<Detection[]> => {
    const { canvas, fit } = prepareInput(source, cw, ch, spec.size, spec.resize, x, y)
    const rgba = canvas.getContext('2d')!.getImageData(0, 0, spec.size, spec.size).data
    const tensor = new ort.Tensor('float32', rgbaToTensor(rgba, spec.size), [1, 3, spec.size, spec.size])
    const start = performance.now()
    let outputs: Record<string, ort.Tensor> | undefined
    try {
      outputs = await session.run({ [sessionInput]: tensor })
    } catch (error) {
      // A wrong input resolution is recoverable: learn the real size from ORT
      // and run once more instead of failing the whole detection.
      if (!retried && error instanceof Error && resizeFromError(spec, error.message)) {
        return predict(cw, ch, x, y, true)
      }
      throw error
    } finally {
      tensor.dispose()
      // Inference timing excludes the canvas/tensor setup above.
      inferenceMs += performance.now() - start
    }
    try {
      const { detections: found, peak: framePeak } = decode(spec, outputs[sessionOutput], labels, cw, ch, fit, conf, iou)
      peak = Math.max(peak, framePeak)
      return x === 0 && y === 0 ? found : found.map(d => {
        const [l, t, r, b] = d.bbox
        return { ...d, bbox: [round1(l + x), round1(t + y), round1(r + x), round1(b + y)] }
      })
    } finally {
      Object.values(outputs).forEach(o => o.dispose())
    }
  }

  let dets = await predict(w, h)

  // Upscale-and-scan, for targets that are tiny in a large photo. The photo
  // is split into a few overlapping tiles and each tile is scaled *up* to the
  // model input, so a 40 px parcel becomes 200+ px and crosses the confidence
  // threshold. Every tile uses real pixels — nothing is invented.
  if (scanSmallObjects && Math.max(w, h) > spec.size) {
    const tileWidth = Math.max(1, Math.ceil(w * 0.6))
    const tileHeight = Math.max(1, Math.ceil(h * 0.6))
    const xs = [...new Set([0, Math.max(0, w - tileWidth)])]
    const ys = [...new Set([0, Math.max(0, h - tileHeight)])]
    for (const y of ys) {
      for (const x of xs) {
        if (x === 0 && y === 0) continue // already covered by the full-frame pass
        const matches = await predict(tileWidth, tileHeight, x, y)
        for (const match of matches) {
          const [left, top, right, bottom] = match.bbox
          // Boxes whose centre falls inside a neighbouring tile are kept by
          // that tile; keeping them twice only doubles the box count.
          const cx = (left + right) / 2
          const cy = (top + bottom) / 2
          if (cx < 0 || cy < 0 || cx > w || cy > h) continue
          dets.push(match)
        }
      }
    }
    dets = suppressDuplicates(dets, iou)
  }

  return {
    ...summarise(dets, inferenceMs),
    annotatedDataUrl: annotate(source, w, h, dets),
    engine: spec.engine,
    source: 'browser',
    peak_confidence: Math.round(peak * 10000) / 10000,
    conf_threshold: conf,
  }
}

function round1(v: number) { return Math.round(v * 10) / 10 }

/** Convenience wrapper for an uploaded image file. */
export async function detectImageFileInBrowser(
  file: File,
  onProgress?: Progress,
  model: BrowserModel = 'coco',
  scanSmallObjects = false,
): Promise<BrowserDetectResult> {
  const bmp = await createImageBitmap(file).catch(() => {
    throw new Error('Could not read this image. Upload a valid image or export it as JPG or PNG and try again.')
  })
  try {
    return await detectInBrowser(bmp, bmp.width, bmp.height, onProgress, model, scanSmallObjects)
  } finally {
    bmp.close()
  }
}
