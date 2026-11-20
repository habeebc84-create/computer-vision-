// In-browser open-vocabulary detection (Grounding DINO tiny, q8) via
// @huggingface/transformers — used ONLY when the Python backend is
// unreachable AND the user picks "Any object — type its name".
// Real open-vocabulary inference running in this browser, not a simulation:
// the model weights (~204 MB, q8) download once from Hugging Face
// (onnx-community/grounding-dino-tiny-ONNX, CORS-enabled) and are cached by
// Transformers.js in Cache Storage. COCO mode keeps using the fast yolov8n
// ONNX path in browserDetection.ts — this file is only loaded on demand.
import { Detection, getLocalSettings, summarise } from './yoloPost'
import { annotate, BrowserDetectResult } from './browserDetection'

export const OPEN_VOCAB_MODEL_ID = 'onnx-community/grounding-dino-tiny-ONNX'
// q8 weights (model_quantized.onnx) ~204 MB + tokenizer/preprocessor ~1 MB.
export const OPEN_VOCAB_MODEL_SIZE_MB = 205

export type OpenVocabModelState = 'not-downloaded' | 'downloading' | 'ready' | 'error'
let pipelinePromise: Promise<any> | null = null
let modelState: OpenVocabModelState = 'not-downloaded'
let modelError = ''

export function getOpenVocabModelState(): { state: OpenVocabModelState; error: string } {
  return { state: modelState, error: modelError }
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

/** Load the zero-shot pipeline once. Progress is aggregated across files. */
export function ensureOpenVocabModel(onProgress?: (loaded: number, total: number) => void): Promise<any> {
  if (!pipelinePromise) {
    modelState = 'downloading'
    pipelinePromise = (async () => {
      const { pipeline } = await import('@huggingface/transformers')
      // Aggregate per-file progress into one overall percent.
      const seen = new Map<string, { loaded: number; total: number }>()
      const report = () => {
        let loaded = 0, total = 0
        for (const v of seen.values()) { loaded += v.loaded; total += v.total }
        if (total > 0) onProgress?.(loaded, total)
      }
      const detector = await pipeline('zero-shot-object-detection', OPEN_VOCAB_MODEL_ID, {
        dtype: 'q8',
        progress_callback: (p: any) => {
          if (p.status === 'download' || p.status === 'progress') {
            seen.set(p.file || p.name || 'file', { loaded: p.loaded || 0, total: p.total || 0 })
            report()
          } else if (p.status === 'done' && p.file) {
            const prev = seen.get(p.file)
            if (prev) { seen.set(p.file, { loaded: prev.total, total: prev.total }); report() }
          }
        },
      })
      modelState = 'ready'
      return detector
    })().catch(e => {
      modelState = 'error'
      modelError = e?.message || String(e)
      pipelinePromise = null
      throw e
    })
  }
  return pipelinePromise
}

function toCanvas(source: CanvasImageSource, w: number, h: number): HTMLCanvasElement {
  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  canvas.getContext('2d')!.drawImage(source, 0, 0, w, h)
  return canvas
}

/**
 * Run real Grounding DINO inference in this browser on an image/video/canvas
 * source. Returns the same BrowserDetectResult shape as the COCO path, with
 * class_name = the matched prompt text. Pipeline boxes are already in
 * original-image pixels (verified in the Node test) — they are only clamped
 * here, since the pipeline can return values a pixel or two outside the image.
 */
export async function detectOpenVocabInBrowser(
  source: CanvasImageSource,
  w: number,
  h: number,
  prompts: string[],
  onProgress?: (loaded: number, total: number) => void,
  onPrompt?: (done: number, total: number, prompt: string) => void,
): Promise<BrowserDetectResult> {
  const detector = await ensureOpenVocabModel(onProgress)
  const { conf, iou: overlapThreshold } = getLocalSettings()
  const threshold = Math.max(0.1, Math.min(conf, 0.9))
  const canvas = toCanvas(source, w, h)
  // One detector call PER prompt. This is deliberate, verified in the Node
  // test: passing prompts as an array fails (the exported text encoder has
  // a fixed batch size of 1 — "Got: 3, Expected: 1"), and joining them into
  // one query ("bus. person. elephant.") silently misses objects — it found
  // only the bus, no people, in both q8 and fp32. A phrase without its
  // trailing period breaks the text encoder (Gather out-of-bounds), so every
  // phrase gets one. Cost: one inference run per prompt (~10-20 s each on a
  // throttled CPU, faster on a real laptop) — slow, but correct.
  const dets: Detection[] = []
  let totalMs = 0
  for (let i = 0; i < prompts.length; i++) {
    const phrase = prompts[i].endsWith('.') ? prompts[i] : `${prompts[i]}.`
    const t0 = performance.now()
    const out = await detector(canvas, [phrase], { threshold })
    totalMs += performance.now() - t0
    onPrompt?.(i + 1, prompts.length, prompts[i])
    for (const r of out as Array<{ label: string; score: number; box: { xmin: number; ymin: number; xmax: number; ymax: number } }>) {
      const label = r.label.replace(/\.+$/, '').trim()
      const x1 = Math.max(0, Math.min(w, r.box.xmin))
      const y1 = Math.max(0, Math.min(h, r.box.ymin))
      const x2 = Math.max(0, Math.min(w, r.box.xmax))
      const y2 = Math.max(0, Math.min(h, r.box.ymax))
      if (x2 <= x1 || y2 <= y1) continue
      dets.push({
        class_id: i,
        class_name: label,
        confidence: Math.round(r.score * 10000) / 10000,
        bbox: [Math.round(x1 * 10) / 10, Math.round(y1 * 10) / 10, Math.round(x2 * 10) / 10, Math.round(y2 * 10) / 10],
      })
    }
  }
  // Highest confidence first, then drop near-duplicate boxes for the SAME
  // prompt using the configured IoU threshold. The same box under two different prompts is kept —
  // the object genuinely matches both descriptions.
  dets.sort((a, b) => b.confidence - a.confidence)
  const kept: Detection[] = []
  for (const d of dets) {
    if (kept.some(k => k.class_id === d.class_id && iou(k.bbox, d.bbox) > overlapThreshold)) continue
    kept.push(d)
  }
  return {
    ...summarise(kept, totalMs),
    annotatedDataUrl: annotate(source, w, h, kept),
    source: 'browser',
    engine: 'In-browser (Grounding DINO)',
  }
}

/** Convenience wrapper for an uploaded image file. */
export async function detectOpenVocabFileInBrowser(
  file: File,
  prompts: string[],
  onProgress?: (loaded: number, total: number) => void,
  onPrompt?: (done: number, total: number, prompt: string) => void,
): Promise<BrowserDetectResult> {
  const bmp = await createImageBitmap(file)
  try {
    return await detectOpenVocabInBrowser(bmp, bmp.width, bmp.height, prompts, onProgress, onPrompt)
  } finally {
    bmp.close()
  }
}
