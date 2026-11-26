export interface Detection { class_id: number; class_name: string; confidence: number; bbox: number[] }
export interface DetectResult { detections: Detection[]; count: number; unique_classes: string[]; avg_confidence: number; inference_ms: number; fps: number; record_id?: number; history_error?: string; annotated_url?: string; original_name?: string }
export interface ModelStatus { loaded: boolean; model_path: string; device: string; classes: number; class_names: string[]; loaded_at: string | null; error: string | null; conf: number; iou: number; imgsz: number }
export interface Summary { total_detections: number; total_records: number; images: number; videos: number; live: number; num_classes: number; avg_confidence: number; avg_fps: number; recent: HistoryRecord[] }
export interface HistoryRecord { id: number; created_at: string; input_type: string; classes: string[]; object_count: number; avg_conf: number; fps: number; inference_ms: number; output_path: string; model: string }
export interface BuiltinDataset { id: string; name: string; description: string; images: number; classes: number; size_mb: number | null; url: string; license: string; status: string; format?: string; external_url?: string; note?: string; imported_images?: number; imported_labels?: number; progress?: number; error?: string | null }
export interface DatasetSplit {
  train: number
  val: number
  unannotated_images: string[]
  duplicate_images: string[]
  classes_without_boxes: string[]
  classes_without_validation: string[]
}
export interface AvailableModel { id: string; path: string; kind: 'coco' | 'world' | 'trained'; name: string; classes: number | null; description: string; size_mb: number; downloaded: boolean; loaded?: boolean }

/**
 * API base URL. When the frontend is served by the Python backend (python run.py)
 * this is empty and relative /api/* calls go to the same origin. When the frontend
 * is hosted elsewhere (e.g. Vercel, which serves static files only and has NO
 * backend), set VITE_API_URL to the backend's URL, e.g. http://localhost:8000
 * or https://my-backend.onrender.com — then rebuild and redeploy.
 */
const BACKEND_KEY = 'ods-backend-url'
const defaultBase = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')
export const API_BASE: string = (() => {
  try { return localStorage.getItem(BACKEND_KEY) || defaultBase }
  catch { return defaultBase }
})()

export function restoreBackend() {
  localStorage.removeItem(BACKEND_KEY)
  window.location.reload()
}

export async function connectBackend(input: string) {
  const url = new URL(input)
  const local = ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname)
  if ((url.protocol !== 'https:' && !(local && url.protocol === 'http:')) ||
      url.username || url.password || url.search || url.hash) {
    throw new Error('Enter an HTTPS backend URL or a local server address.')
  }
  const base = url.href.replace(/\/+$/, '')
  const health = await req<{ status: string; model_loaded: boolean }>(base + '/api/health', {
    signal: AbortSignal.timeout(20000),
  })
  if (health?.status !== 'ok' || typeof health.model_loaded !== 'boolean') {
    throw new Error('This address did not return an object detection backend.')
  }
  localStorage.setItem(BACKEND_KEY, base)
  window.location.reload()
}

/** Prefix a backend-served path (/outputs/..., /dataset/..., /api/...) with API_BASE. */
export function resolveUrl(u: string): string {
  if (!u) return u
  if (/^https?:\/\//i.test(u)) return u
  return API_BASE + (u.startsWith('/') ? u : `/${u}`)
}

const BACKEND_OFFLINE_404 = 'Backend not reachable (API returned 404). Run the Python backend with python run.py, or set VITE_API_URL to your backend URL.'
const BACKEND_OFFLINE_NET = 'Backend not reachable — could not connect. Run the Python backend with python run.py, or set VITE_API_URL to your backend URL.'

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const url = /^https?:\/\//i.test(path) ? path : API_BASE + path
  let r: Response
  try {
    r = await fetch(url, init)
  } catch {
    // Network failure (backend down, wrong VITE_API_URL, CORS) — never show a stack trace.
    throw new ApiError(BACKEND_OFFLINE_NET, 0)
  }
  const text = await r.text()
  if (!r.ok) {
    let message: string | undefined
    if (r.headers.get('content-type')?.includes('application/json')) {
      try {
        const data = JSON.parse(text)
        const detail = data.detail || data.message
        if (typeof detail === 'string') message = detail
      } catch { /* Proxies can return invalid JSON. */ }
    }
    if (r.status === 404 && !message) message = BACKEND_OFFLINE_404
    throw new ApiError(message || `Request failed (${r.status})`, r.status)
  }
  try {
    return JSON.parse(text) as T
  } catch {
    throw new ApiError('Backend returned an invalid response.', 0)
  }
}

export class ApiError extends Error {
  constructor(message: string, readonly status: number) { super(message) }

  get backendUnavailable() { return this.status === 0 || this.status === 404 || this.status >= 500 }
}

export async function detectWithFallback<T>(backend: () => Promise<T>, browser: () => Promise<T>) {
  let backendError: ApiError | undefined
  if (await isBackendOnline()) {
    try {
      return { result: await backend(), source: 'backend' as const }
    } catch (error) {
      if (!(error instanceof ApiError) || !error.backendUnavailable) throw error
      backendError = error
    }
  }
  try {
    return { result: await browser(), source: 'browser' as const }
  } catch (error) {
    if (backendError) throw new Error(`${backendError.message} Browser detection also failed: ${String(error)}`)
    throw error
  }
}

export function parseDetectionPrompts(text: string): string[] {
  const prompts = new Map<string, string>()
  for (const part of text.split(',')) {
    const name = part.trim()
    if (!name || prompts.has(name.toLowerCase())) continue
    if ([...name].length > 40) throw new Error('Object names must be 40 characters or fewer.')
    prompts.set(name.toLowerCase(), name)
  }
  if (!prompts.size) throw new Error('Type at least one object name, such as microphone or headphones.')
  if (prompts.size > 20) throw new Error('At most 20 object names at a time.')
  return [...prompts.values()]
}

// Cached backend reachability: health is cheap, but Live checks it every
// frame tick — cache the answer for 10 s so the check costs nothing.
let _healthCache: { at: number; online: boolean } | null = null
export async function isBackendOnline(): Promise<boolean> {
  const now = Date.now()
  if (_healthCache && now - _healthCache.at < 10_000) return _healthCache.online
  let online = false
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 4000)
  try { online = (await api.health({ signal: controller.signal })).status === 'ok' }
  catch { online = false }
  finally { clearTimeout(timeout) }
  _healthCache = { at: Date.now(), online }
  return online
}
export function backendOnlineCached(): boolean | null {
  return _healthCache && Date.now() - _healthCache.at < 10_000 ? _healthCache.online : null
}

export const api = {
  health: (init?: RequestInit) => req<{ status: string; model_loaded: boolean; device: string }>('/api/health', init),
  modelStatus: () => req<ModelStatus>('/api/model/status'),
  loadModel: (path: string) => req<ModelStatus>('/api/model/load', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path }) }),
  getSettings: () => req<any>('/api/settings'),
  saveSettings: (s: any) => req<any>('/api/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(s) }),
  listModels: () => req<AvailableModel[]>('/api/models/available'),
  detectImage: async (file: File) => {
    const fd = new FormData(); fd.append('file', file)
    return req<DetectResult>('/api/detect/image', { method: 'POST', body: fd })
  },
  openVocabImage: async (file: File, classes: string[]) => {
    const fd = new FormData(); fd.append('file', file); fd.append('classes', JSON.stringify(classes))
    return req<DetectResult & { prompts?: string[] }>('/api/detect/open-vocab/image', { method: 'POST', body: fd })
  },
  openVocabFrame: async (blob: Blob, classes: string[]) => {
    const fd = new FormData(); fd.append('frame', blob, 'frame.jpg'); fd.append('classes', JSON.stringify(classes))
    return req<DetectResult & { prompts?: string[] }>('/api/detect/open-vocab/frame', { method: 'POST', body: fd })
  },
  detectFrame: async (blob: Blob) => {
    const fd = new FormData(); fd.append('frame', blob, 'frame.jpg')
    return req<DetectResult>('/api/detect/frame', { method: 'POST', body: fd })
  },
  saveFrame: async (blob: Blob, prompts?: string[]) => {
    const fd = new FormData(); fd.append('frame', blob, 'frame.jpg')
    if (prompts) fd.append('classes', JSON.stringify(prompts))
    return req<DetectResult>('/api/detect/frame/save', { method: 'POST', body: fd })
  },
  startVideo: async (file: File) => {
    const fd = new FormData(); fd.append('file', file)
    return req<{ job_id: string }>('/api/video/start', { method: 'POST', body: fd })
  },
  videoStatus: (id: string) => req<any>(`/api/video/${id}`),
  summary: () => req<Summary>('/api/analytics/summary'),
  classDist: () => req<{ name: string; count: number }[]>('/api/analytics/class-distribution'),
  activity: () => req<{ date: string; sessions: number; objects: number }[]>('/api/analytics/activity'),
  confidence: () => req<{ range: string; count: number }[]>('/api/analytics/confidence'),
  history: (q = '', init?: RequestInit) => req<HistoryRecord[]>(`/api/history${q}`, init),
  deleteHistory: (id: number) => req(`/api/history/${id}`, { method: 'DELETE' }),
  dsStats: () => req<any>('/api/dataset/stats'),
  dsImages: () => req<any[]>('/api/dataset/images'),
  dsImportCoco: async (file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    return req<{ saved: string[]; skipped: string[]; classes: number }>('/api/dataset/import-coco', { method: 'POST', body: fd })
  },
  dsUpload: async (files: FileList | File[]) => {
    const fd = new FormData(); Array.from(files).forEach(f => fd.append('files', f))
    return req<{ saved: string[]; skipped: string[]; errors?: { file: string; message: string }[] }>('/api/dataset/upload', { method: 'POST', body: fd })
  },
  dsDelete: (name: string) => req(`/api/dataset/images/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  dsClasses: () => req<{ classes: string[] }>('/api/dataset/classes'),
  dsSetClasses: (classes: string[]) => req('/api/dataset/classes', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ classes }) }),
  dsAnnotations: (name: string) => req<any>(`/api/dataset/annotations/${encodeURIComponent(name)}`),
  dsSaveAnnotations: (name: string, boxes: any[]) => req(`/api/dataset/annotations/${encodeURIComponent(name)}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ boxes }) }),
  dsSplit: (train_pct: number) => req<DatasetSplit>('/api/dataset/split', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ train_pct }) }),
  dsYaml: () => req<{ content: string }>('/api/dataset/data-yaml'),
  dsCatalog: () => req<BuiltinDataset[]>('/api/datasets'),
  dsDownload: (id: string, force = false) => req<any>(`/api/datasets/${encodeURIComponent(id)}/download${force ? '?force=true' : ''}`, { method: 'POST' }),
  dsStatus: (id: string) => req<BuiltinDataset>(`/api/datasets/${encodeURIComponent(id)}/status`),
  trainingStatus: () => req<any>('/api/training/status'),
  trainingStart: (cfg: any) => req('/api/training/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(cfg) }),
  trainingStop: () => req('/api/training/stop', { method: 'POST' }),
  loadBest: () => req<ModelStatus>('/api/training/load-best', { method: 'POST' }),
  generateReport: (format: 'csv' | 'pdf') => req<{ file: string; url: string; records: number }>('/api/reports/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ format }) }),
}
export function toast(msg: string, kind: 'ok' | 'err' = 'ok') {
  window.dispatchEvent(new CustomEvent('app-toast', { detail: { msg, kind, id: Date.now() } }))
}
