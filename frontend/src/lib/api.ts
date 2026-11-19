export interface Detection { class_id: number; class_name: string; confidence: number; bbox: number[] }
export interface DetectResult { detections: Detection[]; count: number; unique_classes: string[]; avg_confidence: number; inference_ms: number; fps: number; record_id?: number; annotated_url?: string; original_name?: string }
export interface ModelStatus { loaded: boolean; model_path: string; device: string; classes: number; class_names: string[]; loaded_at: string | null; error: string | null; conf: number; iou: number; imgsz: number }
export interface Summary { total_detections: number; total_records: number; images: number; videos: number; live: number; num_classes: number; avg_confidence: number; avg_fps: number; recent: HistoryRecord[] }
export interface HistoryRecord { id: number; created_at: string; input_type: string; classes: string[]; object_count: number; avg_conf: number; fps: number; inference_ms: number; output_path: string; model: string }
export interface BuiltinDataset { id: string; name: string; description: string; images: number; classes: number; size_mb: number; url: string; license: string; status: string; imported_images?: number; imported_labels?: number; progress?: number; error?: string | null }
export interface AvailableModel { id: string; path: string; kind: 'coco' | 'world'; name: string; classes: number | null; description: string; size_mb: number; downloaded: boolean; loaded?: boolean }

/**
 * API base URL. When the frontend is served by the Python backend (python run.py)
 * this is empty and relative /api/* calls go to the same origin. When the frontend
 * is hosted elsewhere (e.g. Vercel, which serves static files only and has NO
 * backend), set VITE_API_URL to the backend's URL, e.g. http://localhost:8000
 * or https://my-backend.onrender.com — then rebuild and redeploy.
 */
export const API_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')

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
    throw new Error(BACKEND_OFFLINE_NET)
  }

  const text = await r.text()
  if (!r.ok) {
    // Real backend errors are JSON with `detail` — keep those verbatim.
    const ct = r.headers.get('content-type') || ''
    if (ct.includes('application/json')) {
      let data: any
      try { data = JSON.parse(text) } catch { /* fall through to friendly errors */ }
      if (data?.detail || data?.message) throw new Error(String(data.detail || data.message))
    }
    // A non-JSON error body (e.g. Vercel's 404 HTML page when no backend exists)
    // must never be surfaced as the error message.
    if (r.status === 404 && path.startsWith('/api')) throw new Error(BACKEND_OFFLINE_404)
    throw new Error(`Request failed (${r.status})`)
  }
  if (!text) return {} as T
  try {
    return JSON.parse(text) as T
  } catch {
    throw new Error(`Request failed (${r.status})`)
  }
}
export const api = {
  health: () => req<{ status: string; model_loaded: boolean; device: string }>('/api/health'),
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
  saveFrame: async (blob: Blob) => {
    const fd = new FormData(); fd.append('frame', blob, 'frame.jpg')
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
  history: (q = '') => req<HistoryRecord[]>(`/api/history${q}`),
  deleteHistory: (id: number) => req(`/api/history/${id}`, { method: 'DELETE' }),
  dsStats: () => req<any>('/api/dataset/stats'),
  dsImages: () => req<any[]>('/api/dataset/images'),
  dsUpload: async (files: FileList | File[]) => {
    const fd = new FormData(); Array.from(files).forEach(f => fd.append('files', f))
    return req<any>('/api/dataset/upload', { method: 'POST', body: fd })
  },
  dsDelete: (name: string) => req(`/api/dataset/images/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  dsClasses: () => req<{ classes: string[] }>('/api/dataset/classes'),
  dsSetClasses: (classes: string[]) => req('/api/dataset/classes', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ classes }) }),
  dsAnnotations: (name: string) => req<any>(`/api/dataset/annotations/${encodeURIComponent(name)}`),
  dsSaveAnnotations: (name: string, boxes: any[]) => req(`/api/dataset/annotations/${encodeURIComponent(name)}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ boxes }) }),
  dsSplit: (train_pct: number) => req<any>('/api/dataset/split', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ train_pct }) }),
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
