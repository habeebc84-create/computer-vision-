export interface Detection { class_id: number; class_name: string; confidence: number; bbox: number[] }
export interface DetectResult { detections: Detection[]; count: number; unique_classes: string[]; avg_confidence: number; inference_ms: number; fps: number; record_id?: number; annotated_url?: string; original_name?: string }
export interface ModelStatus { loaded: boolean; model_path: string; device: string; classes: number; class_names: string[]; loaded_at: string | null; error: string | null; conf: number; iou: number; imgsz: number }
export interface Summary { total_detections: number; total_records: number; images: number; videos: number; live: number; num_classes: number; avg_confidence: number; avg_fps: number; recent: HistoryRecord[] }
export interface HistoryRecord { id: number; created_at: string; input_type: string; classes: string[]; object_count: number; avg_conf: number; fps: number; inference_ms: number; output_path: string; model: string }

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init)
  const text = await r.text()
  let data: any = {}
  try { data = text ? JSON.parse(text) : {} } catch { data = { detail: text } }
  if (!r.ok) throw new Error(data.detail || data.message || `Request failed (${r.status})`)
  return data as T
}
export const api = {
  health: () => req<{ status: string; model_loaded: boolean; device: string }>('/api/health'),
  modelStatus: () => req<ModelStatus>('/api/model/status'),
  loadModel: (path: string) => req<ModelStatus>('/api/model/load', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path }) }),
  getSettings: () => req<any>('/api/settings'),
  saveSettings: (s: any) => req<any>('/api/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(s) }),
  detectImage: async (file: File) => {
    const fd = new FormData(); fd.append('file', file)
    return req<DetectResult>('/api/detect/image', { method: 'POST', body: fd })
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
  trainingStatus: () => req<any>('/api/training/status'),
  trainingStart: (cfg: any) => req('/api/training/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(cfg) }),
  trainingStop: () => req('/api/training/stop', { method: 'POST' }),
  loadBest: () => req<ModelStatus>('/api/training/load-best', { method: 'POST' }),
  generateReport: (format: 'csv' | 'pdf') => req<{ file: string; url: string; records: number }>('/api/reports/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ format }) }),
}
export function toast(msg: string, kind: 'ok' | 'err' = 'ok') {
  window.dispatchEvent(new CustomEvent('app-toast', { detail: { msg, kind, id: Date.now() } }))
}
