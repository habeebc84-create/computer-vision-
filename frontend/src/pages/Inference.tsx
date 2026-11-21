import { useEffect, useState } from 'react'
import { Camera, Image as ImageIcon, ScanEye, ArrowRight } from 'lucide-react'
import { api } from '../lib/api'
import type { PageId } from '../components/Layout'

export default function Inference({ go }: { go: (p: PageId) => void }) {
  const [model, setModel] = useState<any>(null)
  const [recent, setRecent] = useState<any[]>([])
  useEffect(() => {
    api.modelStatus().then(setModel).catch(() => null)
    api.summary().then(s => setRecent(s.recent || [])).catch(() => null)
  }, [])
  const cards = [
    { id: 'live' as PageId, icon: Camera, title: 'Live webcam detection', body: 'Browser camera → frame capture → YOLOv8 inference → boxes, labels, confidence, FPS and inference ms drawn live on a canvas overlay.', cta: 'Open live detection' },
    { id: 'image' as PageId, icon: ImageIcon, title: 'Image detection', body: 'Upload a JPG/PNG. The model returns an annotated image plus a full detections table (object, confidence, bounding box, class ID).', cta: 'Open image detection' },
  ]
  return (
    <div className="space-y-4">
      <div className="card flex flex-wrap items-center gap-3">
        <div className="rounded-xl bg-sky-500/15 p-3 text-sky-300"><ScanEye size={22} /></div>
        <div className="min-w-0 flex-1">
          <h2 className="font-semibold">Module 3 — Inference &amp; Visualization</h2>
          <p className="text-sm text-slate-400">Both inference modes run the same loaded model: <b className="text-slate-200">{model?.model_path?.split('/').pop() || 'yolov8n.pt'}</b> · {model?.classes ?? 0} classes · device {model?.device || 'cpu'} · conf {model?.conf} · IoU {model?.iou} · imgsz {model?.imgsz}. Change these in Settings — they apply to real inference.</p>
        </div>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        {cards.map(c => (
          <button key={c.id} onClick={() => go(c.id)} className="card text-left transition hover:border-sky-500/40">
            <c.icon size={22} className="text-sky-400" />
            <h3 className="mt-3 font-semibold">{c.title}</h3>
            <p className="mt-1 text-sm text-slate-400">{c.body}</p>
            <span className="mt-3 inline-flex items-center gap-1 text-sm font-semibold text-sky-300">{c.cta} <ArrowRight size={14} /></span>
          </button>
        ))}
      </div>
      <div className="card">
        <h2 className="font-semibold">How inference works here</h2>
        <ol className="mt-3 list-decimal space-y-1.5 pl-5 text-sm text-slate-300">
          <li>Frames/images are sent to the FastAPI backend (this computer, port 8000).</li>
          <li>The loaded YOLOv8 model runs <code className="rounded bg-white/10 px-1">model.predict()</code> with your conf / IoU / image-size settings.</li>
          <li>Returned boxes are drawn with class name + confidence %; counts, average confidence, inference ms and measured FPS are shown.</li>
          <li>Image detections, completed video sessions, and saved live snapshots appear in Detection History. Browser results are saved on this device.</li>
        </ol>
        <h3 className="mt-5 font-semibold">Recent inference sessions</h3>
        {recent.length ? <ul className="mt-2 space-y-1 text-sm text-slate-300">{recent.slice(0, 5).map(r => <li key={r.id}>#{r.id} · {r.created_at} · {r.input_type} · {r.object_count} object(s) · {r.classes.join(', ') || 'none'}</li>)}</ul> : <p className="mt-2 text-sm text-slate-400">No inference sessions recorded yet.</p>}
      </div>
    </div>
  )
}
