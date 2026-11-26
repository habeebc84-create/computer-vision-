import { useEffect, useState } from 'react'
import { Camera, Image as ImageIcon, ScanEye, ArrowRight, Video } from 'lucide-react'
import { api } from '../lib/api'
import type { PageId } from '../components/Layout'

export default function Inference({ go, online }: { online: boolean | null; go: (p: PageId) => void }) {
  const [model, setModel] = useState<any>(null)
  const [recent, setRecent] = useState<any[]>([])
  useEffect(() => {
    if (!online) return
    api.modelStatus().then(setModel).catch(() => null)
    api.summary().then(s => setRecent(s.recent || [])).catch(() => null)
  }, [online])
  const cards = [
    { id: 'live' as PageId, icon: Camera, title: 'Live webcam detection', body: 'See boxes, labels and confidence scores as your camera captures each frame.', cta: 'Open live detection' },
    { id: 'video' as PageId, icon: Video, title: 'Video detection', body: 'Analyze sampled frames in your browser or process a full video on a connected backend.', cta: 'Open video detection' },
    { id: 'image' as PageId, icon: ImageIcon, title: 'Image detection', body: 'Upload a JPG/PNG. The model returns an annotated image plus a full detections table (object, confidence, bounding box, class ID).', cta: 'Open image detection' },
  ]
  return (
    <div className="space-y-4">
      <div className="card flex flex-wrap items-center gap-3">
        <div className="rounded-xl bg-sky-500/15 p-3 text-sky-300"><ScanEye size={22} /></div>
        <div className="min-w-0 flex-1">
          <h2 className="font-semibold">Module 3 — Inference &amp; Visualization</h2>
          <p className="text-sm text-slate-400">Choose an input to begin. {online && model?.loaded ? 'A backend model is loaded and available.' : 'Image, live and sampled video detection can run in your browser.'}</p>
        </div>
      </div>
      <div className="grid gap-4 md:grid-cols-3">
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
        <h2 className="font-semibold">Saved sessions</h2>
        <p className="my-3 text-sm text-slate-300">Review saved images, video summaries and camera snapshots in Detection History.</p>
        <button className="btn-ghost" onClick={() => go('history')}>View detection history</button>
        {online && recent.length ? <ul className="mt-2 space-y-1 text-sm text-slate-300">{recent.slice(0, 5).map(r => <li key={r.id}>#{r.id} · {r.created_at} · {r.input_type} · {r.object_count} object(s) · {r.classes.join(', ') || 'none'}</li>)}</ul> : null}
      </div>
    </div>
  )
}
