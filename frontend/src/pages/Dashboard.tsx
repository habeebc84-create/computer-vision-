import { useEffect, useState } from 'react'
import { Activity, Camera, Cpu, Gauge, Image as ImageIcon, Layers, Play, ScanEye, Target, TrendingUp } from 'lucide-react'
import { api, Summary } from '../lib/api'
import { StatCard, EmptyState } from '../components/ui'
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { PageId } from '../components/Layout'

export default function Dashboard({ go }: { go: (p: PageId) => void }) {
  const [s, setS] = useState<Summary | null>(null)
  const [dist, setDist] = useState<any[]>([])
  const [act, setAct] = useState<any[]>([])
  const [model, setModel] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  // Backend is offline when the summary call fails; the in-browser model is
  // then the real model, so track its state the same way the header badge does.
  const [backendOffline, setBackendOffline] = useState(false)
  const [browserModel, setBrowserModel] = useState<{ state: string; error: string }>({ state: 'not-downloaded', error: '' })
  useEffect(() => {
    Promise.all([api.summary(), api.classDist(), api.activity(), api.modelStatus()])
      .then(([a, b, c, d]) => { setS(a); setDist(b); setAct(c); setModel(d) })
      .catch(() => setBackendOffline(true)).finally(() => setLoading(false))
  }, [])
  useEffect(() => {
    if (!backendOffline) return
    let unsub: (() => void) | null = null
    let cancelled = false
    import('../lib/browserDetection').then(m => {
      if (cancelled) return
      setBrowserModel(m.getBrowserModelState())
      unsub = m.subscribeBrowserModelState(setBrowserModel)
      m.ensureBrowserModel().catch(() => null) // failures surface via subscription
    }).catch(() => null)
    return () => { cancelled = true; unsub?.() }
  }, [backendOffline])
  if (loading) return <div className="card animate-pulse text-sm text-slate-400">Loading dashboard…</div>
  const empty = !s || s.total_records === 0
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-6">
        <StatCard label="Total detections" value={String(s?.total_detections ?? 0)} icon={<Target size={20} />} />
        <StatCard label="Images processed" value={String(s?.images ?? 0)} icon={<ImageIcon size={20} />} accent="bg-cyan-500/15 text-cyan-300" />
        <StatCard label="Object classes" value={String(s?.num_classes ?? 0)} icon={<Layers size={20} />} accent="bg-emerald-500/15 text-emerald-300" />
        <StatCard label="Avg FPS" value={s?.avg_fps ? s.avg_fps.toFixed(1) : '—'} sub="Measured, not guaranteed" icon={<Gauge size={20} />} accent="bg-amber-500/15 text-amber-300" />
        <StatCard label="Model status" value={backendOffline ? (browserModel.state === 'ready' ? 'Ready' : browserModel.state === 'downloading' ? 'Loading…' : browserModel.state === 'error' ? 'Failed' : 'Not loaded') : (model?.loaded ? 'Ready' : 'Not loaded')} sub={backendOffline ? 'yolov8n.onnx · in-browser' : model?.model_path?.split('/').pop()} icon={<Cpu size={20} />} accent={(backendOffline ? browserModel.state === 'ready' : model?.loaded) ? 'bg-emerald-500/15 text-emerald-300' : 'bg-red-500/15 text-red-300'} />
      </div>
      <div className="grid gap-4 lg:grid-cols-3">
        <div className="card lg:col-span-2">
          <h2 className="flex items-center gap-2 font-semibold"><TrendingUp size={16} className="text-sky-400" /> Detection activity</h2>
          {act.length ? (
            <div className="mt-4 h-56"><ResponsiveContainer width="100%" height="100%">
              <LineChart data={act}><CartesianGrid stroke="#1e293b" /><XAxis dataKey="date" stroke="#8b98b8" fontSize={11} /><YAxis stroke="#8b98b8" fontSize={11} /><Tooltip contentStyle={{ background: '#0e1730', border: '1px solid #334155' }} /><Line type="monotone" dataKey="objects" stroke="#0ea5e9" strokeWidth={2} name="Objects" /><Line type="monotone" dataKey="sessions" stroke="#8b5cf6" strokeWidth={2} name="Sessions" /></LineChart>
            </ResponsiveContainer></div>
          ) : <div className="mt-4"><EmptyState title="No detection activity yet" body="Run a live or image detection and activity will appear here from real records." action={<button className="btn-primary" onClick={() => go('live')}><Play size={15} /> Start live detection</button>} /></div>}
        </div>
        <div className="card">
          <h2 className="flex items-center gap-2 font-semibold"><ScanEye size={16} className="text-cyan-400" /> Class distribution</h2>
          {dist.length ? (
            <div className="mt-4 h-56"><ResponsiveContainer width="100%" height="100%">
              <BarChart data={dist.slice(0, 8)} layout="vertical"><CartesianGrid stroke="#1e293b" /><XAxis type="number" stroke="#8b98b8" fontSize={11} /><YAxis type="category" dataKey="name" stroke="#8b98b8" fontSize={11} width={80} /><Tooltip contentStyle={{ background: '#0e1730', border: '1px solid #334155' }} /><Bar dataKey="count" fill="#22d3ee" radius={[0, 6, 6, 0]}>{dist.slice(0, 8).map((_, i) => <Cell key={i} fill={['#0ea5e9', '#22d3ee', '#8b5cf6', '#34d399', '#f59e0b', '#f472b6', '#a78bfa', '#38bdf8'][i % 8]} />)}</Bar></BarChart>
            </ResponsiveContainer></div>
          ) : <p className="mt-4 text-sm text-slate-400">No classes detected yet. Distribution is computed from recorded detections only.</p>}
          <div className="mt-4 space-y-2 text-sm">
            <p className="flex justify-between"><span className="text-slate-400">Avg confidence</span><b>{s?.avg_confidence ? (s.avg_confidence * 100).toFixed(1) + '%' : '—'}</b></p>
            <p className="flex justify-between"><span className="text-slate-400">Model</span><b className="truncate max-w-[180px]">{model?.model_path?.split('/').pop() || '—'}</b></p>
            <p className="flex justify-between"><span className="text-slate-400">Device</span><b>{model?.device || '—'}</b></p>
            <p className="flex justify-between"><span className="text-slate-400">Webcam</span><b>Browser camera</b></p>
          </div>
          <div className="mt-4 flex gap-2">
            <button className="btn-primary flex-1 justify-center" onClick={() => go('live')}><Camera size={15} /> Live</button>
            <button className="btn-ghost flex-1 justify-center" onClick={() => go('image')}><ImageIcon size={15} /> Image</button>
          </div>
        </div>
      </div>
      <div className="card">
        <h2 className="flex items-center gap-2 font-semibold"><Activity size={16} className="text-sky-400" /> Recent detections</h2>
        {s?.recent?.length ? (
          <div className="mt-3 overflow-x-auto"><table className="w-full text-sm">
            <thead><tr className="border-b border-white/10 text-left text-xs uppercase tracking-wider text-slate-400"><th className="py-2">Date / time</th><th className="py-2">Type</th><th className="py-2">Classes</th><th className="py-2">Objects</th><th className="py-2">Avg conf</th><th className="py-2">FPS</th></tr></thead>
            <tbody>{s.recent.map(r => <tr key={r.id} className="border-b border-white/5"><td className="py-2">{r.created_at}</td><td className="py-2 capitalize">{r.input_type}</td><td className="py-2">{r.classes.join(', ') || '—'}</td><td className="py-2">{r.object_count}</td><td className="py-2">{(r.avg_conf * 100).toFixed(0)}%</td><td className="py-2">{r.fps?.toFixed(1)}</td></tr>)}</tbody>
          </table></div>
        ) : <p className="mt-3 text-sm text-slate-400">No detections recorded yet — this table fills in after your first real detection.</p>}
      </div>
    </div>
  )
}
