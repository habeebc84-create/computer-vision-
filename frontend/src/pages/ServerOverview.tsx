import { useEffect, useState } from 'react'
import { Activity, ArrowUpRight, Cpu, Gauge, Image as ImageIcon, Layers, ScanEye, Target, TrendingUp } from 'lucide-react'
import { api, Summary } from '../lib/api'
import { StatCard, EmptyState } from '../components/ui'
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { PageId } from '../components/Layout'

export default function ServerOverview({ go }: { go: (p: PageId) => void }) {
  const [s, setS] = useState<Summary | null>(null)
  const [dist, setDist] = useState<any[]>([])
  const [act, setAct] = useState<any[]>([])
  const [model, setModel] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  useEffect(() => {
    let cancelled = false
    void Promise.allSettled([api.summary(), api.classDist(), api.activity(), api.modelStatus()])
      .then(([summary, distribution, activity, status]) => {
        if (cancelled) return
        if (summary.status === 'fulfilled') setS(summary.value)
        if (distribution.status === 'fulfilled') setDist(distribution.value)
        if (activity.status === 'fulfilled') setAct(activity.value)
        if (status.status === 'fulfilled') setModel(status.value)
        setLoading(false)
      })
    return () => { cancelled = true }
  }, [])
  return (
    <div className="space-y-6">
      <section aria-labelledby="metrics-heading">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <h2 id="metrics-heading" className="text-lg font-semibold tracking-tight">Detection metrics</h2>
          {!loading && !s && <span className="text-xs text-slate-400">Backend metrics unavailable</span>}
        </div>
        {loading ? <div className="card text-sm text-slate-400" role="status">Loading detection metrics…</div> : <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-5">
        <StatCard label="Total detections" value={s ? String(s.total_detections) : '—'} icon={<Target size={20} />} />
        <StatCard label="Images processed" value={s ? String(s.images) : '—'} icon={<ImageIcon size={20} />} accent="bg-indigo-400/15 text-indigo-200" />
        <StatCard label="Object classes" value={s ? String(s.num_classes) : '—'} icon={<Layers size={20} />} accent="bg-emerald-500/15 text-emerald-300" />
        <StatCard label="Avg FPS" value={s?.avg_fps ? s.avg_fps.toFixed(1) : '—'} icon={<Gauge size={20} />} accent="bg-amber-500/15 text-amber-300" />
        <StatCard label="Model status" value={model?.loaded ? 'Ready' : 'On demand'} sub={model?.model_path?.split(/[\\/]/).pop()} icon={<Cpu size={20} />} />
        </div>}
      </section>
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <div className="card">
          <h2 className="flex items-center gap-2 font-semibold"><TrendingUp size={16} className="text-indigo-300" /> Detection activity</h2>
          {act.length ? (
            <div className="mt-4 h-56"><ResponsiveContainer width="100%" height="100%">
              <LineChart data={act}><CartesianGrid stroke="#1e293b" /><XAxis dataKey="date" stroke="#8b98b8" fontSize={11} /><YAxis stroke="#8b98b8" fontSize={11} /><Tooltip contentStyle={{ background: '#151b28', border: '1px solid #334155' }} /><Line type="monotone" dataKey="objects" stroke="#0ea5e9" strokeWidth={2} name="Objects" /><Line type="monotone" dataKey="sessions" stroke="#8b5cf6" strokeWidth={2} name="Sessions" /></LineChart>
            </ResponsiveContainer></div>
          ) : <div className="mt-4"><EmptyState title="No detection activity yet" body="Saved detections will appear here over time." /></div>}
        </div>
        <div className="card">
          <h2 className="flex items-center gap-2 font-semibold"><ScanEye size={16} className="text-indigo-300" /> Class distribution</h2>
          {dist.length ? (
            <div className="mt-4 h-56"><ResponsiveContainer width="100%" height="100%">
              <BarChart data={dist.slice(0, 8)} layout="vertical"><CartesianGrid stroke="#1e293b" /><XAxis type="number" stroke="#8b98b8" fontSize={11} /><YAxis type="category" dataKey="name" stroke="#8b98b8" fontSize={11} width={80} /><Tooltip contentStyle={{ background: '#151b28', border: '1px solid #334155' }} /><Bar dataKey="count" fill="#22d3ee" radius={[0, 6, 6, 0]}>{dist.slice(0, 8).map((_, i) => <Cell key={i} fill={['#0ea5e9', '#22d3ee', '#8b5cf6', '#34d399', '#f59e0b', '#f472b6', '#a78bfa', '#38bdf8'][i % 8]} />)}</Bar></BarChart>
            </ResponsiveContainer></div>
          ) : <p className="mt-4 text-sm text-slate-400">Detected classes will appear after the first saved result.</p>}
          <div className="mt-4 space-y-2 text-sm">
            <p className="flex justify-between"><span className="text-slate-400">Avg confidence</span><b>{s?.avg_confidence ? (s.avg_confidence * 100).toFixed(1) + '%' : '—'}</b></p>
            <p className="flex justify-between"><span className="text-slate-400">Model</span><b className="truncate max-w-[180px]">{model?.model_path?.split('/').pop() || '—'}</b></p>
            <p className="flex justify-between"><span className="text-slate-400">Device</span><b>{model?.device || '—'}</b></p>
          </div>

        </div>
      </div>
      <div className="card">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="flex items-center gap-2 font-semibold"><Activity size={16} className="text-indigo-300" /> Recent detections</h2>
          <button className="btn-ghost" onClick={() => go('history')}>View history <ArrowUpRight size={15} /></button>
        </div>
        {s?.recent?.length ? (
          <div className="mt-3 overflow-x-auto"><table className="w-full text-sm">
            <thead><tr className="border-b border-white/10 text-left text-xs uppercase tracking-wider text-slate-400"><th className="py-2">Date / time</th><th className="py-2">Type</th><th className="py-2">Classes</th><th className="py-2">Objects</th><th className="py-2">Avg conf</th><th className="py-2">FPS</th></tr></thead>
            <tbody>{s.recent.map(r => <tr key={r.id} className="border-b border-white/5"><td className="py-2">{r.created_at}</td><td className="py-2 capitalize">{r.input_type}</td><td className="py-2">{r.classes.join(', ') || '—'}</td><td className="py-2">{r.object_count}</td><td className="py-2">{(r.avg_conf * 100).toFixed(0)}%</td><td className="py-2">{r.fps?.toFixed(1)}</td></tr>)}</tbody>
          </table></div>
        ) : <p className="mt-3 text-sm text-slate-400">No backend detections to show. Open Detection History for results saved in this browser.</p>}
      </div>
    </div>
  )
}