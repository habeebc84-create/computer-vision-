import { useEffect, useState } from 'react'
import { Download, FileText } from 'lucide-react'
import { api, toast } from '../lib/api'
import { EmptyState } from '../components/ui'
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

export default function Analytics() {
  const [dist, setDist] = useState<any[]>([])
  const [conf, setConf] = useState<any[]>([])
  const [summary, setSummary] = useState<any>(null)
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    Promise.all([api.classDist(), api.confidence(), api.summary()]).then(([a, b, c]) => { setDist(a); setConf(b); setSummary(c) }).catch(() => null)
  }, [])
  const report = async (fmt: 'csv' | 'pdf') => {
    setBusy(true)
    try { const r = await api.generateReport(fmt); toast(`Report generated — ${r.records} record(s)`); window.open(r.url, '_blank') }
    catch (e: any) { toast(e.message, 'err') } finally { setBusy(false) }
  }
  const empty = !summary || summary.total_records === 0
  const most = dist[0]
  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="card">
          <h2 className="font-semibold">Class distribution</h2>
          {dist.length ? <div className="mt-4 h-64"><ResponsiveContainer width="100%" height="100%"><BarChart data={dist}><CartesianGrid stroke="#1e293b" /><XAxis dataKey="name" stroke="#8b98b8" fontSize={11} /><YAxis stroke="#8b98b8" fontSize={11} /><Tooltip contentStyle={{ background: '#0e1730', border: '1px solid #334155' }} /><Bar dataKey="count" radius={[6, 6, 0, 0]}>{dist.map((_, i) => <Cell key={i} fill={['#0ea5e9', '#22d3ee', '#8b5cf6', '#34d399', '#f59e0b', '#f472b6'][i % 6]} />)}</Bar></BarChart></ResponsiveContainer></div> : <p className="mt-3 text-sm text-slate-400">No class data yet — charts update from recorded detections only.</p>}
          {most && <p className="mt-2 text-sm">Most frequently detected: <b className="capitalize">{most.name}</b> ({most.count} session(s))</p>}
        </div>
        <div className="card">
          <h2 className="font-semibold">Confidence distribution</h2>
          {conf.some(c => c.count > 0) ? <div className="mt-4 h-64"><ResponsiveContainer width="100%" height="100%"><BarChart data={conf}><CartesianGrid stroke="#1e293b" /><XAxis dataKey="range" stroke="#8b98b8" fontSize={11} /><YAxis stroke="#8b98b8" fontSize={11} /><Tooltip contentStyle={{ background: '#0e1730', border: '1px solid #334155' }} /><Bar dataKey="count" fill="#8b5cf6" radius={[6, 6, 0, 0]} /></BarChart></ResponsiveContainer></div> : <p className="mt-3 text-sm text-slate-400">No confidence data yet.</p>}
          <div className="mt-3 space-y-1 text-sm">
            <p className="flex justify-between"><span className="text-slate-400">Total objects detected</span><b>{summary?.total_detections ?? 0}</b></p>
            <p className="flex justify-between"><span className="text-slate-400">Average confidence</span><b>{summary?.avg_confidence ? (summary.avg_confidence * 100).toFixed(1) + '%' : '—'}</b></p>
            <p className="flex justify-between"><span className="text-slate-400">Average FPS</span><b>{summary?.avg_fps ? summary.avg_fps.toFixed(1) : '—'}</b></p>
          </div>
        </div>
      </div>
      <div className="card">
        <h2 className="flex items-center gap-2 font-semibold"><FileText size={16} className="text-sky-400" /> Reports</h2>
        {empty ? <div className="mt-3"><EmptyState title="No records to report on yet" body="Run a detection first — reports are generated from your recorded detection history, never from sample data." /></div> : (
          <div className="mt-3 flex flex-wrap gap-2">
            <button className="btn-primary" onClick={() => report('csv')} disabled={busy}><Download size={15} /> Download CSV report</button>
            <button className="btn-ghost" onClick={() => report('pdf')} disabled={busy}><Download size={15} /> Download PDF report</button>
            <p className="w-full text-xs text-slate-500">Reports include project name, date/time, model, input type, detected objects, counts, confidence scores and FPS.</p>
          </div>
        )}
      </div>
    </div>
  )
}
