import { useEffect, useState } from 'react'
import { Download, Search, Trash2 } from 'lucide-react'
import { api, HistoryRecord, resolveUrl, toast } from '../lib/api'
import { EmptyState } from '../components/ui'

export default function History() {
  const [rows, setRows] = useState<HistoryRecord[]>([])
  const [search, setSearch] = useState('')
  const [type, setType] = useState('')
  const [from, setFrom] = useState('')
  const [to, setTo] = useState('')
  const [open, setOpen] = useState<HistoryRecord | null>(null)
  const load = async () => {
    const q = new URLSearchParams()
    if (search) q.set('search', search)
    if (type) q.set('type', type)
    if (from) q.set('date_from', from)
    if (to) q.set('date_to', to)
    const s = q.toString()
    setRows(await api.history(s ? `?${s}` : ''))
  }
  useEffect(() => { load().catch(() => null) }, [])
  const del = async (id: number) => { await api.deleteHistory(id); setRows(r => r.filter(x => x.id !== id)); toast(`Record #${id} deleted`) }
  return (
    <div className="space-y-4">
      <div className="card flex flex-wrap gap-3">
        <div className="relative min-w-[200px] flex-1"><Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" /><input className="input !pl-9" placeholder="Search by class or type…" value={search} onChange={e => setSearch(e.target.value)} /></div>
        <select className="input max-w-[160px]" value={type} onChange={e => setType(e.target.value)}><option value="">All types</option><option value="image">Image</option><option value="live">Live</option></select>
        <input type="date" className="input max-w-[160px]" value={from} onChange={e => setFrom(e.target.value)} aria-label="From date" />
        <input type="date" className="input max-w-[160px]" value={to} onChange={e => setTo(e.target.value)} aria-label="To date" />
        <button className="btn-primary" onClick={load}>Apply filters</button>
      </div>
      {rows.length ? (
        <div className="card overflow-x-auto">
          <table className="w-full min-w-[720px] text-sm">
            <thead><tr className="border-b border-white/10 text-left text-xs uppercase tracking-wider text-slate-400"><th className="py-2">ID</th><th className="py-2">Date / time</th><th className="py-2">Type</th><th className="py-2">Classes</th><th className="py-2">Objects</th><th className="py-2">Avg conf</th><th className="py-2">FPS</th><th className="py-2">Actions</th></tr></thead>
            <tbody>{rows.map(r => (
              <tr key={r.id} className="border-b border-white/5">
                <td className="py-2">#{r.id}</td><td className="py-2">{r.created_at}</td><td className="py-2 capitalize">{r.input_type}</td>
                <td className="py-2">{r.classes.join(', ') || '—'}</td><td className="py-2">{r.object_count}</td>
                <td className="py-2">{(r.avg_conf * 100).toFixed(0)}%</td><td className="py-2">{r.fps?.toFixed(1)}</td>
                <td className="py-2"><div className="flex gap-1">
                  <button className="btn-ghost !px-2 !py-1" onClick={() => setOpen(r)}>View</button>
                  {r.output_path && <a className="btn-ghost !px-2 !py-1" href={resolveUrl(`/${r.output_path}`)} download><Download size={13} /></a>}
                  <button className="btn-danger !px-2 !py-1" onClick={() => del(r.id)} aria-label={`Delete record ${r.id}`}><Trash2 size={13} /></button>
                </div></td>
              </tr>))}
            </tbody>
          </table>
        </div>
      ) : <EmptyState title="No detection history yet" body="Detection sessions are stored in SQLite (database/detections.db) after real image or saved live detections. Nothing is shown here until then." />}
      {open && (
        <div className="card">
          <h2 className="font-semibold">Record #{open.id} — details</h2>
          <dl className="mt-3 grid gap-2 text-sm sm:grid-cols-2">
            <div><dt className="text-slate-400">Date / time</dt><dd>{open.created_at}</dd></div>
            <div><dt className="text-slate-400">Input type</dt><dd className="capitalize">{open.input_type}</dd></div>
            <div><dt className="text-slate-400">Detected classes</dt><dd>{open.classes.join(', ') || 'None'}</dd></div>
            <div><dt className="text-slate-400">Object count</dt><dd>{open.object_count}</dd></div>
            <div><dt className="text-slate-400">Average confidence</dt><dd>{(open.avg_conf * 100).toFixed(1)}%</dd></div>
            <div><dt className="text-slate-400">Processing FPS</dt><dd>{open.fps?.toFixed(1)}</dd></div>
            <div><dt className="text-slate-400">Inference time</dt><dd>{open.inference_ms} ms</dd></div>
            <div><dt className="text-slate-400">Model</dt><dd className="break-all">{open.model || '—'}</dd></div>
            <div><dt className="text-slate-400">Output file</dt><dd className="break-all">{open.output_path || '—'}</dd></div>
          </dl>
          {open.output_path && <a className="btn-ghost mt-3" href={resolveUrl(`/${open.output_path}`)} download><Download size={15} /> Download output</a>}
          <button className="btn-ghost ml-2 mt-3" onClick={() => setOpen(null)}>Close</button>
        </div>
      )}
    </div>
  )
}
