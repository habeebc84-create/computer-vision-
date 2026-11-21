import { ReactNode } from 'react'
import { Inbox } from 'lucide-react'

export function StatCard({ label, value, sub, icon, accent }: { label: string; value: string; sub?: string; icon: ReactNode; accent?: string }) {
  return (
    <div className="card flex items-start justify-between gap-3">
      <div>
        <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">{label}</p>
        <p className="mt-2 text-2xl font-bold tracking-tight">{value}</p>
        {sub && <p className="mt-1 text-xs text-slate-400">{sub}</p>}
      </div>
      <div className={`rounded-xl p-2.5 ${accent || 'bg-sky-500/15 text-sky-300'}`}>{icon}</div>
    </div>
  )
}

export function EmptyState({ title, body, action }: { title: string; body: string; action?: ReactNode }) {
  return (
    <div className="card flex flex-col items-center justify-center py-12 text-center">
      <div className="rounded-2xl bg-white/5 p-4 text-slate-400"><Inbox size={28} /></div>
      <h3 className="mt-4 font-semibold">{title}</h3>
      <p className="mt-1 max-w-md text-sm text-slate-400">{body}</p>
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}

export function DetectionTable({ dets }: { dets: import('../lib/api').Detection[] }) {
  if (!dets.length) return <p className="text-sm text-slate-400">No objects detected in this input.</p>
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead><tr className="text-left text-xs uppercase tracking-wider text-slate-400 border-b border-white/10">
          <th className="py-2 pr-2">Object</th><th className="py-2 pr-2">Confidence</th><th className="py-2 pr-2">Bounding box</th><th className="py-2">Class ID</th>
        </tr></thead>
        <tbody>
          {dets.map((d, i) => (
            <tr key={i} className="border-b border-white/5">
              <td className="py-2 pr-2 font-medium capitalize">{d.class_name}</td>
              <td className="py-2 pr-2"><span className="chip">{(d.confidence * 100).toFixed(1)}%</span></td>
              <td className="py-2 pr-2 font-mono text-xs text-slate-300">[{d.bbox.map(v => Math.round(v)).join(', ')}]</td>
              <td className="py-2">{d.class_id}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
