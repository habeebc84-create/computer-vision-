import { ReactNode } from 'react'
import { Inbox } from 'lucide-react'

export function StatCard({ label, value, sub, icon, accent }: { label: string; value: string; sub?: string; icon: ReactNode; accent?: string }) {
  return (
    <div className="card">
      <div className={`mb-3 inline-flex rounded-lg p-2 ${accent || 'bg-indigo-400/15 text-indigo-200'}`}>{icon}</div>
      <p className="text-sm text-slate-400">{label}</p>
      <p className="mt-1 break-words text-xl font-bold tracking-tight sm:text-2xl">{value}</p>
      {sub && <p className="mt-1 break-words text-xs text-slate-400">{sub}</p>}
    </div>
  )
}

export function EmptyState({ title, body, action }: { title: string; body: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-white/10 bg-ink/40 px-4 py-9 text-center">
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
      <table className="w-full min-w-[365px] text-sm">
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
