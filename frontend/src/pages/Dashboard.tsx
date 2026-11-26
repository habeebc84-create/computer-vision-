import { lazy, Suspense, useEffect, useState } from 'react'
import { Camera, Image as ImageIcon, Layers, Target, Video } from 'lucide-react'
import { StatCard } from '../components/ui'
import type { PageId } from '../components/Layout'
import { listBrowserHistory, type BrowserHistoryRecord } from '../lib/browserHistory'

const ServerOverview = lazy(() => import('./ServerOverview'))

function BrowserOverview({ go }: { go: (page: PageId) => void }) {
  const [rows, setRows] = useState<BrowserHistoryRecord[] | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    let active = true
    listBrowserHistory().then(data => { if (active) setRows(data) })
      .catch(error => { if (active) setError(String(error.message || error)) })
    return () => { active = false }
  }, [])
  if (error) return <p role="alert" className="card">{error}</p>
  if (!rows) return <p role="status">Loading browser history…</p>
  return <section className="space-y-4" aria-labelledby="saved-heading">
    <h2 id="saved-heading" className="text-xl font-semibold">Saved on this device</h2>
    <div className="grid gap-3 sm:grid-cols-3">
      <StatCard label="Saved sessions" value={String(rows.length)} icon={<Camera size={20} />} />
      <StatCard label="Objects detected" value={String(rows.reduce((sum, row) => sum + row.object_count, 0))} icon={<Target size={20} />} />
      <StatCard label="Distinct classes" value={String(new Set(rows.flatMap(row => row.classes)).size)} icon={<Layers size={20} />} />
    </div>
    <div className="card">
      <div className="flex items-center justify-between gap-3"><h3 className="font-semibold">Recent detections</h3>
        <button className="btn-ghost" onClick={() => go('history')}>View history</button></div>
      {rows.length ? <ul className="mt-3 divide-y divide-white/10">{rows.slice(0, 5).map(row =>
        <li key={row.id} className="flex flex-wrap items-baseline justify-between gap-2 py-3 text-sm">
          <div className="min-w-0"><p className="break-words font-semibold">{row.original_name || row.input_type}</p>
          <time className="text-slate-400" dateTime={row.created_at}>{new Date(row.created_at).toLocaleString()}</time></div>
          <span>{row.object_count} objects</span>
        </li>)}</ul> : <p className="mt-3 text-sm text-slate-300">Your first saved detection will appear here.</p>}
    </div>
  </section>
}

export default function Dashboard({ go, online }: { go: (page: PageId) => void; online: boolean | null }) {
  return <div className="space-y-6">
      <section aria-labelledby="workspace-heading" className="grid overflow-hidden rounded-2xl border border-indigo-300/20 bg-[#1c2340] md:grid-cols-[1.12fr_1fr]">
        <div className="flex flex-col justify-center px-6 py-8 sm:px-9 sm:py-10">
          <h2 id="workspace-heading" className="max-w-lg text-3xl font-bold tracking-tight sm:text-4xl">Object detection workspace</h2>
          <p className="mt-3 max-w-md text-sm leading-6 text-slate-300">Choose a photo, video or camera feed to see detected objects and their confidence scores.</p>
          <div className="mt-6 flex flex-wrap gap-2.5">
            <button className="btn-primary" onClick={() => go('image')}><ImageIcon size={17} /> Upload an image</button>
            <button className="btn-ghost" onClick={() => go('live')}><Camera size={17} /> Start live detection</button>
            <button className="btn-ghost" onClick={() => go('video')}><Video size={17} /> Analyze video</button>
          </div>
        </div>
        <div className="relative min-h-[155px] overflow-hidden md:min-h-[260px]">
          <img src={`${import.meta.env.BASE_URL}city-street.webp`} alt="" fetchPriority="high" className="absolute inset-0 h-full w-full object-cover object-center" />
          <div aria-hidden="true" className="absolute inset-0 bg-gradient-to-b from-[#1c2340]/60 via-transparent to-indigo-950/40 md:bg-gradient-to-r" />
        </div>
      </section>
    {online === null ? <p role="status">Checking connection…</p> : online ?
      <Suspense fallback={<p role="status">Loading server metrics…</p>}><ServerOverview go={go} /></Suspense> :
      <BrowserOverview go={go} />}
  </div>
}
