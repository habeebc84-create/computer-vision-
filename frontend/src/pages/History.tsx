import { useEffect, useState, type FormEvent } from 'react'
import { Download, Search, Trash2 } from 'lucide-react'
import { api, ApiError, resolveUrl, toast, type HistoryRecord } from '../lib/api'
import {
  BROWSER_HISTORY_CHANGED, deleteBrowserHistory, getBrowserHistoryPreview, listBrowserHistory,
  type BrowserHistoryRecord,
} from '../lib/browserHistory'
import { EmptyState } from '../components/ui'

type HistoryRow = (HistoryRecord & { source: 'backend' }) | (BrowserHistoryRecord & { source: 'browser' })

const initialFilters = { search: '', type: '', from: '', to: '' }
const inputTypes: Record<string, string> = {
  image: 'Image', video: 'Video', live: 'Live snapshot', 'open-vocab': 'Open vocabulary',
}
const formatDate = new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' })

function rowTime(row: HistoryRow) {
  return new Date(row.created_at.replace(' ', 'T'))
}

function displayTime(row: HistoryRow) {
  const date = rowTime(row)
  return Number.isFinite(date.getTime()) ? formatDate.format(date) : row.created_at
}

function rowKey(row: HistoryRow) {
  return `${row.source}:${row.id}`
}

function displayId(row: HistoryRow) {
  return row.source === 'backend' ? `#${row.id}` : row.id.slice(0, 8)
}

function outputUrl(row: HistoryRow) {
  return row.source === 'backend' && row.output_path ? resolveUrl(row.output_path) : ''
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error)
}

function HistoryDetails({ row, onClose }: { row: HistoryRow; onClose: () => void }) {
  const hasBrowserPreview = row.source === 'browser' && row.has_preview
  const [preview, setPreview] = useState({ url: '', error: '', loading: hasBrowserPreview })
  useEffect(() => {
    if (row.source !== 'browser' || !row.has_preview) return
    let active = true
    let url = ''
    getBrowserHistoryPreview(row.id).then(blob => {
      if (!active) return
      if (!blob) {
        setPreview({ url: '', error: 'The saved preview could not be found.', loading: false })
        return
      }
      url = URL.createObjectURL(blob)
      setPreview({ url, error: '', loading: false })
    }).catch(error => {
      if (active) setPreview({ url: '', error: errorMessage(error), loading: false })
    })
    return () => {
      active = false
      if (url) URL.revokeObjectURL(url)
    }
  }, [row])

  const url = outputUrl(row) || preview.url
  const video = row.source === 'backend' && row.input_type === 'video'
  const previewName = row.input_type === 'video' ? 'Annotated sampled frame' : 'Annotated detection'
  return (
    <section className="card" aria-label="Detection details">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="font-semibold">Record {displayId(row)} details</h2>
        <button className="btn-ghost" onClick={onClose}>Close</button>
      </div>
      <dl className="mt-3 grid gap-2 text-sm sm:grid-cols-2">
        <div><dt className="text-slate-400">Date / time</dt><dd>{displayTime(row)}</dd></div>
        <div><dt className="text-slate-400">Source</dt><dd>{row.source === 'browser' ? 'This browser' : 'Backend'}</dd></div>
        <div><dt className="text-slate-400">Input type</dt><dd>{inputTypes[row.input_type] ?? row.input_type}</dd></div>
        {row.source === 'browser' && row.original_name && (
          <div><dt className="text-slate-400">File</dt><dd className="break-all">{row.original_name}</dd></div>
        )}
        <div><dt className="text-slate-400">Detected classes</dt><dd>{row.classes.join(', ') || 'None'}</dd></div>
        <div><dt className="text-slate-400">Object count</dt><dd>{row.object_count}</dd></div>
        <div><dt className="text-slate-400">Average confidence</dt><dd>{(row.avg_conf * 100).toFixed(1)}%</dd></div>
        <div><dt className="text-slate-400">Processing FPS</dt><dd>{row.fps?.toFixed(1)}</dd></div>
        <div><dt className="text-slate-400">Inference time</dt><dd>{row.inference_ms} ms</dd></div>
        <div><dt className="text-slate-400">Model</dt><dd className="break-all">{row.model || '—'}</dd></div>
      </dl>
      {preview.loading && <p className="mt-3 text-sm text-slate-400" role="status">Loading saved preview…</p>}
      {preview.error && <p className="mt-3 text-sm text-red-300" role="alert">{preview.error}</p>}
      {url && (
        <div className="mt-4 space-y-3">
          {video
            ? <video className="max-h-96 max-w-full rounded-xl" src={url} controls preload="metadata" aria-label="Processed video" />
            : <img className="max-h-96 max-w-full rounded-xl object-contain" src={url} alt={previewName} />}
          <a className="btn-ghost" href={url} download={row.source === 'browser' ? `detection-${displayId(row)}.jpg` : true}>
            <Download size={15} /> {row.source === 'browser' ? 'Download preview' : 'Download output'}
          </a>
        </div>
      )}
    </section>
  )
}

export default function History() {
  const [browserRows, setBrowserRows] = useState<HistoryRow[]>([])
  const [backendRows, setBackendRows] = useState<HistoryRow[]>([])
  const [loading, setLoading] = useState({ browser: true, backend: true })
  const [notices, setNotices] = useState({ browser: '', backend: '' })
  const [filters, setFilters] = useState(initialFilters)
  const [applied, setApplied] = useState(initialFilters)
  const [open, setOpen] = useState<HistoryRow | null>(null)
  const [deleting, setDeleting] = useState<string | null>(null)

  useEffect(() => {
    let active = true
    let browserRequest = 0
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 8000)
    const query = new URLSearchParams()
    if (applied.search) query.set('search', applied.search)
    if (applied.type) query.set('type', applied.type)
    if (applied.from) query.set('date_from', applied.from)
    if (applied.to) query.set('date_to', applied.to)

    setLoading({ browser: true, backend: true })
    setNotices({ browser: '', backend: '' })
    setBrowserRows([])
    setBackendRows([])

    const refreshBrowser = async () => {
      const request = ++browserRequest
      try {
        const records = await listBrowserHistory(applied)
        if (!active || request !== browserRequest) return
        setBrowserRows(records.map(record => ({ ...record, source: 'browser' })))
        setNotices(current => ({ ...current, browser: '' }))
      } catch (error) {
        if (active && request === browserRequest) {
          setNotices(current => ({ ...current, browser: `Could not load browser history. ${errorMessage(error)}` }))
        }
      } finally {
        if (active && request === browserRequest) setLoading(current => ({ ...current, browser: false }))
      }
    }

    void refreshBrowser()
    window.addEventListener(BROWSER_HISTORY_CHANGED, refreshBrowser)
    const search = query.toString()
    api.history(search ? `?${search}` : '', { signal: controller.signal }).then(records => {
      if (active) setBackendRows(records.map(record => ({ ...record, source: 'backend' })))
    }).catch(error => {
      if (!active) return
      const unreachable = error instanceof ApiError && (error.status === 0 || error.status === 404)
      setNotices(current => ({
        ...current,
        backend: unreachable
          ? 'The backend is unavailable. Browser history is available on this device.'
          : `Could not load backend history. ${errorMessage(error)}`,
      }))
    }).finally(() => {
      clearTimeout(timeout)
      if (active) setLoading(current => ({ ...current, backend: false }))
    })

    return () => {
      active = false
      clearTimeout(timeout)
      controller.abort()
      window.removeEventListener(BROWSER_HISTORY_CHANGED, refreshBrowser)
    }
  }, [applied])

  const applyFilters = (event: FormEvent) => {
    event.preventDefault()
    setOpen(null)
    setApplied({ ...filters, search: filters.search.trim() })
  }

  const remove = async (row: HistoryRow) => {
    setDeleting(rowKey(row))
    try {
      if (row.source === 'browser') {
        await deleteBrowserHistory(row.id)
        setBrowserRows(records => records.filter(record => record.id !== row.id))
      } else {
        await api.deleteHistory(row.id)
        setBackendRows(records => records.filter(record => record.id !== row.id))
      }
      setOpen(current => current && rowKey(current) === rowKey(row) ? null : current)
      toast('Detection record deleted')
    } catch (error) {
      toast(`Could not delete the record. ${errorMessage(error)}`, 'err')
    } finally {
      setDeleting(null)
    }
  }

  const rows = [...browserRows, ...backendRows].sort((a, b) => rowTime(b).getTime() - rowTime(a).getTime())
  const filtering = Object.values(applied).some(Boolean)
  const pending = loading.browser || loading.backend

  return (
    <div className="space-y-4">
      <form className="card flex flex-wrap gap-3" onSubmit={applyFilters}>
        <div className="relative min-w-0 basis-full flex-1 sm:min-w-[200px]">
          <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input className="input !pl-9" aria-label="Search detections" placeholder="Search detections…"
            value={filters.search} onChange={event => setFilters({ ...filters, search: event.target.value })} />
        </div>
        <select className="input sm:max-w-[160px]" aria-label="Detection type" value={filters.type}
          onChange={event => setFilters({ ...filters, type: event.target.value })}>
          <option value="">All types</option>
          {Object.entries(inputTypes).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </select>
        <input type="date" className="input sm:max-w-[160px]" value={filters.from}
          max={filters.to || undefined} onChange={event => setFilters({ ...filters, from: event.target.value })} aria-label="From date" />
        <input type="date" className="input sm:max-w-[160px]" value={filters.to}
          min={filters.from || undefined} onChange={event => setFilters({ ...filters, to: event.target.value })} aria-label="To date" />
        <button type="submit" className="btn-primary" disabled={deleting !== null}>Apply filters</button>
      </form>
      {notices.backend && <p className="text-sm text-slate-400" role="status">{notices.backend}</p>}
      {notices.browser && <p className="text-sm text-red-300" role="alert">{notices.browser}</p>}
      {rows.length ? (
        <div className="card overflow-x-auto" role="region" aria-label="Detection history records" tabIndex={0}>
          <table className="w-full min-w-[40rem] text-sm">
            <thead><tr className="border-b border-white/10 text-left text-xs uppercase tracking-wider text-slate-400">
              <th scope="col" className="p-2">ID</th><th scope="col" className="p-2">Date / time</th>
              <th scope="col" className="p-2">Type</th><th scope="col" className="p-2">Classes</th>
              <th scope="col" className="p-2">Objects</th><th scope="col" className="p-2">Avg conf</th>
              <th scope="col" className="p-2">FPS</th><th scope="col" className="p-2">Actions</th>
            </tr></thead>
            <tbody>{rows.map(row => (
              <tr key={rowKey(row)} className="border-b border-white/5">
                <td className="p-2 whitespace-nowrap">{displayId(row)}</td>
                <td className="p-2 whitespace-nowrap">
                  {displayTime(row)}
                  <span className="block text-xs text-slate-400">{row.source === 'browser' ? 'This browser' : 'Backend'}</span>
                </td>
                <td className="p-2">{inputTypes[row.input_type] ?? row.input_type}</td>
                <td className="p-2 max-w-64">
                  <span className="line-clamp-2">{row.classes.join(', ') || '—'}</span>
                  {row.source === 'browser' && row.original_name && (
                    <span className="block truncate text-xs text-slate-400" title={row.original_name}>{row.original_name}</span>
                  )}
                </td>
                <td className="p-2">{row.object_count}</td>
                <td className="p-2">{(row.avg_conf * 100).toFixed(0)}%</td>
                <td className="p-2">{row.fps?.toFixed(1)}</td>
                <td className="p-2"><div className="flex gap-1">
                  <button className="btn-ghost !px-2 !py-1" onClick={() => setOpen(row)}>View</button>
                  {outputUrl(row) && <a className="btn-ghost !px-2 !py-1" href={outputUrl(row)} download aria-label={`Download record ${displayId(row)}`}><Download size={13} /></a>}
                  <button className="btn-danger !px-2 !py-1" onClick={() => void remove(row)}
                    disabled={deleting !== null} aria-label={`Delete record ${displayId(row)}`}><Trash2 size={13} /></button>
                </div></td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      ) : pending ? <p className="card text-sm text-slate-400" role="status">Loading detection history…</p>
        : <EmptyState title={filtering ? 'No matching detections' : 'No detection history yet'}
          body={filtering ? 'Change the filters to see more records.' : 'Run image or video detection, or save a live snapshot, to add a record.'} />}
      {open && <HistoryDetails key={rowKey(open)} row={open} onClose={() => setOpen(null)} />}
    </div>
  )
}
