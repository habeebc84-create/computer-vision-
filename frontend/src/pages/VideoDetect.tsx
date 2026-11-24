import { useEffect, useRef, useState } from 'react'
import { Download, FileVideo, Loader2, Upload } from 'lucide-react'
import { api, resolveUrl } from '../lib/api'
import { saveBrowserHistory } from '../lib/browserHistory'
import { DetectionTable } from '../components/ui'
import type { VideoSample } from '../lib/browserVideo'

export default function VideoDetect() {
  const [file, setFile] = useState<File | null>(null)
  const [job, setJob] = useState<any>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [historyStatus, setHistoryStatus] = useState('')
  const [mode, setMode] = useState<'browser' | 'backend'>('browser')
  const [samples, setSamples] = useState<VideoSample[]>([])
  const [selected, setSelected] = useState(0)
  const [sampleTotal, setSampleTotal] = useState(0)
  const active = useRef<AbortController | null>(null)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => () => {
    active.current?.abort()
    if (timer.current) clearTimeout(timer.current)
  }, [])

  const clearResults = () => {
    setError(''); setJob(null); setSamples([]); setSelected(0); setSampleTotal(0); setHistoryStatus('')
  }
  const pick = (next?: File) => {
    if (!next || busy) return
    clearResults()
    setFile(null)
    if (!/\.(mp4|mov|avi|webm)$/i.test(next.name) || !next.size || next.size > 100 * 1024 * 1024) {
      setError('Choose a nonempty MP4, MOV, AVI or WebM video up to 100 MB.')
      return
    }
    setFile(next)
  }
  const start = async () => {
    if (!file || busy) return
    const controller = new AbortController()
    active.current = controller
    clearResults()
    setBusy(true)
    const finish = (error?: unknown) => {
      if (controller.signal.aborted) return
      if (error) setError(error instanceof Error ? error.message : String(error))
      setBusy(false)
    }
    try {
      if (mode === 'browser') {
        const { sampleVideo } = await import('../lib/browserVideo')
        const results: VideoSample['result'][] = []
        const startedAt = performance.now()
        await sampleVideo(file, controller.signal, (sample, _done, total) => {
          results.push(sample.result)
          setSampleTotal(total)
          setSamples(current => [...current, sample])
        })
        if (controller.signal.aborted) return
        const elapsedMs = Math.max(1, performance.now() - startedAt)
        const detections = results.flatMap(result => result.detections)
        const confidence = detections.reduce((sum, detection) => sum + detection.confidence, 0)
        const preview = results.find(result => result.count > 0) || results[0]
        setHistoryStatus('Saving to Detection History…')
        try {
          await saveBrowserHistory({
            inputType: 'video',
            count: detections.length,
            classes: [...new Set(detections.map(detection => detection.class_name))].sort(),
            avgConfidence: detections.length ? confidence / detections.length : 0,
            inferenceMs: results.reduce((sum, result) => sum + result.inference_ms, 0),
            fps: results.length * 1000 / elapsedMs,
            model: preview?.engine || 'YOLOE · Objects365',
            name: file.name,
            previewDataUrl: preview?.annotatedDataUrl,
          })
          if (!controller.signal.aborted) setHistoryStatus('Saved to Detection History in this browser.')
        } catch {
          if (!controller.signal.aborted) setHistoryStatus('Could not save this result to Detection History.')
        }
        finish()
        return
      }
      const { job_id } = await api.startVideo(file)
      const poll = async () => {
        if (controller.signal.aborted) return
        try {
          const status = await api.videoStatus(job_id)
          if (controller.signal.aborted) return
          setJob(status)
          if (status.status === 'completed' || status.status === 'failed') finish()
          else timer.current = setTimeout(poll, 1500)
        } catch (error) { finish(error) }
      }
      await poll()
    } catch (error) { finish(error) }
  }
  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <div className="card lg:col-span-2">
        <fieldset className="mb-4 flex flex-wrap gap-2">
          <legend className="label">Processing mode</legend>
          <button type="button" disabled={busy} aria-pressed={mode === 'browser'} className={mode === 'browser' ? 'btn-primary' : 'btn-ghost'} onClick={() => { setMode('browser'); clearResults() }}>Browser · 365 categories</button>
          <button type="button" disabled={busy} aria-pressed={mode === 'backend'} className={mode === 'backend' ? 'btn-primary' : 'btn-ghost'} onClick={() => { setMode('backend'); clearResults() }}>Backend · all frames</button>
        </fieldset>
        <label onDragOver={event => event.preventDefault()} onDrop={event => { event.preventDefault(); pick(event.dataTransfer.files[0]) }} className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-white/15 bg-white/[0.02] px-4 py-10 text-center hover:border-sky-500/50">
          <FileVideo size={28} className="text-sky-400" />
          <span className="mt-2 font-semibold">Drop a video here, or click to browse</span>
          <span className="text-xs text-slate-400">MP4 / WebM / MOV / AVI · up to 100 MB</span>
          <input type="file" accept=".mp4,.webm,.mov,.avi,video/mp4,video/webm,video/quicktime" className="hidden" disabled={busy} onChange={event => { pick(event.currentTarget.files?.[0]); event.currentTarget.value = '' }} />
        </label>
        {file && <p className="mt-3 text-sm text-slate-300">Selected: <b>{file.name}</b> · {(file.size / 1048576).toFixed(1)} MB</p>}
        {error && <p className="mt-3 rounded-xl bg-red-500/10 border border-red-500/30 px-3 py-2 text-sm text-red-300">{error}</p>}
        <div className="mt-4 flex flex-wrap gap-2">
          <button className="btn-primary" onClick={start} disabled={busy || !file}>{busy ? <><Loader2 size={15} className="animate-spin" /> {mode === 'browser' && !sampleTotal ? 'Loading detector…' : 'Processing…'}</> : <><Upload size={15} /> Process video</>}</button>
          {job?.status === 'completed' && <a className="btn-ghost" href={resolveUrl(`/api/video/${job.id}/download`)}><Download size={15} /> Download processed video</a>}
        </div>
        {sampleTotal > 0 && (
          <div className="mt-4 space-y-3">
            <p role="status" className="text-sm text-slate-300">{samples.length} / {sampleTotal} sampled frames analyzed</p>
            {historyStatus && <p role="status" className="text-sm text-slate-300">{historyStatus}</p>}
            <div className="flex flex-wrap gap-2" aria-label="Sampled frames">
              {samples.map((sample, index) => (
                <button type="button" key={sample.time} aria-pressed={selected === index}
                  className={selected === index ? 'btn-primary' : 'btn-ghost'} onClick={() => setSelected(index)}>
                  {sample.time.toFixed(2)}s
                </button>
              ))}
            </div>
            {samples[selected] && <>
              <img className="w-full rounded-xl" alt="Annotated video frame" src={samples[selected].result.annotatedDataUrl} />
              {samples[selected].result.modelNotice && <p role="status" className="text-sm text-amber-200">{samples[selected].result.modelNotice}</p>}
              <DetectionTable dets={samples[selected].result.detections} />
            </>}
          </div>
        )}
        {job && (
          <div className="mt-5">
            <div className="flex justify-between text-sm"><span className="capitalize font-semibold">{job.status}</span><span>{job.progress}% · {job.frames}/{job.total_frames} frames</span></div>
            <div className="mt-2 h-3 overflow-hidden rounded-full bg-white/10"><div className="h-full rounded-full bg-gradient-to-r from-sky-500 to-cyan-400 transition-all" style={{ width: `${job.progress}%` }} /></div>
            {job.history_error && <p className="mt-2 text-sm text-amber-200">{job.history_error}</p>}
            {job.status === 'failed' && <p className="mt-2 text-sm text-red-300">{job.error}</p>}
            {job.status === 'completed' && job.summary && (
              <div className="mt-3 grid grid-cols-3 gap-3 text-center text-sm">
                <div className="rounded-xl bg-white/5 p-3"><p className="text-lg font-bold">{job.summary.frames}</p><p className="text-xs text-slate-400">Frames</p></div>
                <div className="rounded-xl bg-white/5 p-3"><p className="text-lg font-bold">{job.summary.objects}</p><p className="text-xs text-slate-400">Detections across frames</p></div>
                <div className="rounded-xl bg-white/5 p-3"><p className="text-lg font-bold">{job.fps.toFixed(1)}</p><p className="text-xs text-slate-400">Processing FPS</p></div>
              </div>
            )}
            {job.status === 'completed' && job.summary?.per_class && <p className="mt-3 text-sm text-slate-300">Classes: {Object.entries(job.summary.per_class).map(([k, v]) => `${k} (${v})`).join(', ')}</p>}
          </div>
        )}
      </div>
      <div className="card">
        <h2 className="font-semibold">Video results</h2>
        <p className="mt-2 text-sm text-slate-400">{mode === 'browser'
          ? 'The browser samples up to 24 frames across the video. Select a timestamp to inspect its boxes and labels.'
          : 'The backend processes every frame with the model selected in Settings. Download the annotated MP4 when it finishes.'}</p>
      </div>
    </div>
  )
}
