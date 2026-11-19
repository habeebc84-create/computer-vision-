import { useEffect, useRef, useState } from 'react'
import { Download, FileVideo, Loader2, Upload } from 'lucide-react'
import { api, toast } from '../lib/api'

export default function VideoDetect() {
  const [file, setFile] = useState<File | null>(null)
  const [job, setJob] = useState<any>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const timer = useRef<any>(null)

  useEffect(() => () => { if (timer.current) clearInterval(timer.current) }, [])

  const start = async () => {
    if (!file) return
    setBusy(true); setError(''); setJob(null)
    try {
      const { job_id } = await api.startVideo(file)
      toast('Video processing started — frame-by-frame YOLOv8 inference')
      timer.current = setInterval(async () => {
        try {
          const s = await api.videoStatus(job_id)
          setJob(s)
          if (s.status === 'completed' || s.status === 'failed') clearInterval(timer.current)
        } catch { /* keep polling */ }
      }, 1500)
    } catch (e: any) { setError(e.message); toast(e.message, 'err') }
    finally { setBusy(false) }
  }
  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <div className="card lg:col-span-2">
        <label className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-white/15 bg-white/[0.02] px-4 py-10 text-center hover:border-sky-500/50">
          <FileVideo size={28} className="text-sky-400" />
          <span className="mt-2 font-semibold">Drop a video here, or click to browse</span>
          <span className="text-xs text-slate-400">MP4 / AVI / MOV · processed frame by frame on the backend</span>
          <input type="file" accept="video/mp4,video/avi,video/quicktime,.mov,.avi" className="hidden" onChange={e => setFile(e.target.files?.[0] || null)} />
        </label>
        {file && <p className="mt-3 text-sm text-slate-300">Selected: <b>{file.name}</b> · {(file.size / 1048576).toFixed(1)} MB</p>}
        {error && <p className="mt-3 rounded-xl bg-red-500/10 border border-red-500/30 px-3 py-2 text-sm text-red-300">{error}</p>}
        <div className="mt-4 flex gap-2">
          <button className="btn-primary" onClick={start} disabled={busy || !file}>{busy ? <><Loader2 size={15} className="animate-spin" /> Uploading…</> : <><Upload size={15} /> Process video</>}</button>
          {job?.status === 'completed' && <a className="btn-ghost" href={`/api/video/${job.id}/download`}><Download size={15} /> Download processed video</a>}
        </div>
        {job && (
          <div className="mt-5">
            <div className="flex justify-between text-sm"><span className="capitalize font-semibold">{job.status}</span><span>{job.progress}% · {job.frames}/{job.total_frames} frames</span></div>
            <div className="mt-2 h-3 overflow-hidden rounded-full bg-white/10"><div className="h-full rounded-full bg-gradient-to-r from-sky-500 to-cyan-400 transition-all" style={{ width: `${job.progress}%` }} /></div>
            {job.status === 'failed' && <p className="mt-2 text-sm text-red-300">{job.error}</p>}
            {job.status === 'completed' && job.summary && (
              <div className="mt-3 grid grid-cols-3 gap-3 text-center text-sm">
                <div className="rounded-xl bg-white/5 p-3"><p className="text-lg font-bold">{job.summary.frames}</p><p className="text-xs text-slate-400">Frames</p></div>
                <div className="rounded-xl bg-white/5 p-3"><p className="text-lg font-bold">{job.summary.objects}</p><p className="text-xs text-slate-400">Objects detected</p></div>
                <div className="rounded-xl bg-white/5 p-3"><p className="text-lg font-bold">{job.fps.toFixed(1)}</p><p className="text-xs text-slate-400">Processing FPS</p></div>
              </div>
            )}
            {job.status === 'completed' && job.summary?.per_class && <p className="mt-3 text-sm text-slate-300">Classes: {Object.entries(job.summary.per_class).map(([k, v]) => `${k} (${v})`).join(', ')}</p>}
          </div>
        )}
      </div>
      <div className="card">
        <h2 className="font-semibold">How video detection works</h2>
        <ol className="mt-3 list-decimal space-y-2 pl-5 text-sm text-slate-300">
          <li>Upload an MP4, AVI or MOV file.</li>
          <li>The backend reads every frame with OpenCV.</li>
          <li>Each frame runs through the loaded YOLOv8 model.</li>
          <li>Boxes, labels and confidence are drawn on the frame.</li>
          <li>Download the processed MP4 when it finishes.</li>
        </ol>
        <p className="mt-4 text-xs text-slate-500">Processing runs in a background job, so the page stays responsive. Progress above is real — it counts frames the model has actually processed.</p>
      </div>
    </div>
  )
}
