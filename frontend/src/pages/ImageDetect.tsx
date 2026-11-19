import { useState } from 'react'
import { Download, ImagePlus, Loader2, Upload } from 'lucide-react'
import { api, DetectResult, resolveUrl, toast } from '../lib/api'
import { DetectionTable } from '../components/ui'

export default function ImageDetect() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [result, setResult] = useState<DetectResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const pick = (f: File | undefined) => {
    if (!f) return
    if (!/image\/(jpeg|png)/.test(f.type)) { setError('Please choose a JPG or PNG image.'); return }
    setFile(f); setPreview(URL.createObjectURL(f)); setResult(null); setError('')
  }
  const run = async () => {
    if (!file) return
    setBusy(true); setError('')
    try { const r = await api.detectImage(file); setResult(r); toast(`Detected ${r.count} object(s) in ${(r.inference_ms).toFixed(0)} ms`) }
    catch (e: any) { setError(e.message); toast(e.message, 'err') }
    finally { setBusy(false) }
  }
  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <div className="card lg:col-span-2">
        <label className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-white/15 bg-white/[0.02] px-4 py-10 text-center hover:border-sky-500/50">
          <ImagePlus size={28} className="text-sky-400" />
          <span className="mt-2 font-semibold">Drop an image here, or click to browse</span>
          <span className="text-xs text-slate-400">JPG / PNG · up to 20 MB · processed by the real YOLOv8 model</span>
          <input type="file" accept="image/jpeg,image/png" className="hidden" onChange={e => pick(e.target.files?.[0])} />
        </label>
        {error && <p className="mt-3 rounded-xl bg-red-500/10 border border-red-500/30 px-3 py-2 text-sm text-red-300">{error}</p>}
        {preview && (
          <div className="mt-4">
            <div className="grid gap-3 sm:grid-cols-2">
              <figure><figcaption className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-400">Original</figcaption><img src={preview} alt="Original upload" className="w-full rounded-xl border border-white/10" /></figure>
              <figure><figcaption className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-400">Annotated — YOLOv8 output</figcaption>
                {result?.annotated_url ? <img src={resolveUrl(result.annotated_url)} alt="Annotated detection result" className="w-full rounded-xl border border-white/10" /> : <div className="flex h-full min-h-[180px] items-center justify-center rounded-xl border border-white/10 text-sm text-slate-500">Run detection to see bounding boxes</div>}
              </figure>
            </div>
            <div className="mt-4 flex flex-wrap gap-2">
              <button className="btn-primary" onClick={run} disabled={busy || !file}>{busy ? <><Loader2 size={15} className="animate-spin" /> Detecting…</> : <><Upload size={15} /> Run detection</>}</button>
              {result?.annotated_url && <a className="btn-ghost" href={resolveUrl(result.annotated_url)} download><Download size={15} /> Download annotated image</a>}
            </div>
          </div>
        )}
      </div>
      <div className="space-y-4">
        <div className="card">
          <h2 className="font-semibold">Detection summary</h2>
          {result ? (
            <div className="mt-3 space-y-2 text-sm">
              <p className="flex justify-between"><span className="text-slate-400">Total objects</span><b>{result.count}</b></p>
              <p className="flex justify-between"><span className="text-slate-400">Unique classes</span><b>{result.unique_classes.join(', ') || '—'}</b></p>
              <p className="flex justify-between"><span className="text-slate-400">Avg confidence</span><b>{(result.avg_confidence * 100).toFixed(1)}%</b></p>
              <p className="flex justify-between"><span className="text-slate-400">Inference time</span><b>{result.inference_ms.toFixed(0)} ms</b></p>
              <p className="flex justify-between"><span className="text-slate-400">FPS (equivalent)</span><b>{result.fps.toFixed(1)}</b></p>
              <p className="flex justify-between"><span className="text-slate-400">History record</span><b>#{result.record_id}</b></p>
            </div>
          ) : <p className="mt-3 text-sm text-slate-400">Upload an image and run detection — results and statistics appear here.</p>}
        </div>
        <div className="card"><h2 className="font-semibold">Detected objects</h2><div className="mt-3">{result ? <DetectionTable dets={result.detections} /> : <p className="text-sm text-slate-400">No detection run yet.</p>}</div></div>
      </div>
    </div>
  )
}
