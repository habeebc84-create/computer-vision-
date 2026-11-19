import { useEffect, useState } from 'react'
import { Download, ImagePlus, Loader2, Sparkles, Upload } from 'lucide-react'
import { api, DetectResult, isBackendOnline, resolveUrl, toast } from '../lib/api'
// NOTE: browserDetection is dynamically imported only when the backend is
// offline, so the ~600 KB ONNX Runtime bundle stays out of the main chunk.
import { DetectionTable } from '../components/ui'

function parsePrompts(text: string): string[] {
  return [...new Set(text.split(',').map(s => s.trim()).filter(Boolean))]
}

export default function ImageDetect() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [result, setResult] = useState<DetectResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [mode, setMode] = useState<'coco' | 'world'>('coco')
  const [classesText, setClassesText] = useState('bus, person, traffic light')
  // Where the last result actually came from — never mislabelled.
  const [source, setSource] = useState<'backend' | 'browser' | null>(null)
  const [modelPct, setModelPct] = useState<number | null>(null)

  // If the world model was chosen in Settings, start in Any-object mode.
  useEffect(() => {
    api.modelStatus().then(m => { if (m?.model_path?.includes('world')) setMode('world') }).catch(() => null)
  }, [])

  const pick = (f: File | undefined) => {
    if (!f) return
    if (!/image\/(jpeg|png)/.test(f.type)) { setError('Please choose a JPG or PNG image.'); return }
    setFile(f); setPreview(URL.createObjectURL(f)); setResult(null); setError(''); setSource(null)
  }
  const run = async () => {
    if (!file) return
    setBusy(true); setError(''); setModelPct(null)
    try {
      let r: DetectResult
      if (mode === 'world') {
        // YOLO-World only runs on the backend — there is no browser build of it.
        if (!(await isBackendOnline())) throw new Error('Any-object (YOLO-World) detection needs the Python backend — it cannot run in the browser. Switch to COCO mode to detect in-browser, or start the backend with python run.py.')
        const prompts = parsePrompts(classesText)
        if (prompts.length === 0) throw new Error('Type at least one object name to detect, e.g. "bus, dog".')
        if (prompts.length > 20) throw new Error('At most 20 object names at a time.')
        r = await api.openVocabImage(file, prompts)
        setSource('backend')
      } else if (await isBackendOnline()) {
        r = await api.detectImage(file)
        setSource('backend')
      } else {
        // Backend unreachable: real YOLOv8 inference right here in the browser.
        const { detectImageFileInBrowser } = await import('../lib/browserDetection')
        r = await detectImageFileInBrowser(file, (loaded, total) => {
          if (total > 0) setModelPct(Math.round((loaded / total) * 100))
        })
        setModelPct(null)
        setSource('browser')
      }
      setResult(r); toast(`Detected ${r.count} object(s) in ${(r.inference_ms).toFixed(0)} ms`)
    }
    catch (e: any) { setError(e.message); toast(e.message, 'err') }
    finally { setBusy(false); setModelPct(null) }
  }
  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <div className="card lg:col-span-2">
        <div className="flex flex-wrap items-center gap-2" role="tablist" aria-label="Detection mode">
          <button className={mode === 'coco' ? 'btn-primary' : 'btn-ghost'} onClick={() => setMode('coco')}>COCO classes (80)</button>
          <button className={mode === 'world' ? 'btn-primary' : 'btn-ghost'} onClick={() => setMode('world')}><Sparkles size={15} /> Any object — type its name</button>
        </div>
        {mode === 'world' && (
          <div className="mt-3">
            <label className="label">Objects to find (comma-separated, up to 20)</label>
            <input className="input" value={classesText} onChange={e => setClassesText(e.target.value)} placeholder="red bus, traffic light, dog" aria-label="Object names to detect" />
            <p className="mt-1 text-xs text-slate-500">YOLO-World detects anything you name — not just the 80 COCO classes. Be specific (“red bus” works better than “thing”). The world model (~25 MB) downloads once on first use.</p>
          </div>
        )}
        <label className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-white/15 bg-white/[0.02] px-4 py-10 text-center hover:border-sky-500/50">
          <ImagePlus size={28} className="text-sky-400" />
          <span className="mt-2 font-semibold">Drop an image here, or click to browse</span>
          <span className="text-xs text-slate-400">JPG / PNG · up to 20 MB · processed by the real YOLOv8 model</span>
          <input type="file" accept="image/jpeg,image/png" className="hidden" onChange={e => pick(e.target.files?.[0])} />
        </label>
        {error && <p className="mt-3 rounded-xl bg-red-500/10 border border-red-500/30 px-3 py-2 text-sm text-red-300">{error}</p>}
        {modelPct !== null && <p className="mt-3 rounded-xl bg-sky-500/10 border border-sky-500/30 px-3 py-2 text-sm text-sky-200">Downloading detection model… {modelPct}% (~12 MB, once only — it is cached in your browser after that).</p>}
        {source === 'browser' && <p className="mt-3 rounded-xl bg-violet-500/10 border border-violet-500/30 px-3 py-2 text-sm text-violet-200">Running in your browser — no backend connected. Real YOLOv8 inference on your device; nothing was uploaded, and this result is not saved to history.</p>}
        {preview && (
          <div className="mt-4">
            <div className="grid gap-3 sm:grid-cols-2">
              <figure><figcaption className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-400">Original</figcaption><img src={preview} alt="Original upload" className="w-full rounded-xl border border-white/10" /></figure>
              <figure><figcaption className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-400">Annotated — YOLOv8 output</figcaption>
                {(() => {
                  const src = (result as any)?.annotatedDataUrl || (result?.annotated_url ? resolveUrl(result.annotated_url) : '')
                  return src ? <img src={src} alt="Annotated detection result" className="w-full rounded-xl border border-white/10" /> : <div className="flex h-full min-h-[180px] items-center justify-center rounded-xl border border-white/10 text-sm text-slate-500">Run detection to see bounding boxes</div>
                })()}
              </figure>
            </div>
            <div className="mt-4 flex flex-wrap gap-2">
              <button className="btn-primary" onClick={run} disabled={busy || !file}>{busy ? <><Loader2 size={15} className="animate-spin" /> Detecting…</> : <><Upload size={15} /> Run detection</>}</button>
              {(() => {
                const src = (result as any)?.annotatedDataUrl || (result?.annotated_url ? resolveUrl(result.annotated_url) : '')
                return src ? <a className="btn-ghost" href={src} download="annotated.jpg"><Download size={15} /> Download annotated image</a> : null
              })()}
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
              {(result as any).prompts && <p className="flex justify-between"><span className="text-slate-400">Looking for</span><b className="text-right">{(result as any).prompts.join(', ')}</b></p>}
              <p className="flex justify-between"><span className="text-slate-400">Unique classes</span><b>{result.unique_classes.join(', ') || '—'}</b></p>
              <p className="flex justify-between"><span className="text-slate-400">Avg confidence</span><b>{(result.avg_confidence * 100).toFixed(1)}%</b></p>
              <p className="flex justify-between"><span className="text-slate-400">Inference time</span><b>{result.inference_ms.toFixed(0)} ms</b></p>
              <p className="flex justify-between"><span className="text-slate-400">FPS (equivalent)</span><b>{result.fps.toFixed(1)}</b></p>
              <p className="flex justify-between"><span className="text-slate-400">Engine</span><b>{source === 'browser' ? 'In-browser (ONNX)' : 'Backend (PyTorch)'}</b></p>
              <p className="flex justify-between"><span className="text-slate-400">History record</span><b>{result.record_id ? `#${result.record_id}` : '— not saved (in-browser)'}</b></p>
            </div>
          ) : <p className="mt-3 text-sm text-slate-400">Upload an image and run detection — results and statistics appear here.</p>}
        </div>
        <div className="card"><h2 className="font-semibold">Detected objects</h2><div className="mt-3">{result ? <DetectionTable dets={result.detections} /> : <p className="text-sm text-slate-400">No detection run yet.</p>}</div></div>
      </div>
    </div>
  )
}
