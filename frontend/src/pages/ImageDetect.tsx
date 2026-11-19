import { useEffect, useState } from 'react'
import { Download, ImagePlus, Loader2, Sparkles, Upload } from 'lucide-react'
import { api, ApiError, DetectResult, isBackendOnline, resolveUrl, toast } from '../lib/api'
import type { BrowserModel } from '../lib/browserDetection'
import { DetectionTable } from '../components/ui'

type Mode = 'broad' | 'package' | 'standard' | 'world'

function parsePrompts(text: string): string[] {
  return [...new Set(text.split(',').map(s => s.trim()).filter(Boolean))]
}

export default function ImageDetect() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [result, setResult] = useState<DetectResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [chosenMode, setMode] = useState<Mode | null>(null)
  const [backendModel, setBackendModel] = useState('')
  const worldSelected = backendModel.includes('world')
  const customSelected = Boolean(backendModel) && !/^yolov8[nsmlx]\.pt$/.test(backendModel)
  const backendSelected = backendModel && backendModel !== 'yolov8n.pt'
  const mode = chosenMode ?? (worldSelected ? 'world' : backendSelected ? 'standard' : 'broad')
  const [classesText, setClassesText] = useState('bus, person, traffic light')
  // Where the last result actually came from — never mislabelled.
  const [source, setSource] = useState<'backend' | 'browser' | null>(null)
  const [modelPct, setModelPct] = useState<number | null>(null)

  // Honor an explicit backend selection while defaulting a fresh install to 365 classes.
  useEffect(() => {
    let mounted = true
    api.modelStatus().then(m => {
      if (mounted && m?.model_path) setBackendModel(m.model_path.split(/[/\\]/).pop() || '')
    }).catch(() => null)
    return () => { mounted = false }
  }, [])

  useEffect(() => {
    if (!file) return
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const pick = (f: File | undefined) => {
    if (!f || busy) return
    if (!/^image\/(jpeg|jpg|png)$/.test(f.type)) { setError('Please choose a JPG or PNG image.'); return }
    if (f.size > 20 * 1024 * 1024) { setError('Image larger than 20 MB.'); return }
    setFile(f); setResult(null); setError(''); setSource(null)
  }

  const runInBrowser = async (image: File, model: BrowserModel) => {
    const { detectImageFileInBrowser } = await import('../lib/browserDetection')
    const result = await detectImageFileInBrowser(image, (loaded, total) => {
      if (total > 0) setModelPct(Math.min(100, Math.round((loaded / total) * 100)))
    }, model)
    setSource('browser')
    return result
  }
  const run = async () => {
    if (!file) return
    setMode(mode); setBusy(true); setError(''); setModelPct(null); setResult(null); setSource(null)
    try {
      let r: DetectResult
      if (mode === 'broad' || mode === 'package') {
        r = await runInBrowser(file, mode)
      } else if (mode === 'world') {
        // YOLO-World only runs on the backend — there is no browser build of it.
        if (!(await isBackendOnline())) throw new Error('Searching by name needs the Python backend. Start it with python run.py or use Everyday objects.')
        const prompts = parsePrompts(classesText)
        if (prompts.length === 0) throw new Error('Type at least one object name to detect, e.g. "bus, dog".')
        if (prompts.length > 20) throw new Error('At most 20 object names at a time.')
        r = await api.openVocabImage(file, prompts)
        setSource('backend')
      } else if (await isBackendOnline()) {
        try {
          r = await api.detectImage(file)
          setSource('backend')
        } catch (e) {
          if (customSelected || !(e instanceof ApiError) || !e.backendUnavailable) throw e
          r = await runInBrowser(file, 'coco')
        }
      } else {
        if (customSelected) throw new Error(`Start the Python backend to use ${backendModel}.`)
        r = await runInBrowser(file, 'coco')
      }
      setResult(r); toast(`Detected ${r.count} object(s) in ${(r.inference_ms).toFixed(0)} ms`)
    }
    catch (e: any) { setError(e.message); toast(e.message, 'err') }
    finally { setBusy(false); setModelPct(null) }
  }
  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <div className="card lg:col-span-2">
        <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Detection mode">
          {([
            ['broad', 'Everyday objects (365)'],
            ['package', 'Packages (trained)'],
            ['standard', customSelected ? `Backend model (${backendModel})` : 'COCO classes (80)'],
            ['world', 'Search by name'],
          ] as const).map(([id, label]) => (
            <button key={id} className={mode === id ? 'btn-primary' : 'btn-ghost'} aria-pressed={mode === id}
              disabled={busy} onClick={() => { setMode(id); setResult(null); setSource(null); setError('') }}>
              {id === 'world' && <Sparkles size={15} />}{label}
            </button>
          ))}
        </div>
        {mode === 'broad' && <p className="my-3 text-xs text-slate-400">Detects 365 categories in your browser, including lamps, shoes, signs and tools. The model downloads about 38 MB on first use.</p>}
        {mode === 'package' && <p className="my-3 text-xs text-slate-400">Finds packages using a detector fine-tuned on labeled package photos.</p>}
        {mode === 'world' && (
          <div className="mt-3">
            <label className="label">Objects to find (comma-separated, up to 20)</label>
            <input className="input" value={classesText} onChange={e => setClassesText(e.target.value)} placeholder="red bus, traffic light, dog" aria-label="Object names to detect" />
            <p className="mt-1 text-xs text-slate-500">YOLO-World searches for the objects you name using the Python backend. Use specific names such as “red bus” or “traffic light”.</p>
          </div>
        )}
        <label onDragOver={e => e.preventDefault()} onDrop={e => {
          e.preventDefault()
          pick(e.dataTransfer.files[0])
        }} className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-white/15 bg-white/[0.02] px-4 py-10 text-center hover:border-sky-500/50">
          <ImagePlus size={28} className="text-sky-400" />
          <span className="mt-2 font-semibold">Drop an image here, or click to browse</span>
          <span className="text-xs text-slate-400">JPG / PNG · up to 20 MB</span>
          <input type="file" accept="image/jpeg,image/png" className="hidden" disabled={busy} onChange={e => {
            pick(e.target.files?.[0])
            e.target.value = ''
          }} />
        </label>
        {error && <p className="mt-3 rounded-xl bg-red-500/10 border border-red-500/30 px-3 py-2 text-sm text-red-300">{error}</p>}
        {modelPct !== null && <p className="mt-3 rounded-xl bg-sky-500/10 border border-sky-500/30 px-3 py-2 text-sm text-sky-200">Downloading detection model… {modelPct}%</p>}
        {source === 'browser' && <p className="mt-3 rounded-xl bg-violet-500/10 border border-violet-500/30 px-3 py-2 text-sm text-violet-200">Detection ran in your browser. This result was not saved to backend history.</p>}
        {preview && (
          <div className="mt-4">
            <div className="grid gap-3 sm:grid-cols-2">
              <figure><figcaption className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-400">Original</figcaption><img src={preview} alt="Original upload" className="w-full rounded-xl border border-white/10" /></figure>
              <figure><figcaption className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-400">Annotated detections</figcaption>
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
              <p className="flex justify-between"><span className="text-slate-400">Detector</span><b>{mode === 'broad' ? 'YOLOE · Objects365' : mode === 'package' ? 'Package detector' : mode === 'world' ? 'YOLO-World' : source === 'backend' ? backendModel || 'Backend model' : 'YOLOv8n · COCO'}</b></p>
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
