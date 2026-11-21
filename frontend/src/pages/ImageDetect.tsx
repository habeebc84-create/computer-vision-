import { useEffect, useState } from 'react'
import { Download, ImagePlus, Loader2, Sparkles, Upload } from 'lucide-react'
import { api, isBackendOnline, DetectResult, detectWithFallback, parseDetectionPrompts, resolveUrl, toast } from '../lib/api'
import { saveBrowserHistory } from '../lib/browserHistory'
import type { BrowserDetectResult } from '../lib/browserDetection'
import { DetectionTable } from '../components/ui'

import { COCO_NAMES } from '../lib/yoloPost'

type Mode = 'broad' | 'expanded' | 'package' | 'coco' | 'world'
type ImageResult = DetectResult & Partial<Pick<BrowserDetectResult, 'annotatedDataUrl' | 'engine'>> & { prompts?: string[] }

const COCO_GROUPS: { title: string; items: string[] }[] = [
  { title: 'People', items: ['person'] },
  { title: 'Vehicles', items: ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat'] },
  { title: 'Street', items: ['traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench'] },
  { title: 'Animals', items: ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'] },
  { title: 'Accessories', items: ['backpack', 'umbrella', 'handbag', 'tie', 'suitcase'] },
  { title: 'Sports', items: ['frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket'] },
  { title: 'Kitchen', items: ['bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl'] },
  { title: 'Food', items: ['banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake'] },
  { title: 'Home', items: ['chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet'] },
  { title: 'Electronics', items: ['tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone'] },
  { title: 'Appliances', items: ['microwave', 'oven', 'toaster', 'sink', 'refrigerator'] },
  { title: 'Everyday', items: ['book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'] },
]

function WhatCanItRecognise() {
  return (
    <div className="card mt-4">
      <h2 className="font-semibold">What can it recognise?</h2>
      <p className="mt-2 text-sm text-slate-400">
        In <b className="text-slate-200">COCO mode</b> the fast YOLOv8 model recognises these {COCO_NAMES.length} everyday
        classes — no typing needed. Anything else, switch to <b className="text-slate-200">“Any object — type its name”</b> and
        describe it in your own words: with the backend connected that uses YOLO-World; without a backend it runs
        Grounding DINO in your browser instead. Try “microphone”, “sunglasses” or “traffic cone” — none of those are in the COCO list.
      </p>
      <div className="mt-3 space-y-3">
        {COCO_GROUPS.map(g => (
          <div key={g.title}>
            <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">{g.title} · {g.items.length}</p>
            <div className="mt-1 flex flex-wrap gap-1.5">
              {g.items.map(c => <span key={c} className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-xs text-slate-300">{c}</span>)}
            </div>
          </div>
        ))}
      </div>
      <div className="mt-4 rounded-xl border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-200">
        <b>Honest limits.</b> Open-vocabulary mode is slower and experimental: it can confuse lookalike objects
        (“cup” vs “bowl”), needs a specific phrase (“red bus” beats “thing”), and tiny objects at the edge of a photo
        can still be missed — the same edge-object limit as the COCO path. In the browser it downloads a ~205 MB model
        once and checks one object at a time, each taking several seconds — best for photos, not smooth video.
      </div>
    </div>
  )
}

export default function ImageDetect() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [result, setResult] = useState<ImageResult | null>(null)
  const [historyStatus, setHistoryStatus] = useState('')
  const [scanSmallObjects, setScanSmallObjects] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [chosenMode, setMode] = useState<Mode | null>(null)
  const [backendModel, setBackendModel] = useState('')
  const mode = chosenMode ?? (backendModel.includes('world') ? 'world' : backendModel && backendModel !== 'yolov8n.pt' ? 'coco' : 'broad')
  const customSelected = Boolean(backendModel) && !/^yolov8[nsmlx]\.pt$/.test(backendModel)
  const [classesText, setClassesText] = useState('bus, person, traffic light')
  // Where the last result actually came from — never mislabelled.
  const [source, setSource] = useState<'backend' | 'browser' | null>(null)
  const [modelPct, setModelPct] = useState<number | null>(null)
  const [promptStatus, setPromptStatus] = useState('')

  useEffect(() => {
    let mounted = true
    api.modelStatus().then(status => {
      if (mounted) setBackendModel(status.model_path?.split(/[/\\]/).pop() || '')
    }).catch(() => null)
    return () => { mounted = false }
  }, [])

  useEffect(() => {
    if (!file) { setPreview(''); return }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const pick = (f: File | undefined) => {
    if (!f || busy) return
    setFile(null); setResult(null); setSource(null)
    // Phones often save HEIC/HEIF, which browsers (except Safari) cannot
    // decode — say so plainly instead of the generic format error.
    if (/heic|heif/i.test(f.type) || /\.hei[cf]$/i.test(f.name)) {
      setError('This photo is HEIC (the iPhone format) — browsers cannot read it here. Please convert it to JPG and try again.')
      return
    }
    if (f.size > 20 * 1024 * 1024) { setError('Choose an image up to 20 MB.'); return }
    if (!/^image\/(jpeg|png|webp|gif|bmp|avif)$/.test(f.type) && !/\.(jpe?g|png|webp|gif|bmp|avif)$/i.test(f.name)) { setError('Please choose a JPG, PNG, WebP, GIF, BMP or AVIF image.'); return }
    setFile(f); setResult(null); setError(''); setSource(null)
  }
  const run = async () => {
    if (!file) return
    setMode(mode); setBusy(true); setResult(null); setSource(null); setError(''); setModelPct(null); setPromptStatus(''); setHistoryStatus('')
    try {
      const prompts = mode === 'world' ? parseDetectionPrompts(classesText) : null
      const progress = (loaded: number, total: number) => {
        if (total > 0) setModelPct(Math.min(100, Math.round(loaded / total * 100)))
      }
      const runDetector = async (): Promise<{ result: ImageResult; source: 'browser' | 'backend' }> => {
        if (mode === 'broad' || mode === 'expanded' || mode === 'package') {
          const { detectImageFileInBrowser } = await import('../lib/browserDetection')
          return { result: await detectImageFileInBrowser(file, progress, mode, mode === 'broad' && scanSmallObjects), source: 'browser' as const }
        }
        if (mode === 'coco' && customSelected) {
          if (!(await isBackendOnline())) throw new Error(`Start the Python backend to use ${backendModel}.`)
          return { result: await api.detectImage(file), source: 'backend' as const }
        }
        return detectWithFallback<ImageResult>(
          () => prompts ? api.openVocabImage(file, prompts) : api.detectImage(file),
          async () => {
            if (prompts) {
              const { detectOpenVocabFileInBrowser } = await import('../lib/browserOpenVocab')
              return detectOpenVocabFileInBrowser(file, prompts, progress,
                (done, total, prompt) => setPromptStatus(`Checking “${prompt}” (${done} of ${total})`))
            }
            const { detectImageFileInBrowser } = await import('../lib/browserDetection')
            return detectImageFileInBrowser(file, progress)
          },
        )
      }
      const { result: r, source } = await runDetector()
      setSource(source)
      setResult(r); toast(`Detected ${r.count} object(s) in ${(r.inference_ms).toFixed(0)} ms`)
      if (source === 'browser') {
        setHistoryStatus('Saving…')
        try {
          await saveBrowserHistory({
            inputType: mode === 'world' ? 'open-vocab' : 'image',
            count: r.count,
            classes: r.unique_classes,
            avgConfidence: r.avg_confidence,
            inferenceMs: r.inference_ms,
            fps: r.fps,
            model: r.engine || 'In-browser (ONNX)',
            name: file.name,
            previewDataUrl: r.annotatedDataUrl,
          })
          setHistoryStatus('Saved in this browser')
        } catch {
          setHistoryStatus('Could not save in this browser')
        }
      }
    }
    catch (e: any) { setError(e.message); toast(e.message, 'err') }
    finally { setBusy(false); setModelPct(null); setPromptStatus('') }
  }
  const annotatedSrc = result?.annotatedDataUrl || (result?.annotated_url ? resolveUrl(result.annotated_url) : '')
  return (
    <>
    <div className="grid gap-4 xl:grid-cols-3">
      <div className="card xl:col-span-2">
        <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Detection mode">
          {([
            ['broad', 'Everyday objects (365)'],
            ['expanded', 'Expanded objects (4,585)'],
            ['package', 'Packages (trained)'],
            ['coco', customSelected ? `Backend model (${backendModel})` : 'COCO classes (80)'],
            ['world', 'Search by name'],
          ] as const).map(([id, label]) => (
            <button key={id} className={mode === id ? 'btn-primary' : 'btn-ghost'} aria-pressed={mode === id}
              disabled={busy} onClick={() => { setMode(id); setResult(null); setSource(null); setError('') }}>
              {id === 'world' && <Sparkles size={15} />}{label}
            </button>
          ))}
        </div>
        {mode === 'broad' && <p className="my-3 text-sm text-slate-400">Search for 365 categories in your browser, including lamps, shoes, signs and tools.</p>}
        {mode === 'broad' && (
          <label className="mb-3 flex items-start gap-2 text-sm text-slate-300">
            <input type="checkbox" className="mt-0.5 h-4 w-4 accent-indigo-400" checked={scanSmallObjects}
              disabled={busy} onChange={event => setScanSmallObjects(event.target.checked)} />
            <span>Look for small objects <span className="text-slate-400">(slower on large photos)</span></span>
          </label>
        )}
        {mode === 'expanded' && <p className="my-3 text-sm text-slate-400">Search a larger vocabulary with YOLOE. The first run downloads about 125 MB and can take longer on a phone.</p>}
        {mode === 'package' && <p className="my-3 text-sm text-slate-400">Find packages using a detector fine-tuned on labeled package photos.</p>}
        {mode === 'world' && (
          <div className="mt-3">
            <label className="label">Objects to find (comma-separated, up to 20)</label>
            <input className="input" value={classesText} onChange={e => setClassesText(e.target.value)} placeholder="red bus, traffic light, dog" aria-label="Object names to detect" />
            <p className="mt-1 text-xs text-slate-500">Search for objects outside the 80 COCO classes using specific names such as “red bus”. With the backend connected this uses YOLO-World (~25 MB). Without a backend it runs Grounding DINO in your browser instead (~205 MB download once, slower, experimental).</p>
            <div className="mt-2 flex flex-wrap gap-1.5" aria-label="Try an object outside the COCO list">
              {['microphone', 'sunglasses', 'headphones', 'traffic cone', 'whiteboard', 'coffee mug'].map(p => (
                <button key={p} type="button" className="rounded-full border border-violet-500/40 bg-violet-500/10 px-2.5 py-1 text-xs text-violet-200 hover:bg-violet-500/20" onClick={() => setClassesText(p)} title="Try this non-COCO object">{p}</button>
              ))}
            </div>
          </div>
        )}
        <label onDragOver={event => event.preventDefault()} onDrop={event => {
          event.preventDefault()
          pick(event.dataTransfer.files[0])
        }} className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-white/15 bg-white/[0.02] px-4 py-10 text-center hover:border-sky-500/50">
          <ImagePlus size={28} className="text-sky-400" />
          <span className="mt-2 font-semibold">Drop an image here, or click to browse</span>
          <span className="text-xs text-slate-400">JPG / PNG / WebP / GIF / BMP / AVIF · up to 20 MB</span>
          <input type="file" accept="image/jpeg,image/png,image/webp,image/gif,image/bmp,image/avif" className="hidden" disabled={busy} onChange={e => { pick(e.target.files?.[0]); e.target.value = '' }} />
        </label>
        {error && <p className="mt-3 rounded-xl bg-red-500/10 border border-red-500/30 px-3 py-2 text-sm text-red-300">{error}</p>}
        {modelPct !== null && <p role="status" className="mt-3 rounded-xl border border-sky-500/30 bg-sky-500/10 px-3 py-2 text-sm text-sky-200">Downloading detection model… {modelPct}%</p>}
        {busy && promptStatus && <p className="mt-3 rounded-xl bg-violet-500/10 border border-violet-500/30 px-3 py-2 text-sm text-violet-200">{promptStatus} Open-vocab checks one object at a time — each takes several seconds.</p>}
        {source === 'browser' && <p className="mt-3 rounded-xl border border-violet-500/30 bg-violet-500/10 px-3 py-2 text-sm text-violet-200">Detection ran in your browser.</p>}
        {preview && (
          <div className="mt-4">
            <div className="grid gap-3 sm:grid-cols-2">
              <figure><figcaption className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-400">Original</figcaption><img src={preview} alt="Original upload" className="w-full rounded-xl border border-white/10" /></figure>
              <figure><figcaption className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-400">Annotated result</figcaption>
                {annotatedSrc ? <img src={annotatedSrc} alt="Annotated detection result" className="w-full rounded-xl border border-white/10" /> : <div className="flex h-full min-h-[180px] items-center justify-center rounded-xl border border-white/10 text-sm text-slate-500">Run detection to see bounding boxes</div>}
              </figure>
            </div>
            <div className="mt-4 flex flex-wrap gap-2">
              <button className="btn-primary" onClick={run} disabled={busy || !file}>{busy ? <><Loader2 size={15} className="animate-spin" /> Detecting…</> : <><Upload size={15} /> Run detection</>}</button>
              {annotatedSrc && <a className="btn-ghost" href={annotatedSrc} download="annotated.jpg"><Download size={15} /> Download annotated image</a>}
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
              {result.count === 0 && mode === 'broad' && !scanSmallObjects && (
                <p className="text-slate-300">No objects met the confidence threshold. For small or distant objects, try enabling “Look for small objects”.</p>
              )}
              {result.prompts && <p className="flex justify-between"><span className="text-slate-400">Looking for</span><b className="text-right">{result.prompts.join(', ')}</b></p>}
              <p className="flex justify-between"><span className="text-slate-400">Unique classes</span><b>{result.unique_classes.join(', ') || '—'}</b></p>
              <p className="flex justify-between"><span className="text-slate-400">Avg confidence</span><b>{(result.avg_confidence * 100).toFixed(1)}%</b></p>
              <p className="flex justify-between"><span className="text-slate-400">Inference time</span><b>{result.inference_ms.toFixed(0)} ms</b></p>
              <p className="flex justify-between"><span className="text-slate-400">FPS (equivalent)</span><b>{result.fps.toFixed(1)}</b></p>
              <p className="flex justify-between"><span className="text-slate-400">Engine</span><b>{result.engine || (source === 'browser' ? 'In-browser (ONNX)' : 'Backend (PyTorch)')}</b></p>
              <p className="flex justify-between" role="status"><span className="text-slate-400">History record</span><b>{source === 'browser' ? historyStatus : result.record_id ? `#${result.record_id}` : result.history_error || '—'}</b></p>
            </div>
          ) : <p className="mt-3 text-sm text-slate-400">Upload an image and run detection — results and statistics appear here.</p>}
        </div>
        <div className="card"><h2 className="font-semibold">Detected objects</h2><div className="mt-3">{result ? <DetectionTable dets={result.detections} /> : <p className="text-sm text-slate-400">No detection run yet.</p>}</div></div>
      </div>
    </div>
    <WhatCanItRecognise />
    </>
  )
}
