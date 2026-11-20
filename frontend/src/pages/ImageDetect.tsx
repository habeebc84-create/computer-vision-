import { useEffect, useState } from 'react'
import { Download, ImagePlus, Loader2, Sparkles, Upload } from 'lucide-react'
import { api, DetectResult, detectWithFallback, parseDetectionPrompts, resolveUrl, toast } from '../lib/api'
// NOTE: browserDetection is dynamically imported only when the backend is
// offline, so the ~600 KB ONNX Runtime bundle stays out of the main chunk.
import { DetectionTable } from '../components/ui'

import { COCO_NAMES } from '../lib/yoloPost'

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
  const [result, setResult] = useState<DetectResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [mode, setMode] = useState<'coco' | 'world'>('coco')
  const [classesText, setClassesText] = useState('bus, person, traffic light')
  // Where the last result actually came from — never mislabelled.
  const [source, setSource] = useState<'backend' | 'browser' | null>(null)
  const [modelPct, setModelPct] = useState<number | null>(null)
  const [promptStatus, setPromptStatus] = useState('')

  // If the world model was chosen in Settings, start in Any-object mode.
  useEffect(() => {
    api.modelStatus().then(m => { if (m?.model_path?.includes('world')) setMode('world') }).catch(() => null)
  }, [])

  useEffect(() => {
    if (!file) return
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const pick = (f: File | undefined) => {
    if (!f || busy) return
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
    setBusy(true); setResult(null); setSource(null); setError(''); setModelPct(null); setPromptStatus('')
    try {
      const prompts = mode === 'world' ? parseDetectionPrompts(classesText) : null
      const progress = (loaded: number, total: number) => {
        if (total > 0) setModelPct(Math.round(loaded / total * 100))
      }
      const { result: r, source } = await detectWithFallback<DetectResult>(
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
      setSource(source)
      setResult(r); toast(`Detected ${r.count} object(s) in ${(r.inference_ms).toFixed(0)} ms`)
    }
    catch (e: any) { setError(e.message); toast(e.message, 'err') }
    finally { setBusy(false); setModelPct(null); setPromptStatus('') }
  }
  return (
    <>
    <div className="grid gap-4 xl:grid-cols-3">
      <div className="card xl:col-span-2">
        <div className="flex flex-wrap items-center gap-2" role="tablist" aria-label="Detection mode">
          <button className={mode === 'coco' ? 'btn-primary' : 'btn-ghost'} disabled={busy} onClick={() => setMode('coco')}>COCO classes (80)</button>
          <button className={mode === 'world' ? 'btn-primary' : 'btn-ghost'} disabled={busy} onClick={() => setMode('world')}><Sparkles size={15} /> Any object — type its name</button>
        </div>
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
          <span className="text-xs text-slate-400">JPG / PNG / WebP / GIF / BMP / AVIF · up to 20 MB · processed by the real YOLOv8 model</span>
          <input type="file" accept="image/jpeg,image/png,image/webp,image/gif,image/bmp,image/avif" className="hidden" disabled={busy} onChange={e => pick(e.target.files?.[0])} />
        </label>
        {error && <p className="mt-3 rounded-xl bg-red-500/10 border border-red-500/30 px-3 py-2 text-sm text-red-300">{error}</p>}
        {modelPct !== null && <p className="mt-3 rounded-xl bg-sky-500/10 border border-sky-500/30 px-3 py-2 text-sm text-sky-200">Downloading {mode === 'world' ? 'open-vocab' : 'detection'} model… {modelPct}% ({mode === 'world' ? '~205 MB' : '~12 MB'}, once only — it is cached in your browser after that).</p>}
        {busy && promptStatus && <p className="mt-3 rounded-xl bg-violet-500/10 border border-violet-500/30 px-3 py-2 text-sm text-violet-200">{promptStatus} Open-vocab checks one object at a time — each takes several seconds.</p>}
        {source === 'browser' && <p className="mt-3 rounded-xl bg-violet-500/10 border border-violet-500/30 px-3 py-2 text-sm text-violet-200">Running in your browser. Real {mode === 'world' ? 'Grounding DINO open-vocabulary' : 'YOLOv8'} inference on your device; browser results are not saved to history.</p>}
        {preview && (
          <div className="mt-4">
            <div className="grid gap-3 sm:grid-cols-2">
              <figure><figcaption className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-400">Original</figcaption><img src={preview} alt="Original upload" className="w-full rounded-xl border border-white/10" /></figure>
              <figure><figcaption className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-400">Annotated result</figcaption>
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
              <p className="flex justify-between"><span className="text-slate-400">Engine</span><b>{(result as any).engine || (source === 'browser' ? 'In-browser (ONNX)' : 'Backend (PyTorch)')}</b></p>
              <p className="flex justify-between"><span className="text-slate-400">History record</span><b>{result.record_id ? `#${result.record_id}` : result.history_error || '— not saved (in-browser)'}</b></p>
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
