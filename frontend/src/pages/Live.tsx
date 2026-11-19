import { useEffect, useRef, useState } from 'react'
import { Camera, CameraOff, Download, Play, Sparkles, Square, Video } from 'lucide-react'
import { api, DetectResult, isBackendOnline, toast } from '../lib/api'
// NOTE: browserDetection is dynamically imported only when the backend is
// offline, so the ~600 KB ONNX Runtime bundle stays out of the main chunk.
import { DetectionTable } from '../components/ui'

export default function Live() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const timerRef = useRef<any>(null)
  const inFlightRef = useRef(false)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<DetectResult | null>(null)
  const [fps, setFps] = useState(0)
  const [error, setError] = useState('')
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([])
  const [deviceId, setDeviceId] = useState('')
  const [busy, setBusy] = useState(false)
  const [mode, setMode] = useState<'coco' | 'world'>('coco')
  const [classesText, setClassesText] = useState('person, car, dog')
  const [source, setSource] = useState<'backend' | 'browser' | null>(null)
  const [modelPct, setModelPct] = useState<number | null>(null)
  // Refs so the interval callback always reads the latest mode/prompts.
  const modeRef = useRef(mode); modeRef.current = mode
  const classesRef = useRef(classesText); classesRef.current = classesText

  useEffect(() => {
    navigator.mediaDevices?.enumerateDevices().then(ds => setDevices(ds.filter(d => d.kind === 'videoinput'))).catch(() => null)
    api.modelStatus().then(m => { if (m?.model_path?.includes('world')) setMode('world') }).catch(() => null)
    return () => stop()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const draw = (dets: any[], w: number, h: number) => {
    const c = canvasRef.current; if (!c) return
    c.width = w; c.height = h
    const ctx = c.getContext('2d'); if (!ctx) return
    ctx.clearRect(0, 0, w, h)
    const colors = ['#38bdf8', '#22d3ee', '#a78bfa', '#34d399', '#fb923c', '#f472b6', '#facc15']
    dets.forEach((d, i) => {
      const [x1, y1, x2, y2] = d.bbox
      const color = colors[d.class_id % colors.length]
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
      const label = `${d.class_name} ${(d.confidence * 100).toFixed(0)}%`
      ctx.font = '600 13px Inter, sans-serif'
      const tw = ctx.measureText(label).width
      ctx.fillStyle = color; ctx.fillRect(x1, y1 - 20, tw + 10, 20)
      ctx.fillStyle = '#050914'; ctx.fillText(label, x1 + 5, y1 - 6)
    })
  }

  const captureLoop = async () => {
    // In-flight guard: never stack overlapping inference requests — if the
    // previous frame is still being processed (slow CPU inference), skip
    // this tick instead of queuing another POST.
    if (inFlightRef.current) return
    const v = videoRef.current; if (!v || v.readyState < 2) return
    inFlightRef.current = true
    try {
      let res: DetectResult
      if (modeRef.current === 'world') {
        // YOLO-World only runs on the backend — there is no browser build of it.
        if (!(await isBackendOnline())) {
          setError('Any-object (YOLO-World) detection needs the Python backend — it cannot run in the browser. Switch to COCO mode to detect in-browser, or start the backend with python run.py.')
          return
        }
        const off = document.createElement('canvas')
        off.width = v.videoWidth; off.height = v.videoHeight
        off.getContext('2d')!.drawImage(v, 0, 0)
        const blob = await new Promise<Blob | null>(r => off.toBlob(r, 'image/jpeg', 0.8))
        if (!blob) return
        const prompts = [...new Set(classesRef.current.split(',').map(s => s.trim()).filter(Boolean))]
        if (prompts.length === 0) throw new Error('Type at least one object name to detect, e.g. "person, dog".')
        res = await api.openVocabFrame(blob, prompts)
        setSource('backend')
      } else if (await isBackendOnline()) {
        const off = document.createElement('canvas')
        off.width = v.videoWidth; off.height = v.videoHeight
        off.getContext('2d')!.drawImage(v, 0, 0)
        const blob = await new Promise<Blob | null>(r => off.toBlob(r, 'image/jpeg', 0.8))
        if (!blob) return
        res = await api.detectFrame(blob)
        setSource('backend')
      } else {
        // Backend unreachable: real YOLOv8 inference right here in the browser.
        const { detectInBrowser } = await import('../lib/browserDetection')
        res = await detectInBrowser(v, v.videoWidth, v.videoHeight, (loaded, total) => {
          if (total > 0) setModelPct(Math.round((loaded / total) * 100))
        })
        setModelPct(null)
        setSource('browser')
      }
      setResult(res); setFps(res.fps)
      draw(res.detections, v.videoWidth, v.videoHeight)
    } catch (e: any) { setError(e.message || 'Frame inference failed.') }
    finally { inFlightRef.current = false; setModelPct(null) }
  }

  const start = async () => {
    setError(''); setBusy(true)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: deviceId ? { deviceId: { exact: deviceId } } : true, audio: false })
      streamRef.current = stream
      if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play() }
      setRunning(true)
      timerRef.current = setInterval(captureLoop, 160) // ~6 fps, throttled for CPU inference
    } catch (e: any) {
      setError(e?.name === 'NotAllowedError' ? 'Camera access was denied by the browser. Allow camera access, or use Image / Video detection instead.' : `Could not start webcam: ${e?.message || e}`)
    } finally { setBusy(false) }
  }
  const stop = () => {
    if (timerRef.current) clearInterval(timerRef.current)
    streamRef.current?.getTracks().forEach(t => t.stop()); streamRef.current = null
    setRunning(false)
  }
  const saveSnapshot = async () => {
    if (source === 'browser') { toast('Snapshots are saved to history only when the backend is connected — in-browser results are not stored.', 'err'); return }
    const v = videoRef.current; if (!v) return
    const off = document.createElement('canvas'); off.width = v.videoWidth; off.height = v.videoHeight
    off.getContext('2d')!.drawImage(v, 0, 0)
    const blob = await new Promise<Blob | null>(r => off.toBlob(r, 'image/jpeg', 0.9))
    if (!blob) return
    try { const r = await api.saveFrame(blob); toast(`Snapshot saved — ${r.count} object(s), record #${r.record_id}`) }
    catch (e: any) { toast(e.message, 'err') }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <div className="card lg:col-span-2">
        <div className="flex flex-wrap items-center gap-2">
          <select className="input max-w-xs" value={deviceId} onChange={e => setDeviceId(e.target.value)} aria-label="Camera device">
            <option value="">Default camera</option>
            {devices.map((d, i) => <option key={d.deviceId} value={d.deviceId}>{d.label || `Camera ${i + 1}`}</option>)}
          </select>
          {!running
            ? <button className="btn-primary" onClick={start} disabled={busy}><Play size={15} /> {busy ? 'Starting…' : 'Start webcam'}</button>
            : <button className="btn-danger" onClick={stop}><Square size={15} /> Stop webcam</button>}
          <button className="btn-ghost" onClick={saveSnapshot} disabled={!running}><Download size={15} /> Save snapshot</button>
          <span className={`ml-auto chip ${running ? '' : '!bg-white/5 !text-slate-400 !border-white/10'}`}>{running ? '● LIVE' : '○ OFFLINE'}</span>
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <button className={mode === 'coco' ? 'btn-primary' : 'btn-ghost'} onClick={() => setMode('coco')}>COCO classes (80)</button>
          <button className={mode === 'world' ? 'btn-primary' : 'btn-ghost'} onClick={() => setMode('world')}><Sparkles size={15} /> Any object — type its name</button>
        </div>
        {mode === 'world' && (
          <div className="mt-3">
            <label className="label">Objects to find (comma-separated, up to 20)</label>
            <input className="input" value={classesText} onChange={e => setClassesText(e.target.value)} placeholder="person, car, dog" aria-label="Object names to detect" />
            <p className="mt-1 text-xs text-slate-500">YOLO-World searches for the objects you name. First use downloads the detector (~25 MB) and its text encoder (~338 MiB).</p>
          </div>
        )}
        {error && <p className="mt-3 rounded-xl bg-red-500/10 border border-red-500/30 px-3 py-2 text-sm text-red-300">{error}</p>}
        {modelPct !== null && <p className="mt-3 rounded-xl bg-sky-500/10 border border-sky-500/30 px-3 py-2 text-sm text-sky-200">Downloading detection model… {modelPct}% (~12 MB, once only — it is cached in your browser after that).</p>}
        {source === 'browser' && <p className="mt-3 rounded-xl bg-violet-500/10 border border-violet-500/30 px-3 py-2 text-sm text-violet-200">Running in your browser — no backend connected. Real YOLOv8 inference on your device; frames never leave this browser and results are not saved to history.</p>}
        <div className="relative mt-4 overflow-hidden rounded-xl border border-white/10 bg-black aspect-video">
          <video ref={videoRef} playsInline muted className="h-full w-full object-contain" />
          <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
          {!running && <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-slate-400"><CameraOff size={36} /><p className="text-sm">Webcam is off — press “Start webcam” to begin real-time detection.</p><p className="text-xs text-slate-500">Your camera stays in this browser; frames go to the backend when it is connected, otherwise they are detected right here in the browser.</p></div>}
        </div>
        <p className="mt-2 flex items-center gap-2 text-xs text-slate-500"><Video size={13} /> Frames are captured at ~6 fps. No fake stream is ever shown.</p>
      </div>
      <div className="space-y-4">
        <div className="card">
          <h2 className="font-semibold">Live statistics</h2>
          <div className="mt-3 space-y-2 text-sm">
            <p className="flex justify-between"><span className="text-slate-400">Objects in frame</span><b>{result?.count ?? 0}</b></p>
            <p className="flex justify-between"><span className="text-slate-400">Unique classes</span><b>{result?.unique_classes.length ?? 0}</b></p>
            <p className="flex justify-between"><span className="text-slate-400">Avg confidence</span><b>{result ? (result.avg_confidence * 100).toFixed(1) + '%' : '—'}</b></p>
            <p className="flex justify-between"><span className="text-slate-400">Inference time</span><b>{result ? result.inference_ms.toFixed(0) + ' ms' : '—'}</b></p>
            <p className="flex justify-between"><span className="text-slate-400">Measured FPS</span><b>{fps ? fps.toFixed(1) : '—'}</b></p>
            <p className="flex justify-between"><span className="text-slate-400">Engine</span><b>{source === 'browser' ? 'In-browser (ONNX)' : source === 'backend' ? 'Backend (PyTorch)' : '—'}</b></p>
          </div>
        </div>
        <div className="card">
          <h2 className="font-semibold">Detections in current frame</h2>
          <div className="mt-3">{result ? <DetectionTable dets={result.detections} /> : <p className="text-sm text-slate-400">Start the webcam to see live detections here.</p>}</div>
        </div>
      </div>
    </div>
  )
}
