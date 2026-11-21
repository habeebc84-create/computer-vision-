import { useEffect, useRef, useState } from 'react'
import { Camera, CameraOff, Download, Play, Sparkles, Square, Video } from 'lucide-react'
import { api, DetectResult, isBackendOnline, toast } from '../lib/api'
// NOTE: browserDetection is dynamically imported only when the backend is
// offline, so the ~600 KB ONNX Runtime bundle stays out of the main chunk.
import { DetectionTable } from '../components/ui'

export default function Live() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const stageRef = useRef<HTMLDivElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const timerRef = useRef<any>(null)
  const inFlightRef = useRef(false)
  // Aspect ratio of the live camera stream (set once the stream starts), so
  // the stage fits the camera instead of squeezing it into a fixed 16:9 box.
  const [streamAspect, setStreamAspect] = useState<number | null>(null)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<DetectResult | null>(null)
  const [fps, setFps] = useState(0)
  const [error, setError] = useState('')
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([])
  const [deviceId, setDeviceId] = useState('')
  // True while the browser is hiding camera names (no permission granted yet).
  const [labelsHidden, setLabelsHidden] = useState(false)
  const [enablingNames, setEnablingNames] = useState(false)
  const [busy, setBusy] = useState(false)
  const [mode, setMode] = useState<'coco' | 'world'>('coco')
  const [classesText, setClassesText] = useState('person, car, dog')
  const [source, setSource] = useState<'backend' | 'browser' | null>(null)
  const [modelPct, setModelPct] = useState<number | null>(null)
  const [promptStatus, setPromptStatus] = useState('')
  // Refs so the interval callback always reads the latest mode/prompts.
  const modeRef = useRef(mode); modeRef.current = mode
  const classesRef = useRef(classesText); classesRef.current = classesText

  useEffect(() => {
    navigator.mediaDevices?.enumerateDevices().then(ds => {
      const cams = ds.filter(d => d.kind === 'videoinput')
      setDevices(cams)
      // Browsers hide camera names until permission is granted, so before
      // that the dropdown can only show generic "Camera 1" entries.
      setLabelsHidden(cams.length > 0 && cams.every(d => !d.label))
    }).catch(() => null)
    api.modelStatus().then(m => { if (m?.model_path?.includes('world')) setMode('world') }).catch(() => null)
    return () => stop()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const draw = (dets: any[], w: number, h: number) => {
    // Boxes come back in original video pixels, but the video is shown with
    // object-contain inside the stage, so it is letterboxed whenever the
    // camera's aspect differs from the stage's. Drawing the boxes in video
    // pixels across the whole stage put them in the wrong place (or off the
    // picture entirely on portrait phone cameras) — so map every box onto
    // the video's actual displayed rectangle inside the stage.
    const c = canvasRef.current; const v = videoRef.current; const stage = stageRef.current
    if (!c || !v || !stage) return
    const cw = stage.clientWidth; const ch = stage.clientHeight
    if (!cw || !ch || !w || !h) return
    const dpr = window.devicePixelRatio || 1
    c.width = Math.round(cw * dpr); c.height = Math.round(ch * dpr)
    const ctx = c.getContext('2d'); if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, cw, ch)
    const scale = Math.min(cw / w, ch / h)
    const dw = w * scale; const dh = h * scale
    const ox = (cw - dw) / 2; const oy = (ch - dh) / 2
    const colors = ['#38bdf8', '#22d3ee', '#a78bfa', '#34d399', '#fb923c', '#f472b6', '#facc15']
    dets.forEach((d, i) => {
      const [x1, y1, x2, y2] = d.bbox
      const color = colors[d.class_id % colors.length]
      const bx = ox + x1 * scale; const by = oy + y1 * scale
      const bw = (x2 - x1) * scale; const bh = (y2 - y1) * scale
      ctx.strokeStyle = color; ctx.lineWidth = Math.max(2, 3 * scale); ctx.strokeRect(bx, by, bw, bh)
      const label = `${d.class_name} ${(d.confidence * 100).toFixed(0)}%`
      const fs = Math.max(13, Math.round(15 * scale))
      ctx.font = `600 ${fs}px Inter, sans-serif`
      const tw = ctx.measureText(label).width
      const fh = fs + 8
      const ly = Math.max(0, by - fh)
      ctx.fillStyle = color; ctx.fillRect(bx, ly, tw + 10, fh)
      ctx.fillStyle = '#050914'; ctx.fillText(label, bx + 5, ly + fs - 1)
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
        const prompts = [...new Set(classesRef.current.split(',').map(s => s.trim()).filter(Boolean))]
        if (prompts.length === 0) throw new Error('Type at least one object name to detect, e.g. "person, dog".')
        if (await isBackendOnline()) {
          // YOLO-World on the backend when it is connected.
          const off = document.createElement('canvas')
          off.width = v.videoWidth; off.height = v.videoHeight
          off.getContext('2d')!.drawImage(v, 0, 0)
          const blob = await new Promise<Blob | null>(r => off.toBlob(r, 'image/jpeg', 0.8))
          if (!blob) return
          res = await api.openVocabFrame(blob, prompts)
          setSource('backend')
        } else {
          // No backend: real Grounding DINO open-vocabulary inference in the browser.
          // Slow (~1 frame every few seconds on CPU) — the in-flight guard above
          // keeps frames from stacking while one is still being processed.
          const { detectOpenVocabInBrowser } = await import('../lib/browserOpenVocab')
          const off = document.createElement('canvas')
          off.width = v.videoWidth; off.height = v.videoHeight
          res = await detectOpenVocabInBrowser(v, v.videoWidth, v.videoHeight, prompts, (loaded, total) => {
            if (total > 0) setModelPct(Math.round((loaded / total) * 100))
          }, (done, total, prompt) => setPromptStatus(`Looking for “${prompt}”… (${done} of ${total})`))
          setModelPct(null)
          setSource('browser')
        }
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

  const enableCameraNames = async () => {
    // Ask for camera permission once, only to reveal real camera names —
    // never on page load. The temporary stream is stopped immediately and
    // the devices are re-enumerated so the dropdown shows real names.
    setError(''); setEnablingNames(true)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      stream.getTracks().forEach(t => t.stop())
      const ds = await navigator.mediaDevices.enumerateDevices()
      const cams = ds.filter(d => d.kind === 'videoinput')
      setDevices(cams)
      setLabelsHidden(cams.length > 0 && cams.every(d => !d.label))
    } catch (e: any) {
      setError(e?.name === 'NotAllowedError' ? 'Camera permission was denied, so camera names stay hidden. Allow camera access in the browser address bar and try again.' : `Could not access the camera: ${e?.message || e}`)
    } finally { setEnablingNames(false) }
  }

  const start = async () => {
    setError(''); setBusy(true)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: deviceId ? { deviceId: { exact: deviceId } } : true, audio: false })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
        if (videoRef.current.videoWidth > 0) setStreamAspect(videoRef.current.videoWidth / videoRef.current.videoHeight)
      }
      setRunning(true)
      timerRef.current = setInterval(captureLoop, 160) // ~6 fps, throttled for CPU inference
    } catch (e: any) {
      setError(e?.name === 'NotAllowedError'
        ? 'Camera access was denied by the browser. Allow camera access, or use Image detection instead.'
        : e?.name === 'NotFoundError'
          ? 'No camera was found on this device. Connect a camera and try again, or use Image detection instead.'
          : e?.name === 'OverconstrainedError'
            ? 'The selected camera could not be opened. Try “Default camera” instead, or use Image detection.'
            : `Could not start webcam: ${e?.message || e}`)
    } finally { setBusy(false) }
  }
  const stop = () => {
    if (timerRef.current) clearInterval(timerRef.current)
    streamRef.current?.getTracks().forEach(t => t.stop()); streamRef.current = null
    setRunning(false)
    setStreamAspect(null)
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
    <div className="space-y-4">
      <div className="card">
        <div className="flex flex-wrap items-center gap-2">
          <select className="input max-w-xs" value={deviceId} onChange={e => setDeviceId(e.target.value)} aria-label="Camera device">
            <option value="">Default camera</option>
            {devices.map((d, i) => <option key={d.deviceId} value={d.deviceId}>{d.label || `Camera ${i + 1}`}</option>)}
          </select>
          {!running
            ? <button className="btn-primary" onClick={start} disabled={busy}><Play size={15} /> {busy ? 'Starting…' : 'Start webcam'}</button>
            : <button className="btn-danger" onClick={stop}><Square size={15} /> Stop webcam</button>}
          <button className="btn-ghost" onClick={saveSnapshot} disabled={!running}><Download size={15} /> Save snapshot</button>
          <span className={`ml-auto chip ${running ? '' : '!bg-white/5 !text-slate-400 !border-white/10'}`}>{running ? '● LIVE' : '○ STOPPED'}</span>
        </div>
        {labelsHidden && !running && (
          <p className="mt-2 rounded-xl bg-amber-500/10 border border-amber-500/30 px-3 py-2 text-sm text-amber-200">
            Camera names are hidden — browsers do not reveal them until you allow camera access, so this list only shows generic names like “Camera 1”.{' '}
            <button className="font-semibold underline" onClick={enableCameraNames} disabled={enablingNames}>{enablingNames ? 'Asking…' : 'Enable camera names'}</button>
          </p>
        )}
        <p className="mt-2 text-xs text-slate-500">The webcam only works on localhost or a secure (HTTPS) page. If another app is already using the camera, starting it here will fail — close the other app and try again.</p>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <button className={mode === 'coco' ? 'btn-primary' : 'btn-ghost'} onClick={() => setMode('coco')}>COCO classes (80)</button>
          <button className={mode === 'world' ? 'btn-primary' : 'btn-ghost'} onClick={() => setMode('world')}><Sparkles size={15} /> Any object — type its name</button>
        </div>
        {mode === 'world' && (
          <div className="mt-3">
            <label className="label">Objects to find (comma-separated, up to 20)</label>
            <input className="input" value={classesText} onChange={e => setClassesText(e.target.value)} placeholder="person, car, dog" aria-label="Object names to detect" />
            <p className="mt-1 text-xs text-slate-500">Detects anything you name. With the backend connected this uses YOLO-World (~25 MB). Without a backend it runs Grounding DINO in your browser instead (~205 MB download once, slower, experimental — each object checked takes several seconds, so live video is snapshots, not smooth video).</p>
            <div className="mt-2 flex flex-wrap gap-1.5" aria-label="Try an object outside the COCO list">
              {['microphone', 'sunglasses', 'headphones', 'traffic cone', 'whiteboard', 'coffee mug'].map(p => (
                <button key={p} type="button" className="rounded-full border border-violet-500/40 bg-violet-500/10 px-2.5 py-1 text-xs text-violet-200 hover:bg-violet-500/20" onClick={() => setClassesText(p)} title="Try this non-COCO object">{p}</button>
              ))}
            </div>
          </div>
        )}
        {error && <p className="mt-3 rounded-xl bg-red-500/10 border border-red-500/30 px-3 py-2 text-sm text-red-300">{error}</p>}
        {modelPct !== null && <p className="mt-3 rounded-xl bg-sky-500/10 border border-sky-500/30 px-3 py-2 text-sm text-sky-200">Downloading {mode === 'world' ? 'open-vocab' : 'detection'} model… {modelPct}% ({mode === 'world' ? '~205 MB' : '~12 MB'}, once only — it is cached in your browser after that).</p>}
        {running && promptStatus && <p className="mt-3 rounded-xl bg-violet-500/10 border border-violet-500/30 px-3 py-2 text-sm text-violet-200">{promptStatus} Open-vocab checks one object at a time — each takes several seconds.</p>}
        {source === 'browser' && <p className="mt-3 rounded-xl bg-violet-500/10 border border-violet-500/30 px-3 py-2 text-sm text-violet-200">Running in your browser — no backend connected. Real {mode === 'world' ? 'Grounding DINO open-vocabulary' : 'YOLOv8'} inference on your device; frames never leave this browser and results are not saved to history.</p>}
        <div ref={stageRef} className="relative mt-4 max-h-[75vh] w-full overflow-hidden rounded-xl border border-white/10 bg-black aspect-video" style={streamAspect ? { aspectRatio: String(streamAspect) } : undefined}>
          <video ref={videoRef} playsInline muted className="h-full w-full object-contain" />
          <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
          {!running && <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-slate-400"><CameraOff size={36} /><p className="text-sm">Webcam is off — press “Start webcam” to begin real-time detection.</p><p className="text-xs text-slate-500">Your camera stays in this browser; frames go to the backend when it is connected, otherwise they are detected right here in the browser.</p></div>}
        </div>
        <p className="mt-2 flex items-center gap-2 text-xs text-slate-500"><Video size={13} /> Frames are captured at ~6 fps. Boxes are drawn on the video itself — if nothing is boxed, nothing was recognised in that frame. No fake stream is ever shown.</p>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        <div className="card">
          <h2 className="font-semibold">Live statistics</h2>
          <div className="mt-3 space-y-2 text-sm">
            <p className="flex justify-between"><span className="text-slate-400">Objects in frame</span><b>{result?.count ?? 0}</b></p>
            <p className="flex justify-between"><span className="text-slate-400">Unique classes</span><b>{result?.unique_classes.length ?? 0}</b></p>
            <p className="flex justify-between"><span className="text-slate-400">Avg confidence</span><b>{result ? (result.avg_confidence * 100).toFixed(1) + '%' : '—'}</b></p>
            <p className="flex justify-between"><span className="text-slate-400">Inference time</span><b>{result ? result.inference_ms.toFixed(0) + ' ms' : '—'}</b></p>
            <p className="flex justify-between"><span className="text-slate-400">Measured FPS</span><b>{fps ? fps.toFixed(1) : '—'}</b></p>
            <p className="flex justify-between"><span className="text-slate-400">Engine</span><b>{(result as any)?.engine || (source === 'browser' ? 'In-browser (ONNX)' : source === 'backend' ? 'Backend (PyTorch)' : '—')}</b></p>
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
