import { useEffect, useRef, useState } from 'react'
import { CameraOff, Download, Play, Sparkles, Square, Video } from 'lucide-react'
import { api, DetectResult, detectWithFallback, isBackendOnline, parseDetectionPrompts, toast } from '../lib/api'
import { saveBrowserHistory } from '../lib/browserHistory'
import type { BrowserDetectResult } from '../lib/browserDetection'
import { DetectionTable } from '../components/ui'

type FrameResult = DetectResult & Partial<Pick<BrowserDetectResult, 'annotatedDataUrl' | 'engine' | 'peak_confidence' | 'conf_threshold'>>

export default function Live() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const stageRef = useRef<HTMLDivElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const inFlightRef = useRef(false)
  const mountedRef = useRef(false)
  const requestRef = useRef(0)
  const lastAutoSave = useRef(0)
  const autoSaveRef = useRef(true)
  const retryAfter = useRef(0)
  const lastResultAt = useRef(0)
  const [streamAspect, setStreamAspect] = useState<number | null>(null)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<FrameResult | null>(null)
  const [fps, setFps] = useState(0)
  const [error, setError] = useState('')
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([])
  const [deviceId, setDeviceId] = useState('')
  // True while the browser is hiding camera names (no permission granted yet).
  const [labelsHidden, setLabelsHidden] = useState(false)
  const [enablingNames, setEnablingNames] = useState(false)
  const [busy, setBusy] = useState(false)
  const [savingSnapshot, setSavingSnapshot] = useState(false)
  const [mode, setMode] = useState<'coco' | 'small' | 'world'>('coco')
  const [backendModel, setBackendModel] = useState('')
  const [autoSave, setAutoSave] = useState(() => {
    try { return localStorage.getItem('ods-auto-save-live') !== 'off' }
    catch { return true }
  })
  const [autoSaveStatus, setAutoSaveStatus] = useState('')
  autoSaveRef.current = autoSave
  const [classesText, setClassesText] = useState('person, car, dog')
  const [source, setSource] = useState<'backend' | 'browser' | null>(null)
  const [modelPct, setModelPct] = useState<number | null>(null)
  const [promptStatus, setPromptStatus] = useState('')
  // Consecutive frames that ran fine but produced no box — used to explain a
  // silent screen instead of leaving the user guessing why nothing appears.
  const [silentFrames, setSilentFrames] = useState(0)
  // Refs so the interval callback always reads the latest mode/prompts.
  const modeRef = useRef(mode); modeRef.current = mode
  const classesRef = useRef(classesText); classesRef.current = classesText
  const backendModelRef = useRef(backendModel); backendModelRef.current = backendModel

  const refreshCameras = async () => {
    const devices = await navigator.mediaDevices?.enumerateDevices()
    if (!devices || !mountedRef.current) return
    const cameras = devices.filter(device => device.kind === 'videoinput')
    setDevices(cameras.filter(camera => camera.deviceId))
    setLabelsHidden(!cameras.some(camera => camera.label))
    setDeviceId(current => cameras.some(camera => camera.deviceId === current) ? current : '')
  }

  useEffect(() => {
    mountedRef.current = true
    const refresh = () => { void refreshCameras().catch(() => null) }
    refresh()
    navigator.mediaDevices?.addEventListener('devicechange', refresh)
    window.addEventListener('focus', refresh)
    api.modelStatus().then(m => {
      if (!mountedRef.current) return
      const file = m.model_path?.split(/[/\\]/).pop() || ''
      if (file.includes('world')) setMode('world')
      else if (file && !/^yolov8[nsmlx]\.pt$/.test(file)) setBackendModel(file)
    }).catch(() => null)
    // With no backend, detection runs in this browser. Check which weights the
    // deployment actually serves and start on one that works, instead of
    // opening on COCO and failing on the very first frame.
    void isBackendOnline().then(async online => {
      if (online || !mountedRef.current) return
      const { isModelAvailable } = await import('../lib/browserDetection')
      if (await isModelAvailable('coco')) return
      if (!mountedRef.current) return
      if (await isModelAvailable('broad')) return
      if (!mountedRef.current) return
      // No bundled weights and no remote COCO export: open vocabulary is the
      // only path that can still find objects here.
      setMode('world')
    }).catch(() => null)
    return () => {
      mountedRef.current = false
      navigator.mediaDevices?.removeEventListener('devicechange', refresh)
      window.removeEventListener('focus', refresh)
      stop()
    }
  }, [])

  const draw = (dets: DetectResult['detections'], w: number, h: number) => {
    // Boxes come back in original video pixels, but the video is shown with
    // object-contain inside the stage, so it is letterboxed whenever the
    // camera's aspect differs from the stage's. Drawing the boxes in video
    // pixels across the whole stage put them in the wrong place (or off the
    // picture entirely on portrait phone cameras) — so map every box onto
    // the video's actual displayed rectangle inside the stage.
    const c = canvasRef.current
    const stage = stageRef.current
    if (!c || !stage) return
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
    dets.forEach(d => {
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
    if (inFlightRef.current || Date.now() < retryAfter.current) return
    const v = videoRef.current; if (!v || v.readyState < 2) return
    const stream = streamRef.current
    if (!stream || !v.videoWidth || !v.videoHeight) return
    const mode = modeRef.current
    const text = classesRef.current
    const current = () => streamRef.current === stream && modeRef.current === mode &&
      (mode !== 'world' || classesRef.current === text)
    inFlightRef.current = true
    try {
      const prompts = mode === 'world' ? parseDetectionPrompts(text) : null
      const frame = document.createElement('canvas')
      frame.width = v.videoWidth
      frame.height = v.videoHeight
      frame.getContext('2d')!.drawImage(v, 0, 0)
      const progress = (loaded: number, total: number) => {
        if (total > 0 && current()) setModelPct(Math.round(loaded / total * 100))
      }
      const runSmallObjects = async () => {
        const { detectInBrowser } = await import('../lib/browserDetection')
        const result = await detectInBrowser(frame, frame.width, frame.height, progress, 'expanded', true)
        return { result, source: 'browser' as const }
      }
      const runBackend = async (): Promise<FrameResult> => {
        const blob = await new Promise<Blob | null>(resolve => frame.toBlob(resolve, 'image/jpeg', 0.85))
        if (!blob) throw new Error('Could not encode the camera frame.')
        return prompts ? api.openVocabFrame(blob, prompts) : api.detectFrame(blob)
      }
      const runDetection = async () => {
        if (mode === 'small') return runSmallObjects()
        if (mode === 'coco' && backendModelRef.current) {
          return { result: await runBackend(), source: 'backend' as const }
        }
        return detectWithFallback<FrameResult>(runBackend, async () => {
          if (prompts) {
            const { detectOpenVocabInBrowser } = await import('../lib/browserOpenVocab')
            return detectOpenVocabInBrowser(frame, frame.width, frame.height, prompts, progress,
              (done, total, prompt) => {
                if (current()) setPromptStatus(`Checking “${prompt}” (${done} of ${total})`)
              })
          }
          const { detectInBrowser } = await import('../lib/browserDetection')
          return detectInBrowser(frame, frame.width, frame.height, progress)
        })
      }
      const { result: res, source } = await runDetection()
      if (!current()) return
      const now = performance.now()
      setFps(lastResultAt.current ? 1000 / (now - lastResultAt.current) : 0)
      lastResultAt.current = now
      setError(''); setSource(source); setResult(res)
      draw(res.detections, frame.width, frame.height)
      // An empty frame is a real result, but a permanently silent overlay looks
      // broken. Once several frames in a row pass the model yet nothing clears
      // the threshold, say so and name the two things the user can change.
      const peak = res.peak_confidence
      const threshold = res.conf_threshold
      if (res.count === 0 && peak !== undefined && threshold !== undefined) {
        setSilentFrames(frames => {
          const next = frames + 1
          if (next === 6) {
            setError(
              `The detector is running but nothing has passed the ${(threshold * 100).toFixed(0)}% confidence threshold ` +
              `(best candidate so far: ${(peak * 100).toFixed(0)}%). Point the camera at a familiar object, add light, ` +
              'or lower the confidence threshold in Settings.',
            )
          }
          return next
        })
      } else if (res.count > 0) {
        setSilentFrames(0)
      }
      if (autoSaveRef.current && res.count > 0 && Date.now() - lastAutoSave.current >= 30_000) {
        lastAutoSave.current = Date.now()
        try {
          let preview = res.annotatedDataUrl
          if (!preview) {
            const { annotate } = await import('../lib/browserDetection')
            preview = annotate(frame, frame.width, frame.height, res.detections)
          }
          if (!current() || !autoSaveRef.current) return
          await storeLiveFrame(res, preview, source)
          if (current()) setAutoSaveStatus('Snapshot saved in Detection History')
        } catch (error) {
          if (current()) {
            setAutoSave(false)
            setAutoSaveStatus(error instanceof Error ? error.message : 'Could not save the live snapshot.')
          }
        }
      }
    } catch (error) {
      if (current()) {
        retryAfter.current = Date.now() + 3000
        setError(error instanceof Error ? error.message : 'Frame inference failed.')
      }
    } finally {
      inFlightRef.current = false
      if (mountedRef.current) { setModelPct(null); setPromptStatus('') }
    }
  }

  const openCamera = async (preview: boolean) => {
    const request = ++requestRef.current
    const current = () => mountedRef.current && request === requestRef.current
    setError('')
    preview ? setEnablingNames(true) : setBusy(true)
    let stream: MediaStream | undefined
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access requires HTTPS or localhost.')
      stream = await navigator.mediaDevices.getUserMedia({
        video: preview ? true : {
          ...(deviceId && { deviceId: { exact: deviceId } }),
          width: { ideal: 1280 }, height: { ideal: 720 },
        },
        audio: false,
      })
      if (!current()) return
      if (preview) {
        await refreshCameras()
        return
      }
      void refreshCameras().catch(() => null)
      const video = videoRef.current
      if (!video) return
      streamRef.current = stream
      video.srcObject = stream
      await video.play()
      if (!current()) return
      stream.getVideoTracks()[0]?.addEventListener('ended', () => {
        if (streamRef.current !== stream) return
        stop()
        setError('The active camera disconnected or access was revoked. Select a camera and start again.')
        void refreshCameras().catch(() => null)
      }, { once: true })
      lastAutoSave.current = 0
      retryAfter.current = 0
      lastResultAt.current = 0
      setAutoSaveStatus('')
      setRunning(true)
      timerRef.current = setInterval(captureLoop, 160)
    } catch (error) {
      if (!current()) return
      if (!preview) stop()
      const details: Record<string, string> = {
        NotAllowedError: 'Allow camera access in your browser and device settings, then try again.',
        NotFoundError: 'No camera is available. Connect a camera and try again.',
        NotReadableError: 'The camera could not be opened. Close other apps using it and try again.',
        OverconstrainedError: 'The selected camera is unavailable. Choose another camera and try again.',
      }
      const name = error instanceof Error ? error.name : ''
      setError(details[name] || (error instanceof Error ? error.message : 'Could not start the camera.'))
      void refreshCameras().catch(() => null)
    } finally {
      if (stream && stream !== streamRef.current) stream.getTracks().forEach(track => track.stop())
      if (mountedRef.current) { setBusy(false); setEnablingNames(false) }
    }
  }

  const stop = () => {
    requestRef.current++
    if (timerRef.current) clearInterval(timerRef.current)
    timerRef.current = null
    streamRef.current?.getTracks().forEach(track => track.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    if (canvasRef.current) canvasRef.current.width = 0
    if (!mountedRef.current) return
    setStreamAspect(null)
    setRunning(false); setResult(null); setFps(0); setSource(null); setModelPct(null); setPromptStatus('')
    setSilentFrames(0)
  }
  const storeLiveFrame = (result: FrameResult, previewDataUrl: string | undefined, source: 'browser' | 'backend') =>
    saveBrowserHistory({
      inputType: 'live',
      count: result.count,
      classes: result.unique_classes,
      avgConfidence: result.avg_confidence,
      inferenceMs: result.inference_ms,
      fps: result.fps,
      model: result.engine || (source === 'backend' ? 'Backend (PyTorch)' : 'In-browser (ONNX)'),
      previewDataUrl,
    })

  const saveSnapshot = async () => {
    if (!running || !source || !result || savingSnapshot) return
    setSavingSnapshot(true)
    try {
      if (source === 'browser') {
        await storeLiveFrame(result, result.annotatedDataUrl, source)
        toast(`Snapshot saved in this browser · ${result.count} object(s)`)
        return
      }
      const video = videoRef.current
      if (!video?.videoWidth || !video.videoHeight) throw new Error('No camera frame is ready.')
      const frame = document.createElement('canvas')
      frame.width = video.videoWidth
      frame.height = video.videoHeight
      frame.getContext('2d')!.drawImage(video, 0, 0)
      const blob = await new Promise<Blob | null>(resolve => frame.toBlob(resolve, 'image/jpeg', 0.9))
      if (!blob) throw new Error('Could not encode the camera frame.')
      const saved = await api.saveFrame(blob, mode === 'world' ? parseDetectionPrompts(classesText) : undefined)
      if (!saved.record_id) throw new Error(saved.history_error || 'The backend did not return a saved record.')
      toast(`Snapshot saved · ${saved.count} object(s), record #${saved.record_id}`)
    } catch (error) {
      toast(error instanceof Error ? error.message : 'Could not save the snapshot.', 'err')
    } finally {
      setSavingSnapshot(false)
    }
  }

  return (
    <div className="space-y-4">
      <div className="card">
        <div className="flex flex-wrap items-center gap-2">
          <select className="input max-w-xs" value={deviceId} disabled={running || busy || enablingNames} onChange={e => setDeviceId(e.target.value)} aria-label="Camera device">
            <option value="">Default camera</option>
            {devices.map((d, i) => <option key={d.deviceId} value={d.deviceId}>{d.label || `Camera ${i + 1}`}</option>)}
          </select>
          {!running
            ? <button className="btn-primary" onClick={() => openCamera(false)} disabled={busy || enablingNames}><Play size={15} /> {busy ? 'Starting…' : 'Start webcam'}</button>
            : <button className="btn-danger" onClick={stop}><Square size={15} /> Stop webcam</button>}
          <button className="btn-ghost" onClick={saveSnapshot} disabled={!running || !result || savingSnapshot}><Download size={15} /> {savingSnapshot ? 'Saving…' : 'Save snapshot'}</button>
          <span className={`ml-auto chip ${running ? '' : '!bg-white/5 !text-slate-400 !border-white/10'}`}>{running ? '● LIVE' : '○ STOPPED'}</span>
        </div>
        {labelsHidden && !running && (
          <p className="mt-2 rounded-xl bg-amber-500/10 border border-amber-500/30 px-3 py-2 text-sm text-amber-200">
            Camera names are hidden — browsers do not reveal them until you allow camera access, so this list only shows generic names like “Camera 1”.{' '}
            <button className="font-semibold underline" onClick={() => openCamera(true)} disabled={enablingNames || busy}>{enablingNames ? 'Asking…' : 'Enable camera names'}</button>
          </p>
        )}
        <p className="mt-2 text-xs text-slate-500">Camera access requires HTTPS or localhost. The list shows the cameras your browser makes available.</p>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <button className={mode === 'coco' ? 'btn-primary' : 'btn-ghost'} onClick={() => setMode('coco')}>{backendModel ? `Backend model (${backendModel})` : 'COCO classes (80)'}</button>
          <button className={mode === 'small' ? 'btn-primary' : 'btn-ghost'} onClick={() => setMode('small')}>Small objects (expanded)</button>
          <button className={mode === 'world' ? 'btn-primary' : 'btn-ghost'} onClick={() => setMode('world')}><Sparkles size={15} /> Any object — type its name</button>
        </div>
        {mode === 'small' && <p className="mt-2 text-sm text-slate-400">Looks for small items such as pens and combs using the expanded model. It checks crops of larger frames, so updates are slower. Bring the object close and keep it in focus. For phone cases, try “Any object” and select “phone case”.</p>}
        <label className="mt-3 flex items-start gap-2 text-sm text-slate-300">
          <input type="checkbox" className="mt-1 accent-indigo-400" checked={autoSave} onChange={event => {
            autoSaveRef.current = event.target.checked
            setAutoSave(event.target.checked)
            try { localStorage.setItem('ods-auto-save-live', event.target.checked ? 'on' : 'off') }
            catch { /* Live detection still works with storage disabled. */ }
            setAutoSaveStatus('')
          }} />
          <span>Auto-save live history <span className="text-slate-400">(first detection, then up to once every 30 seconds in this browser)</span></span>
        </label>
        {autoSaveStatus && <p role="status" className="mt-2 text-sm text-slate-400">{autoSaveStatus}</p>}
        {mode === 'world' && (
          <div className="mt-3">
            <label className="label">Objects to find (comma-separated, up to 20)</label>
            <input className="input" value={classesText} onChange={e => setClassesText(e.target.value)} placeholder="person, car, dog" aria-label="Object names to detect" />
            <p className="mt-1 text-xs text-slate-500">Searches for the objects you name. With the backend connected this uses YOLO-World (first use downloads its weights and text encoder). Without a backend it runs Grounding DINO in your browser instead (~205 MB download once, slower, experimental — each object checked takes several seconds, so live video is snapshots, not smooth video).</p>
            <div className="mt-2 flex flex-wrap gap-1.5" aria-label="Try an object outside the COCO list">
              {['phone case', 'pen', 'comb', 'microphone', 'sunglasses', 'headphones'].map(p => (
                <button key={p} type="button" className="rounded-full border border-violet-500/40 bg-violet-500/10 px-2.5 py-1 text-xs text-violet-200 hover:bg-violet-500/20" onClick={() => setClassesText(p)} title="Try this non-COCO object">{p}</button>
              ))}
            </div>
          </div>
        )}
        {error && <p className="mt-3 rounded-xl bg-red-500/10 border border-red-500/30 px-3 py-2 text-sm text-red-300">{error}</p>}
        {modelPct !== null && <p className="mt-3 rounded-xl bg-sky-500/10 border border-sky-500/30 px-3 py-2 text-sm text-sky-200">Downloading {mode === 'world' ? 'open-vocab' : 'detection'} model… {modelPct}% ({mode === 'world' ? '~205 MB' : mode === 'small' ? '~125 MB' : '~12 MB'}, once only — it is cached in your browser after that).</p>}
        {running && promptStatus && <p className="mt-3 rounded-xl bg-violet-500/10 border border-violet-500/30 px-3 py-2 text-sm text-violet-200">{promptStatus} Open-vocab checks one object at a time — each takes several seconds.</p>}
        {source === 'browser' && <p className="mt-3 rounded-xl bg-violet-500/10 border border-violet-500/30 px-3 py-2 text-sm text-violet-200">Detection runs in your browser. Use Save snapshot to keep this result immediately.</p>}
        <div ref={stageRef} className="relative mt-4 max-h-[75vh] w-full overflow-hidden rounded-xl border border-white/10 bg-black aspect-video" style={streamAspect ? { aspectRatio: String(streamAspect) } : undefined}>
          <video ref={videoRef} playsInline muted className="absolute inset-0 h-full w-full object-contain" onResize={event => {
            const { videoWidth, videoHeight } = event.currentTarget
            if (videoWidth && videoHeight) setStreamAspect(videoWidth / videoHeight)
          }} />
          <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
          {!running && <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-slate-400"><CameraOff size={36} /><p className="px-4 text-center text-sm">Webcam is off — press “Start webcam” to begin real-time detection.</p></div>}
        </div>
        <p className="mt-2 flex items-center gap-2 text-xs text-slate-500"><Video size={13} /> Camera frames are sampled up to 6 times per second.</p>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        <div className="card">
          <h2 className="font-semibold">Live statistics</h2>
          <div className="mt-3 space-y-2 text-sm">
            <p className="flex justify-between"><span className="text-slate-400">Objects in frame</span><b>{result?.count ?? 0}</b></p>
            <p className="flex justify-between"><span className="text-slate-400">Unique classes</span><b>{result?.unique_classes.length ?? 0}</b></p>
            <p className="flex justify-between"><span className="text-slate-400">Avg confidence</span><b>{result ? (result.avg_confidence * 100).toFixed(1) + '%' : '—'}</b></p>
            <p className="flex justify-between"><span className="text-slate-400">Inference time</span><b>{result ? result.inference_ms.toFixed(0) + ' ms' : '—'}</b></p>
            <p className="flex justify-between"><span className="text-slate-400">Detection updates / second</span><b>{fps ? fps.toFixed(1) : '—'}</b></p>
            <p className="flex justify-between"><span className="text-slate-400">Engine</span><b>{result?.engine || (source === 'browser' ? 'In-browser (ONNX)' : source === 'backend' ? 'Backend (PyTorch)' : '—')}</b></p>
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
