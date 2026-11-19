import { useEffect, useRef, useState } from 'react'
import { Camera, CameraOff, Download, Play, Square, Video } from 'lucide-react'
import { api, DetectResult, toast } from '../lib/api'
import { DetectionTable } from '../components/ui'

export default function Live() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const timerRef = useRef<any>(null)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<DetectResult | null>(null)
  const [fps, setFps] = useState(0)
  const [error, setError] = useState('')
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([])
  const [deviceId, setDeviceId] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    navigator.mediaDevices?.enumerateDevices().then(ds => setDevices(ds.filter(d => d.kind === 'videoinput'))).catch(() => null)
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
    const v = videoRef.current; if (!v || v.readyState < 2) return
    try {
      const off = document.createElement('canvas')
      off.width = v.videoWidth; off.height = v.videoHeight
      off.getContext('2d')!.drawImage(v, 0, 0)
      const blob = await new Promise<Blob | null>(r => off.toBlob(r, 'image/jpeg', 0.8))
      if (!blob) return
      const t0 = performance.now()
      const res = await api.detectFrame(blob)
      setResult(res); setFps(res.fps)
      draw(res.detections, v.videoWidth, v.videoHeight)
      void t0
    } catch (e: any) { setError(e.message || 'Frame inference failed.') }
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
        {error && <p className="mt-3 rounded-xl bg-red-500/10 border border-red-500/30 px-3 py-2 text-sm text-red-300">{error}</p>}
        <div className="relative mt-4 overflow-hidden rounded-xl border border-white/10 bg-black aspect-video">
          <video ref={videoRef} playsInline muted className="h-full w-full object-contain" />
          <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
          {!running && <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-slate-400"><CameraOff size={36} /><p className="text-sm">Webcam is off — press “Start webcam” to begin real-time detection.</p><p className="text-xs text-slate-500">Your camera stays in this browser; frames are sent to the local backend for YOLOv8 inference.</p></div>}
        </div>
        <p className="mt-2 flex items-center gap-2 text-xs text-slate-500"><Video size={13} /> Frames are captured at ~6 fps and sent to the local YOLOv8 backend. No fake stream is ever shown.</p>
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
