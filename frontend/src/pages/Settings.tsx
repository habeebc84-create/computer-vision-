import { useEffect, useState } from 'react'
import { Loader2, Save } from 'lucide-react'
import { api, AvailableModel, toast } from '../lib/api'
import { getLocalSettings, saveLocalSettings } from '../lib/yoloPost'

const FALLBACK_MODELS: AvailableModel[] = [
  { id: 'yolov8n.pt', path: 'models/yolov8n.pt', kind: 'coco', name: 'YOLOv8n (nano)', classes: 80, description: 'Fastest COCO model — 80 fixed classes. Default.', size_mb: 6, downloaded: false },
  { id: 'yolov8s.pt', path: 'models/yolov8s.pt', kind: 'coco', name: 'YOLOv8s (small)', classes: 80, description: 'More accurate than nano, still fast on CPU.', size_mb: 22, downloaded: false },
  { id: 'yolov8m.pt', path: 'models/yolov8m.pt', kind: 'coco', name: 'YOLOv8m (medium)', classes: 80, description: 'Noticeably more accurate; slower on CPU.', size_mb: 52, downloaded: false },
  { id: 'yolov8x.pt', path: 'models/yolov8x.pt', kind: 'coco', name: 'YOLOv8x (extra-large)', classes: 80, description: 'Most accurate COCO model; slowest, downloads ~130 MB once.', size_mb: 130, downloaded: false },
  { id: 'yolov8s-worldv2.pt', path: 'models/yolov8s-worldv2.pt', kind: 'world', name: 'YOLOv8s-World v2 (open-vocabulary)', classes: null, description: "Type any object name and it detects it — not limited to the 80 COCO classes.", size_mb: 25, downloaded: false },
]

export default function Settings() {
  const [s, setS] = useState<any>({ model_path: 'models/yolov8n.pt', conf: 0.25, iou: 0.45, imgsz: 640, device: 'auto' })
  const [model, setModel] = useState<any>(null)
  const [models, setModels] = useState<AvailableModel[]>(FALLBACK_MODELS)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => {
    api.getSettings().then(setS).catch(() => {
      // Backend offline: fall back to the settings saved on this device
      // (they drive in-browser detection).
      const local = getLocalSettings()
      setS((s: any) => ({ ...s, conf: local.conf, iou: local.iou, imgsz: local.imgsz }))
    })
    api.modelStatus().then(setModel).catch(() => null)
    api.listModels().then(setModels).catch(() => null)
  }, [])
  const selectedModel = models.find(m => m.path === s.model_path)
  const modelOptions = [...models]
  if (s.model_path && !modelOptions.some(m => m.path === s.model_path)) {
    modelOptions.push({ id: s.model_path, path: s.model_path, kind: 'coco', name: `Custom — ${s.model_path.split('/').pop()}`, classes: null, description: 'Custom weights (e.g. models/best.pt from training).', size_mb: 0, downloaded: true })
  }
  const save = async () => {
    setBusy(true); setError('')
    // Always keep a copy on this device — in-browser detection reads it.
    saveLocalSettings({ conf: s.conf, iou: s.iou, imgsz: s.imgsz })
    try {
      const saved = await api.saveSettings(s); setS(saved)
      const m = await api.modelStatus(); setModel(m)
      toast('Settings saved and applied to inference')
    } catch (e: any) {
      setError('')
      toast('Backend offline — settings saved on this device only; they apply to in-browser detection', 'err')
    } finally { setBusy(false) }
  }
  const loadModel = async () => {
    setBusy(true); setError('')
    try { const m = await api.loadModel(s.model_path); setModel(m); toast(`Model loaded: ${m.model_path}`) }
    catch (e: any) { setError(e.message); toast(e.message, 'err') } finally { setBusy(false) }
  }
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="card">
        <h2 className="font-semibold">Detection settings</h2>
        <p className="mt-1 text-sm text-slate-400">These values are sent to the backend and used by every real YOLOv8 inference call (live and image). They are also saved on this device, so in-browser detection uses them when the backend is offline.</p>
        <div className="mt-4 space-y-4">
          <div>
            <label className="label">Model</label>
            <select className="input" value={s.model_path} onChange={e => setS({ ...s, model_path: e.target.value })} aria-label="Model">
              {modelOptions.map(m => <option key={m.path} value={m.path}>{m.name}{m.downloaded ? '' : ` — downloads ~${m.size_mb} MB once`}</option>)}
              <option value="models/best.pt">Custom — best.pt (from Module 2 training)</option>
            </select>
            <p className="mt-1 text-xs text-slate-500">{selectedModel?.description || 'Custom weights (e.g. models/best.pt from training).'}{selectedModel?.kind === 'world' ? ' Choosing the world model switches Image and Live detection to Any-object mode — type the object names you want to find.' : ''}</p>
          </div>
          <div><label className="label">Confidence threshold — {s.conf}</label><input type="range" min={0.05} max={0.95} step={0.05} value={s.conf} onChange={e => setS({ ...s, conf: +e.target.value })} className="w-full accent-sky-500" /></div>
          <div><label className="label">IoU threshold — {s.iou}</label><input type="range" min={0.1} max={0.95} step={0.05} value={s.iou} onChange={e => setS({ ...s, iou: +e.target.value })} className="w-full accent-sky-500" /></div>
          <div><label className="label">Image size (imgsz)</label><select className="input" value={s.imgsz} onChange={e => setS({ ...s, imgsz: +e.target.value })}><option value={320}>320 — fastest</option><option value={480}>480</option><option value={640}>640 — default</option><option value={960}>960</option><option value={1280}>1280 — slowest, most detail</option></select></div>
          <div><label className="label">Device</label><select className="input" value={s.device} onChange={e => setS({ ...s, device: e.target.value })}><option value="auto">Auto (CUDA if available, else CPU)</option><option value="cpu">CPU</option><option value="cuda">CUDA (GPU)</option></select></div>
          {error && <p className="rounded-xl bg-red-500/10 border border-red-500/30 px-3 py-2 text-sm text-red-300">{error}</p>}
          <div className="flex gap-2">
            <button className="btn-primary" onClick={save} disabled={busy}>{busy ? <Loader2 size={15} className="animate-spin" /> : <Save size={15} />} Save settings</button>
            <button className="btn-ghost" onClick={loadModel} disabled={busy}>Load this model now</button>
          </div>
        </div>
      </div>
      <div className="space-y-4">
        <div className="card">
          <h2 className="font-semibold">Current model</h2>
          <div className="mt-3 space-y-2 text-sm">
            <p className="flex justify-between"><span className="text-slate-400">Status</span><b>{model?.loaded ? 'Loaded' : 'Not loaded'}</b></p>
            <p className="flex justify-between"><span className="text-slate-400">Path</span><b className="break-all text-right">{model?.model_path || '—'}</b></p>
            <p className="flex justify-between"><span className="text-slate-400">Device</span><b>{model?.device || '—'}</b></p>
            <p className="flex justify-between"><span className="text-slate-400">Classes</span><b>{model?.classes ?? '—'}</b></p>
            <p className="flex justify-between"><span className="text-slate-400">Loaded at</span><b>{model?.loaded_at || '—'}</b></p>
            {model?.error && <p className="text-red-300">{model.error}</p>}
          </div>
          {model?.class_names?.length ? <p className="mt-3 text-xs leading-relaxed text-slate-400">COCO classes: {model.class_names.join(', ')}</p> : null}
        </div>
        <div className="card">
          <h2 className="font-semibold">About this system</h2>
          <p className="mt-2 text-sm text-slate-300">Real-Time Object Detection System using YOLOv8 and Deep Learning — ITA0523 Computer Vision capstone. Backend: FastAPI + Ultralytics YOLOv8 + PyTorch + OpenCV. History: SQLite. Frontend: React + Vite + Tailwind + Recharts.</p>
          <p className="mt-2 text-sm text-slate-400">Pretrained weights (yolov8n.pt, ~6 MB; yolov8x.pt, ~130 MB; yolov8s-worldv2.pt, ~25 MB) download automatically from Ultralytics on first load into <code className="rounded bg-white/10 px-1">models/</code> and are git-ignored. Custom weights from Module 2 are saved as <code className="rounded bg-white/10 px-1">models/best.pt</code>. The world model is open-vocabulary: it detects any object you type, not just the 80 COCO classes.</p>
        </div>
      </div>
    </div>
  )
}
