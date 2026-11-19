import { useEffect, useRef, useState } from 'react'
import { Download, Loader2, Play, Square } from 'lucide-react'
import { api, toast } from '../lib/api'

export default function Training() {
  const [cfg, setCfg] = useState({ model: 'yolov8n.pt', epochs: 10, imgsz: 640, batch: 8, device: 'cpu' })
  const [st, setSt] = useState<any>(null)
  const [busy, setBusy] = useState(false)
  const timer = useRef<any>(null)
  useEffect(() => {
    api.trainingStatus().then(setSt).catch(() => null)
    timer.current = setInterval(() => api.trainingStatus().then(setSt).catch(() => null), 3000)
    return () => clearInterval(timer.current)
  }, [])
  const start = async () => {
    setBusy(true)
    try { await api.trainingStart(cfg); toast('Training started — real Ultralytics run, progress below is live') }
    catch (e: any) { toast(e.message, 'err') } finally { setBusy(false) }
  }
  const stop = async () => { try { await api.trainingStop(); toast('Stop requested — training halts after the current epoch') } catch (e: any) { toast(e.message, 'err') } }
  const m = st?.metrics || {}
  const running = st?.status === 'running'
  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <div className="card">
        <h2 className="font-semibold">Model configuration</h2>
        <div className="mt-3 space-y-3">
          <div><label className="label">Pretrained variant</label>
            <select className="input" value={cfg.model} onChange={e => setCfg({ ...cfg, model: e.target.value })}>
              <option value="yolov8n.pt">YOLOv8n (nano — fastest, recommended on CPU)</option>
              <option value="yolov8s.pt">YOLOv8s (small)</option>
              <option value="yolov8m.pt">YOLOv8m (medium — needs more RAM/time)</option>
            </select></div>
          <div className="grid grid-cols-2 gap-3">
            <div><label className="label">Epochs</label><input type="number" className="input" value={cfg.epochs} min={1} max={300} onChange={e => setCfg({ ...cfg, epochs: +e.target.value })} /></div>
            <div><label className="label">Image size</label><input type="number" className="input" value={cfg.imgsz} step={32} min={320} max={1280} onChange={e => setCfg({ ...cfg, imgsz: +e.target.value })} /></div>
            <div><label className="label">Batch size</label><input type="number" className="input" value={cfg.batch} min={1} max={64} onChange={e => setCfg({ ...cfg, batch: +e.target.value })} /></div>
            <div><label className="label">Device</label><select className="input" value={cfg.device} onChange={e => setCfg({ ...cfg, device: e.target.value })}><option value="cpu">CPU</option><option value="cuda">CUDA (GPU, if available)</option></select></div>
          </div>
          <p className="text-xs text-slate-400">Transfer learning: pretrained COCO weights are fine-tuned on your Module 1 dataset (dataset/data.yaml). Training runs in the Python backend, not in the browser.</p>
          <div className="flex gap-2">
            <button className="btn-primary" onClick={start} disabled={busy || running}>{busy ? <Loader2 size={15} className="animate-spin" /> : <Play size={15} />} Start training</button>
            <button className="btn-danger" onClick={stop} disabled={!running}><Square size={15} /> Stop safely</button>
          </div>
          {st?.best && <div className="rounded-xl bg-emerald-500/10 border border-emerald-500/30 p-3 text-sm">Best weights: <b>best.pt</b> ({st.best.size_mb} MB)
            <div className="mt-2 flex gap-2"><a className="btn-ghost" href="/api/training/model/download"><Download size={14} /> Download best.pt</a>
            <button className="btn-ghost" onClick={async () => { try { await api.loadBest(); toast('best.pt loaded as the active model') } catch (e: any) { toast(e.message, 'err') } }}>Load as active model</button></div></div>}
          {st?.error && <p className="rounded-xl bg-red-500/10 border border-red-500/30 px-3 py-2 text-sm text-red-300">{st.error}</p>}
        </div>
      </div>
      <div className="card lg:col-span-2">
        <h2 className="font-semibold">Training monitor</h2>
        {!st || st.status === 'idle' ? <p className="mt-3 text-sm text-slate-400">No training run yet. Configure the model on the left, prepare a dataset in Module 1, then start training. Progress, loss and mAP values below come from the real Ultralytics trainer (results.csv) — they are never invented.</p> : (
          <>
            <div className="mt-3 flex flex-wrap gap-3 text-sm"><span className="chip capitalize">{st.status}</span><span>Epoch {st.epoch} / {st.total_epochs}</span><span>{st.progress}%</span>{st.started_at && <span className="text-slate-400">Started {st.started_at}</span>}</div>
            <div className="mt-2 h-3 overflow-hidden rounded-full bg-white/10"><div className="h-full rounded-full bg-gradient-to-r from-sky-500 to-violet-500 transition-all" style={{ width: `${st.progress}%` }} /></div>
            <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
              {[['Box loss', m.box_loss], ['Class loss', m.cls_loss], ['Val loss', m.val_loss], ['Precision', m.precision], ['Recall', m.recall], ['mAP50', m.mAP50], ['mAP50-95', m.mAP50_95]].map(([l, v]) => (
                <div key={l as string} className="rounded-xl bg-white/5 p-3"><p className="text-xs text-slate-400">{l}</p><p className="mt-1 text-lg font-bold">{v === undefined || v === null ? '—' : Number(v).toFixed(3)}</p></div>
              ))}
            </div>
            <h3 className="mt-5 font-semibold">Training log</h3>
            <pre className="mt-2 h-64 overflow-y-auto rounded-xl bg-black/50 p-3 font-mono text-xs leading-relaxed text-slate-300">{st.logs?.length ? st.logs.join('\n') : 'No log lines yet.'}</pre>
          </>
        )}
      </div>
    </div>
  )
}
