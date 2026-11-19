import { useEffect, useState } from 'react'
import { Activity, BarChart3, Camera, Database, FileVideo, Gauge, History, Home, Image as ImageIcon, Layers, ScanEye, Settings as SettingsIcon, Video, X, Menu, Cpu } from 'lucide-react'
import { api, ModelStatus } from '../lib/api'

export type PageId = 'dashboard' | 'live' | 'image' | 'video' | 'dataset' | 'training' | 'inference' | 'history' | 'analytics' | 'settings'
export const NAV: { id: PageId; label: string; icon: any; group: string }[] = [
  { id: 'dashboard', label: 'Overview Dashboard', icon: Home, group: 'Detect' },
  { id: 'live', label: 'Live Detection', icon: Camera, group: 'Detect' },
  { id: 'image', label: 'Image Detection', icon: ImageIcon, group: 'Detect' },
  { id: 'video', label: 'Video Detection', icon: FileVideo, group: 'Detect' },
  { id: 'dataset', label: 'Module 1 — Dataset', icon: Database, group: 'Modules' },
  { id: 'training', label: 'Module 2 — Training', icon: Layers, group: 'Modules' },
  { id: 'inference', label: 'Module 3 — Inference', icon: ScanEye, group: 'Modules' },
  { id: 'history', label: 'Detection History', icon: History, group: 'Data' },
  { id: 'analytics', label: 'Analytics & Reports', icon: BarChart3, group: 'Data' },
  { id: 'settings', label: 'Settings', icon: SettingsIcon, group: 'System' },
]

export function Layout({ page, setPage, children }: { page: PageId; setPage: (p: PageId) => void; children: React.ReactNode }) {
  const [open, setOpen] = useState(false)
  const [collapsed, setCollapsed] = useState(false)
  const [model, setModel] = useState<ModelStatus | null>(null)
  useEffect(() => { api.modelStatus().then(setModel).catch(() => null) }, [page])
  const groups = ['Detect', 'Modules', 'Data', 'System']
  const sidebar = (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-3 px-4 py-5">
        <div className="rounded-xl bg-gradient-to-br from-sky-500 to-violet-500 p-2 text-white"><ScanEye size={20} /></div>
        {!collapsed && <div><p className="font-bold leading-tight">Object Detection</p><p className="text-[11px] text-slate-400">YOLOv8 · ITA0523</p></div>}
      </div>
      <nav className="flex-1 space-y-4 overflow-y-auto px-3 pb-4">
        {groups.map(g => (
          <div key={g}>
            {!collapsed && <p className="px-3 pb-1 text-[11px] font-semibold uppercase tracking-wider text-slate-500">{g}</p>}
            {NAV.filter(n => n.group === g).map(n => (
              <button key={n.id} title={n.label} onClick={() => { setPage(n.id); setOpen(false) }}
                className={`navitem ${page === n.id ? 'active' : ''} ${collapsed ? 'justify-center px-2' : ''}`}>
                <n.icon size={18} className="shrink-0" />{!collapsed && n.label}
              </button>
            ))}
          </div>
        ))}
      </nav>
      {!collapsed && (
        <div className="m-3 rounded-xl bg-white/5 border border-white/10 p-3 text-xs">
          <p className="flex items-center gap-1.5 font-semibold text-slate-200"><Cpu size={13} /> Model</p>
          <p className="mt-1 truncate text-slate-400" title={model?.model_path}>{model?.loaded ? 'Loaded' : 'Not loaded'} — {model?.model_path?.split('/').pop() || 'yolov8n.pt'}</p>
          <p className="text-slate-500">{model?.device || 'cpu'} · {model?.classes ?? 0} classes</p>
        </div>
      )}
    </div>
  )
  return (
    <div className="flex min-h-screen">
      <aside className={`fixed inset-y-0 left-0 z-40 hidden w-64 border-r border-white/10 bg-panel/95 backdrop-blur lg:block ${collapsed ? 'lg:w-16' : ''}`}>{sidebar}</aside>
      {open && <div className="fixed inset-0 z-50 bg-black/60 lg:hidden" onClick={() => setOpen(false)}><aside className="h-full w-64 bg-panel" onClick={e => e.stopPropagation()}>{sidebar}</aside></div>}
      <div className={`flex min-w-0 flex-1 flex-col ${collapsed ? 'lg:pl-16' : 'lg:pl-64'}`}>
        <header className="sticky top-0 z-30 flex items-center gap-3 border-b border-white/10 bg-ink/85 px-4 py-3 backdrop-blur">
          <button className="btn-ghost lg:hidden" onClick={() => setOpen(true)} aria-label="Open menu"><Menu size={18} /></button>
          <button className="btn-ghost hidden lg:inline-flex" onClick={() => setCollapsed(!collapsed)} aria-label="Collapse sidebar"><Menu size={18} /></button>
          <div className="min-w-0">
            <h1 className="truncate text-base font-bold sm:text-lg">{NAV.find(n => n.id === page)?.label}</h1>
            <p className="hidden text-xs text-slate-400 sm:block">Real-Time Object Detection System · YOLOv8 &amp; Deep Learning</p>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <span className={`chip ${model?.loaded ? '' : '!bg-red-500/15 !text-red-300 !border-red-500/30'}`}>
              <Activity size={12} /> {model?.loaded ? `Model ready · ${model.device}` : 'Model not loaded'}
            </span>
            <button className="btn-primary hidden sm:inline-flex" onClick={() => setPage('live')}><Video size={15} /> Quick start</button>
          </div>
        </header>
        <main className="mx-auto w-full max-w-7xl flex-1 p-4 sm:p-6">{children}</main>
        <footer className="border-t border-white/5 px-6 py-3 text-center text-xs text-slate-500">
          ITA0523 Computer Vision capstone · YOLOv8 (Ultralytics) · PyTorch · OpenCV · All detections are real model inference — no simulated results.
        </footer>
      </div>
    </div>
  )
}

export function Toasts() {
  const [items, setItems] = useState<{ id: number; msg: string; kind: string }[]>([])
  useEffect(() => {
    const h = (e: any) => { setItems(v => [...v, e.detail]); setTimeout(() => setItems(v => v.filter(i => i.id !== e.detail.id)), 4500) }
    window.addEventListener('app-toast', h); return () => window.removeEventListener('app-toast', h)
  }, [])
  return (
    <div className="fixed bottom-4 right-4 z-[100] space-y-2">
      {items.map(t => (
        <div key={t.id} className={`flex items-center gap-2 rounded-xl border px-4 py-2.5 text-sm shadow-xl ${t.kind === 'err' ? 'border-red-500/40 bg-red-950/90 text-red-200' : 'border-sky-500/40 bg-slate-900/95 text-slate-100'}`}>
          <span>{t.msg}</span><button onClick={() => setItems(v => v.filter(i => i.id !== t.id))} aria-label="Dismiss"><X size={14} /></button>
        </div>
      ))}
    </div>
  )
}
