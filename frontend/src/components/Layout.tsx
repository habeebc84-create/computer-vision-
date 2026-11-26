import { useEffect, useRef, useState } from 'react'
import { BarChart3, Camera, Database, History, Home, Image as ImageIcon, Layers, ScanEye, Settings as SettingsIcon, X, Menu, Cpu, PanelLeftClose, PanelLeftOpen, Video } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { api, ModelStatus } from '../lib/api'

export type PageId = 'dashboard' | 'live' | 'image' | 'video' | 'dataset' | 'training' | 'inference' | 'history' | 'analytics' | 'settings'
export const NAV: { id: PageId; label: string; icon: LucideIcon; group: string }[] = [
  { id: 'dashboard', label: 'Overview', icon: Home, group: 'Detect' },
  { id: 'live', label: 'Live Detection', icon: Camera, group: 'Detect' },
  { id: 'image', label: 'Image Detection', icon: ImageIcon, group: 'Detect' },
  { id: 'video', label: 'Video Detection', icon: Video, group: 'Detect' },
  { id: 'dataset', label: 'Dataset', icon: Database, group: 'Modules' },
  { id: 'training', label: 'Model Training', icon: Layers, group: 'Modules' },
  { id: 'inference', label: 'Inference', icon: ScanEye, group: 'Modules' },
  { id: 'history', label: 'Detection History', icon: History, group: 'Data' },
  { id: 'analytics', label: 'Analytics & Reports', icon: BarChart3, group: 'Data' },
  { id: 'settings', label: 'Settings', icon: SettingsIcon, group: 'System' },
]

export function Layout({ online, page, setPage, children }: { online: boolean | null; page: PageId; setPage: (p: PageId) => void; children: React.ReactNode }) {
  const menu = useRef<HTMLDialogElement>(null)
  const [collapsed, setCollapsed] = useState(false)
  const [model, setModel] = useState<ModelStatus | null>(null)
  useEffect(() => {
    let active = true
    if (online) api.modelStatus().then(value => { if (active) setModel(value) }).catch(() => {})
    return () => { active = false }
  }, [online, page])
  useEffect(() => {
    const desktop = window.matchMedia('(min-width: 1024px)')
    const closeOnDesktop = () => { if (desktop.matches) menu.current?.close() }
    desktop.addEventListener('change', closeOnDesktop)
    return () => desktop.removeEventListener('change', closeOnDesktop)
  }, [])
  const label = NAV.find(item => item.id === page)?.label
  const modelName = online ? model?.model_path?.split(/[\\/]/).pop() || 'Backend detector' : 'Browser detection'
  const choose = (id: PageId) => {
    menu.current?.close()
    setPage(id)
  }
  const sidebar = (compact: boolean, mobile = false) => (
    <div className="flex h-full min-h-0 flex-col">
      <div className={`flex h-[72px] shrink-0 items-center gap-3 border-b border-white/[0.07] ${compact ? 'justify-center' : 'px-5'}`}>
        <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-indigo-400 text-slate-950"><ScanEye size={21} /></span>
        {!compact && <span className="min-w-0 flex-1 text-sm font-bold tracking-tight">Object Detection</span>}
        {mobile && <button type="button" className="icon-button" onClick={() => menu.current?.close()} aria-label="Close navigation"><X size={18} /></button>}
      </div>
      <nav aria-label="Main navigation" className="flex-1 space-y-5 overflow-y-auto px-3 py-5">
        {['Detect', 'Modules', 'Data', 'System'].map(group => (
          <div key={group} className="space-y-1">
            {!compact && <p className="px-3 pb-1 text-[11px] font-semibold uppercase tracking-[.12em] text-slate-400">{group}</p>}
            {NAV.filter(item => item.group === group).map(item => (
              <button key={item.id} type="button" title={item.label} aria-label={item.label}
                aria-current={page === item.id ? 'page' : undefined}
                onClick={() => choose(item.id)}
                className={`navitem ${page === item.id ? 'active' : ''} ${compact ? 'justify-center !px-0' : ''}`}>
                <item.icon size={17} className="shrink-0" />{!compact && item.label}
              </button>
            ))}
          </div>
        ))}
      </nav>
      {!compact && <div className="mx-3 mb-4 flex items-start gap-3 rounded-xl border border-white/10 bg-white/[0.035] p-3">
        <Cpu size={16} className="mt-0.5 shrink-0 text-indigo-300" />
        <div className="min-w-0 text-xs">
          <p className="truncate font-semibold text-slate-200" title={modelName}>{modelName}</p>
          <p className="mt-1 text-slate-400" >{online === null ? 'Checking connection…' : online ? model?.device || 'Server' : 'Image, live and video'}</p>
        </div>
      </div>}
    </div>
  )
  return (
    <div className="min-h-screen">
      <a href="#main-content" className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-50 focus:rounded-md focus:bg-indigo-300 focus:p-2 focus:text-slate-950">Skip to content</a>
      <aside className={`fixed inset-y-0 left-0 z-40 hidden border-r border-white/[0.07] bg-panel lg:block ${collapsed ? 'w-[72px]' : 'w-60'}`}>{sidebar(collapsed)}</aside>
      <dialog ref={menu} aria-label="Navigation" className="navigation-drawer" onClick={event => { if (event.target === event.currentTarget) menu.current?.close() }}>
        {sidebar(false, true)}
      </dialog>
      <div className={`flex min-h-screen min-w-0 flex-col ${collapsed ? 'lg:pl-[72px]' : 'lg:pl-60'}`}>
        <header className="sticky top-0 z-30 border-b border-white/[0.07] bg-ink/95 backdrop-blur">
          <div className="mx-auto flex h-[71px] max-w-[1440px] items-center gap-3 px-4 sm:px-7">
            <button type="button" className="icon-button lg:hidden" onClick={() => menu.current?.showModal()} aria-label="Open navigation"><Menu size={19} /></button>
            <button type="button" className="icon-button hidden lg:inline-grid" onClick={() => setCollapsed(value => !value)} aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}>
              {collapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
            </button>
            <h1 className="min-w-0 truncate text-lg font-semibold tracking-tight sm:text-xl">{label}</h1>
            <span role="status" className="ml-auto inline-flex shrink-0 items-center gap-2 rounded-full border border-white/10 px-2.5 py-1.5 text-xs text-slate-300 sm:px-3">
              <span className={`h-2 w-2 rounded-full ${online === true ? 'bg-emerald-400' : online === false ? 'bg-sky-400' : 'bg-slate-400'}`} />
              {online === true ? 'Connected' : online === false ? 'Browser mode' : 'Checking…'}
            </span>
          </div>
        </header>
        <main id="main-content" tabIndex={-1} className="mx-auto w-full min-w-0 max-w-[1440px] flex-1 p-4 focus:outline-none sm:p-7">{children}</main>
        <footer className="border-t border-white/[0.07] px-4 py-4 text-center text-xs text-slate-400">ITA0523 Computer Vision · YOLOv8</footer>
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
    <div aria-live="polite" className="fixed bottom-4 right-4 z-[100] max-w-[calc(100vw-2rem)] space-y-2">
      {items.map(t => (
        <div key={t.id} className={`flex items-center gap-2 rounded-xl border px-4 py-2.5 text-sm shadow-xl ${t.kind === 'err' ? 'border-red-500/40 bg-red-950/90 text-red-200' : 'border-sky-500/40 bg-slate-900/95 text-slate-100'}`}>
          <span>{t.msg}</span><button onClick={() => setItems(v => v.filter(i => i.id !== t.id))} aria-label="Dismiss"><X size={14} /></button>
        </div>
      ))}
    </div>
  )
}
