import { useEffect, useRef, useState } from 'react'
import { Download, ImagePlus, Plus, Split, Trash2, Upload } from 'lucide-react'
import { api, BuiltinDataset, resolveUrl, toast } from '../lib/api'

interface Box { class_id: number; x_min: number; y_min: number; x_max: number; y_max: number }

export default function Dataset() {
  const [stats, setStats] = useState<any>(null)
  const [images, setImages] = useState<any[]>([])
  const [classes, setClasses] = useState<string[]>([])
  const [newClass, setNewClass] = useState('')
  const [selected, setSelected] = useState<any>(null)
  const [ann, setAnn] = useState<{ width: number; height: number; boxes: Box[] }>({ width: 0, height: 0, boxes: [] })
  const [drawClass, setDrawClass] = useState(0)
  const [trainPct, setTrainPct] = useState(80)
  const [yaml, setYaml] = useState('')
  const [uploadErrors, setUploadErrors] = useState<{ file: string; message: string }[]>([])
  const [busy, setBusy] = useState(false)
  const [catalog, setCatalog] = useState<BuiltinDataset[]>([])
  const [dsBusy, setDsBusy] = useState('')

  const refreshCatalog = async () => {
    try { setCatalog(await api.dsCatalog()) } catch { /* backend offline — banner in Layout explains */ }
  }
  useEffect(() => { refreshCatalog() }, [])
  // Poll only while a download is in progress.
  useEffect(() => {
    if (!catalog.some(d => d.status === 'downloading')) return
    const t = setInterval(refreshCatalog, 2000)
    return () => clearInterval(t)
  }, [catalog])
  const downloadDataset = async (id: string, force = false) => {
    setDsBusy(id)
    try {
      await api.dsDownload(id, force); toast('Dataset download started — importing in the background')
      await refreshCatalog()
    } catch (e: any) { toast(e.message, 'err') } finally { setDsBusy('') }
  }
  const imgRef = useRef<HTMLImageElement>(null)
  const drawStart = useRef<{ x: number; y: number } | null>(null)
  const [draft, setDraft] = useState<Box | null>(null)

  const refresh = async () => {
    const [s, im, c] = await Promise.all([api.dsStats(), api.dsImages(), api.dsClasses()])
    setStats(s); setImages(im); setClasses(c.classes)
  }
  useEffect(() => { refresh().catch(() => null) }, [])

  const upload = async (files: FileList | null) => {
    if (!files?.length) return
    setBusy(true)
    setUploadErrors([])
    try {
      const result = await api.dsUpload(files)
      setUploadErrors(result.errors ?? result.skipped.map(file => ({ file, message: 'The backend skipped this file.' })))
      toast(`Uploaded ${result.saved.length} image(s)`)
      await refresh()
    } catch (e: any) { toast(e.message, 'err') }
    finally { setBusy(false) }
  }
  const selectImage = async (im: any) => {
    setSelected(im)
    try { const a = await api.dsAnnotations(im.name); setAnn(a) } catch (e: any) { toast(e.message, 'err') }
  }
  const addClass = async () => {
    if (!newClass.trim()) return
    const next = [...classes, newClass.trim()]
    try {
      await api.dsSetClasses(next)
      setClasses(next); setNewClass(''); toast(`Class “${newClass.trim()}” added`)
    } catch (e: any) { toast(e.message, 'err') }
  }
  const saveBoxes = async () => {
    if (!selected) return
    try {
      await api.dsSaveAnnotations(selected.name, ann.boxes)
      toast(`Saved ${ann.boxes.length} annotation(s) for ${selected.name}`)
      await refresh()
    } catch (e: any) { toast(e.message, 'err') }
  }
  const split = async () => {
    setBusy(true)
    try {
      const r = await api.dsSplit(trainPct); await refresh()
      const y = await api.dsYaml().catch(() => null); if (y) setYaml(y.content)
      toast(`Split done — ${r.train} train / ${r.val} val${r.warning ? '. ' + r.warning : ''}`)
    } catch (e: any) { toast(e.message, 'err') } finally { setBusy(false) }
  }
  const onDown = (e: React.MouseEvent) => {
    const img = imgRef.current; if (!img || !classes.length) return
    const r = img.getBoundingClientRect()
    const x = (e.clientX - r.left) / r.width * ann.width, y = (e.clientY - r.top) / r.height * ann.height
    drawStart.current = { x, y }
  }
  const onMove = (e: React.MouseEvent) => {
    if (!drawStart.current || !imgRef.current) return
    const img = imgRef.current; const r = img.getBoundingClientRect()
    const x = (e.clientX - r.left) / r.width * ann.width, y = (e.clientY - r.top) / r.height * ann.height
    setDraft({ class_id: drawClass, x_min: Math.min(drawStart.current.x, x), y_min: Math.min(drawStart.current.y, y), x_max: Math.max(drawStart.current.x, x), y_max: Math.max(drawStart.current.y, y) })
  }
  const onUp = () => {
    if (draft && draft.x_max - draft.x_min > 5 && draft.y_max - draft.y_min > 5) setAnn(a => ({ ...a, boxes: [...a.boxes, draft] }))
    setDraft(null); drawStart.current = null
  }
  const scaleX = (imgRef.current?.clientWidth || 1) / (ann.width || 1)
  const scaleY = (imgRef.current?.clientHeight || 1) / (ann.height || 1)

  return (
    <div className="space-y-4">
      <div className="card">
        <h2 className="font-semibold">Recommended datasets — one-click download</h2>
        <p className="mt-1 text-sm text-slate-400">Official Ultralytics COCO samples with real images and real YOLO labels. Download one, then Split the dataset below and go to Module 2 — Training.</p>
        <div className="mt-3 grid gap-3 md:grid-cols-2">
          {catalog.map(d => (
            <div key={d.id} className="rounded-xl border border-white/10 bg-white/[0.03] p-4">
              <div className="flex items-center justify-between gap-2">
                <p className="font-semibold">{d.name}</p>
                <span className="chip">{d.images} images · {d.classes} classes · {d.size_mb} MB</span>
              </div>
              <p className="mt-1 text-sm text-slate-400">{d.description}</p>
              <p className="mt-1 text-xs text-slate-500">Licence: {d.license}</p>
              <div className="mt-3 flex flex-wrap items-center gap-2">
                {d.status === 'imported'
                  ? <><span className="chip !bg-emerald-500/15 !text-emerald-300 !border-emerald-500/30">Imported ✓ {d.imported_images} images · {d.imported_labels} labels</span>
                    <button className="btn-ghost" onClick={() => { downloadDataset(d.id, true); refresh() }} disabled={dsBusy === d.id}>Re-download</button></>
                  : d.status === 'downloading'
                    ? <span className="chip">Downloading… {d.progress ?? 0}%</span>
                    : d.status === 'failed'
                      ? <><span className="chip !bg-red-500/15 !text-red-300 !border-red-500/30">Failed: {d.error || 'download error'}</span>
                        <button className="btn-primary" onClick={() => downloadDataset(d.id)} disabled={dsBusy === d.id}>Retry download</button></>
                      : <button className="btn-primary" onClick={() => downloadDataset(d.id)} disabled={dsBusy === d.id}><Download size={15} /> {dsBusy === d.id ? 'Starting…' : 'Download & import'}</button>}
                {d.status === 'imported' && <button className="btn-ghost" onClick={refresh}>Refresh stats</button>}
              </div>
            </div>
          ))}
          {!catalog.length && <p className="text-sm text-slate-400">Could not load the dataset catalog — is the backend running?</p>}
        </div>
      </div>
      <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
        {[['Total images', stats?.total_images ?? 0], ['Annotations', stats?.total_annotations ?? 0], ['Classes', stats?.num_classes ?? 0], ['Train images', stats?.train_images ?? 0], ['Val images', stats?.val_images ?? 0]].map(([l, v]) => (
          <div key={l as string} className="card"><p className="text-xs uppercase tracking-wider text-slate-400">{l}</p><p className="mt-1 text-2xl font-bold">{v}</p></div>
        ))}
      </div>
      <div className="grid gap-4 lg:grid-cols-3">
        <div className="card">
          <h2 className="font-semibold">A. Image collection</h2>
          <label className="mt-3 flex cursor-pointer flex-col items-center rounded-xl border-2 border-dashed border-white/15 px-4 py-6 text-center text-sm hover:border-sky-500/50">
            <ImagePlus size={22} className="text-sky-400" /><span className="mt-1">Upload JPG, PNG, WebP, GIF, BMP or AVIF</span>
            <input type="file" accept="image/jpeg,image/png,image/webp,image/gif,image/bmp,image/avif" multiple disabled={busy} className="hidden" onChange={e => {
              void upload(e.currentTarget.files)
              e.currentTarget.value = ''
            }} />
          </label>
          {uploadErrors.length > 0 && <ul role="alert" className="mt-2 space-y-1 text-sm text-red-300">
            {uploadErrors.map(({ file, message }, index) => <li key={index}><b>{file}</b> · {message}</li>)}
          </ul>}
          <div className="mt-3 max-h-72 space-y-2 overflow-y-auto">
            {images.map(im => (
              <button key={im.name} onClick={() => selectImage(im)} className={`flex w-full items-center justify-between rounded-xl border px-3 py-2 text-left text-sm ${selected?.name === im.name ? 'border-sky-500/50 bg-sky-500/10' : 'border-white/10 bg-white/[0.03]'}`}>
                <span className="truncate">{im.name}<span className="block text-xs text-slate-400">{im.width}×{im.height} · {im.annotations} box(es) · {im.size_kb} KB</span></span>
                <span onClick={e => { e.stopPropagation(); api.dsDelete(im.name).then(refresh) }} className="p-1 text-slate-400 hover:text-red-300" role="button" aria-label={`Delete ${im.name}`}><Trash2 size={15} /></span>
              </button>
            ))}
            {!images.length && <p className="text-sm text-slate-400">No images yet — upload some to begin.</p>}
          </div>
          <h2 className="mt-5 font-semibold">C. Classes</h2>
          <div className="mt-2 flex flex-wrap gap-2">{classes.map((c, i) => <span key={i} className="chip">{i}: {c}</span>)}{!classes.length && <span className="text-sm text-slate-400">No classes defined yet.</span>}</div>
          <div className="mt-2 flex gap-2"><input className="input" placeholder="New class name, e.g. helmet" value={newClass} onChange={e => setNewClass(e.target.value)} onKeyDown={e => e.key === 'Enter' && addClass()} /><button className="btn-primary" onClick={addClass}><Plus size={15} /> Add</button></div>
        </div>
        <div className="card lg:col-span-2">
          <h2 className="font-semibold">B. Annotation — LabelImg-compatible (YOLO txt)</h2>
          {!selected ? <p className="mt-3 text-sm text-slate-400">Select an image on the left, then drag on it to draw a bounding box. Boxes are saved as YOLO text labels (class, centre-x, centre-y, width, height — normalised).</p> : (
            <>
              <div className="mt-3 flex flex-wrap items-center gap-2">
                <label className="text-xs text-slate-400">Box class</label>
                <select className="input max-w-[200px]" value={drawClass} onChange={e => setDrawClass(+e.target.value)}>
                  {classes.map((c, i) => <option key={i} value={i}>{i}: {c}</option>)}
                </select>
                <button className="btn-primary" onClick={saveBoxes}>Save annotations ({ann.boxes.length})</button>
                <button className="btn-ghost" onClick={() => setAnn(a => ({ ...a, boxes: [] }))}>Clear boxes</button>
              </div>
              <div className="relative mt-3 inline-block max-w-full">
                <img ref={imgRef} src={resolveUrl(`/dataset/images/all/${encodeURIComponent(selected.name)}`)} alt={selected.name} className="max-h-[480px] w-auto max-w-full select-none rounded-xl border border-white/10"
                  onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp} draggable={false} />
                {[...ann.boxes, ...(draft ? [draft] : [])].map((b, i) => (
                  <div key={i} className="absolute border-2 border-sky-400 bg-sky-400/10" style={{ left: b.x_min * scaleX, top: b.y_min * scaleY, width: (b.x_max - b.x_min) * scaleX, height: (b.y_max - b.y_min) * scaleY }}>
                    <span className="absolute -top-5 left-0 whitespace-nowrap bg-sky-500 px-1.5 py-0.5 text-[11px] font-semibold text-white">{classes[b.class_id] ?? b.class_id}</span>
                    <button className="absolute -right-2 -top-2 rounded-full bg-red-500 px-1.5 text-[11px] text-white" onClick={() => setAnn(a => ({ ...a, boxes: a.boxes.filter((_, j) => j !== i) }))} aria-label="Delete box">×</button>
                  </div>
                ))}
              </div>
              <p className="mt-2 text-xs text-slate-500">Note: the preview above is served from the uploaded dataset folder; if it does not load, the file is still stored and usable for training — check the backend uploads.</p>
            </>
          )}
          <h2 className="mt-6 font-semibold">D. Dataset split &amp; data.yaml</h2>
          <div className="mt-2 flex flex-wrap items-center gap-3">
            <label className="text-sm text-slate-300">Train <input type="number" className="input !w-20" value={trainPct} min={50} max={95} onChange={e => setTrainPct(+e.target.value)} /> %</label>
            <span className="text-sm text-slate-400">Val {100 - trainPct}%</span>
            <button className="btn-primary" onClick={split} disabled={busy}><Split size={15} /> Split &amp; generate data.yaml</button>
            <a className="btn-ghost" href={resolveUrl('/api/dataset/export')}><Download size={15} /> Export dataset (.zip)</a>
          </div>
          {yaml && <pre className="mt-3 overflow-x-auto rounded-xl bg-black/40 p-3 font-mono text-xs text-emerald-300">{yaml}</pre>}
          {stats?.per_class && <div className="mt-4"><h3 className="text-sm font-semibold">E. Images per class</h3><div className="mt-2 flex flex-wrap gap-2">{Object.entries(stats.per_class).map(([k, v]) => <span key={k} className="chip">{k}: {String(v)} annotation(s)</span>)}</div></div>}
        </div>
      </div>
    </div>
  )
}
