import { lazy, Suspense, useEffect, useState } from 'react'
import { Layout, Toasts, PageId } from './components/Layout'
import { isBackendOnline } from './lib/api'
import { EmptyState } from './components/ui'
const Dashboard = lazy(() => import('./pages/Dashboard'))
const Live = lazy(() => import('./pages/Live'))
const VideoDetect = lazy(() => import('./pages/VideoDetect'))
const ImageDetect = lazy(() => import('./pages/ImageDetect'))
const Dataset = lazy(() => import('./pages/Dataset'))
const Training = lazy(() => import('./pages/Training'))
const Inference = lazy(() => import('./pages/Inference'))
const History = lazy(() => import('./pages/History'))
const Analytics = lazy(() => import('./pages/Analytics'))
const Settings = lazy(() => import('./pages/Settings'))

export default function App() {
  const [page, setPage] = useState<PageId>('dashboard')
  useEffect(() => { window.scrollTo(0, 0) }, [page])
  const [online, setOnline] = useState<boolean | null>(null)
  useEffect(() => {
    let active = true
    const refresh = () => { void isBackendOnline().then(value => { if (active) setOnline(value) }) }
    refresh()
    window.addEventListener('focus', refresh)
    return () => { active = false; window.removeEventListener('focus', refresh) }
  }, [page])
  const needsServer = ['dataset', 'training', 'analytics'].includes(page)
  let content
  if (needsServer && online !== true) {
    content = online === null ? <p role="status">Checking backend connection…</p> : (
      <EmptyState title="Connect a Python backend"
        body="Dataset preparation, model training and server reports need a Python backend."
        action={<button className="btn-primary" onClick={() => setPage('settings')}>Open connection settings</button>} />
    )
  } else {
    switch (page) {
      case 'dashboard': content = <Dashboard online={online} go={setPage} />; break
      case 'live': content = <Live />; break
      case 'image': content = <ImageDetect />; break
      case 'video': content = <VideoDetect online={online} />; break
      case 'dataset': content = <Dataset />; break
      case 'training': content = <Training />; break
      case 'inference': content = <Inference online={online} go={setPage} />; break
      case 'history': content = <History />; break
      case 'analytics': content = <Analytics />; break
      case 'settings': content = <Settings online={online} />; break
    }
  }
  return (
    <Layout online={online} page={page} setPage={setPage}>
      <Suspense fallback={<p role="status">Loading workspace…</p>}>
        {content}
      </Suspense>
      <Toasts />
    </Layout>
  )
}
