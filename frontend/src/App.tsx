import { useState } from 'react'
import { Layout, Toasts, PageId } from './components/Layout'
import Dashboard from './pages/Dashboard'
import Live from './pages/Live'
import ImageDetect from './pages/ImageDetect'
import Dataset from './pages/Dataset'
import Training from './pages/Training'
import Inference from './pages/Inference'
import History from './pages/History'
import Analytics from './pages/Analytics'
import Settings from './pages/Settings'

export default function App() {
  const [page, setPage] = useState<PageId>('dashboard')
  return (
    <Layout page={page} setPage={setPage}>
      {page === 'dashboard' && <Dashboard go={setPage} />}
      {page === 'live' && <Live />}
      {page === 'image' && <ImageDetect />}
      {page === 'dataset' && <Dataset />}
      {page === 'training' && <Training />}
      {page === 'inference' && <Inference go={setPage} />}
      {page === 'history' && <History />}
      {page === 'analytics' && <Analytics />}
      {page === 'settings' && <Settings />}
      <Toasts />
    </Layout>
  )
}
