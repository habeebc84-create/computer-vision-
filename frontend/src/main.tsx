import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

// After a new deploy, a browser holding the old build can fail to load a
// renamed JS chunk ("Failed to fetch dynamically imported module") and the
// page then looks broken until a manual hard reload. Reload once, on our
// own, when that exact failure happens — never for other errors, and never
// more than once per session (so a genuinely broken build cannot loop).
const STALE_CHUNK_RELOAD_KEY = 'ods-reloaded-once'
function reloadOnceForStaleChunk() {
  try {
    if (sessionStorage.getItem(STALE_CHUNK_RELOAD_KEY)) return
    sessionStorage.setItem(STALE_CHUNK_RELOAD_KEY, '1')
    window.location.reload()
  } catch { /* storage unavailable — leave the page as it is */ }
}
window.addEventListener('vite:preloadError', reloadOnceForStaleChunk)
window.addEventListener('error', (e) => {
  const msg = String((e as ErrorEvent).message || '')
  if (/dynamically imported module|Importing a module script failed|Loading chunk \d+ failed/i.test(msg)) reloadOnceForStaleChunk()
})
window.addEventListener('unhandledrejection', (e) => {
  const msg = String((e as PromiseRejectionEvent).reason?.message || (e as PromiseRejectionEvent).reason || '')
  if (/dynamically imported module|Importing a module script failed|Loading chunk \d+ failed/i.test(msg)) reloadOnceForStaleChunk()
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode><App /></React.StrictMode>
)
