import type { HistoryRecord } from './api'

export interface BrowserHistoryRecord extends Omit<HistoryRecord, 'id'> {
  id: string
  original_name?: string
  has_preview: boolean
}

interface BrowserHistoryInput {
  inputType: string
  count: number
  classes: string[]
  avgConfidence: number
  inferenceMs: number
  fps: number
  model: string
  name?: string
  previewDataUrl?: string
}

interface HistoryFilters {
  search?: string
  type?: string
  from?: string
  to?: string
}

const RECORDS = 'records'
const PREVIEWS = 'previews'
const UNAVAILABLE = 'Browser history is unavailable. Check whether this site is allowed to store data.'
export const BROWSER_HISTORY_CHANGED = 'browser-history-changed'

function openHistory(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('ods-detection-history', 1)
    let blocked = false
    request.onupgradeneeded = () => {
      request.result.createObjectStore(RECORDS, { keyPath: 'id' })
      request.result.createObjectStore(PREVIEWS)
    }
    request.onblocked = () => {
      blocked = true
      reject(new Error('Browser history is open in another tab. Close that tab and try again.'))
    }
    request.onsuccess = () => {
      const db = request.result
      if (blocked) { db.close(); return }
      db.onversionchange = () => db.close()
      resolve(db)
    }
    request.onerror = () => reject(new Error(UNAVAILABLE))
  })
}

async function withHistory<T>(operation: (db: IDBDatabase) => Promise<T>): Promise<T> {
  if (typeof indexedDB === 'undefined') throw new Error(UNAVAILABLE)
  let db: IDBDatabase | undefined
  try {
    db = await openHistory()
    return await operation(db)
  } catch (error) {
    if (error instanceof DOMException) {
      if (error.name === 'QuotaExceededError') throw new Error('Browser storage is full. Free some space and try again.')
      throw new Error(UNAVAILABLE)
    }
    throw error
  } finally {
    db?.close()
  }
}

function committed(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve()
    transaction.onabort = () => reject(transaction.error || new Error(UNAVAILABLE))
  })
}

function resultOf<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error || new Error(UNAVAILABLE))
  })
}

export async function saveBrowserHistory(input: BrowserHistoryInput): Promise<BrowserHistoryRecord> {
  const preview = input.previewDataUrl?.startsWith('data:image/')
    ? await fetch(input.previewDataUrl).then(response => response.blob()).catch(() => undefined)
    : undefined
  const record: BrowserHistoryRecord = {
    id: Array.from(crypto.getRandomValues(new Uint8Array(16)), byte => byte.toString(16).padStart(2, '0')).join(''),
    created_at: new Date().toISOString(),
    input_type: input.inputType,
    classes: Array.from(new Set(input.classes)).sort(),
    object_count: input.count,
    avg_conf: input.avgConfidence,
    fps: input.fps,
    inference_ms: input.inferenceMs,
    output_path: '',
    model: input.model,
    original_name: input.name,
    has_preview: Boolean(preview?.size),
  }

  await withHistory(async db => {
    const transaction = db.transaction([RECORDS, PREVIEWS], 'readwrite')
    transaction.objectStore(RECORDS).add(record)
    if (record.has_preview) transaction.objectStore(PREVIEWS).add(preview, record.id)
    await committed(transaction)
  })
  window.dispatchEvent(new Event(BROWSER_HISTORY_CHANGED))
  return record
}

function localDate(timestamp: string): string {
  const date = new Date(timestamp)
  return [date.getFullYear(), date.getMonth() + 1, date.getDate()]
    .map((part, index) => String(part).padStart(index === 0 ? 4 : 2, '0')).join('-')
}

export async function listBrowserHistory(filters: HistoryFilters = {}): Promise<BrowserHistoryRecord[]> {
  const records = await withHistory(async db => {
    const transaction = db.transaction(RECORDS, 'readonly')
    const [records] = await Promise.all([
      resultOf<BrowserHistoryRecord[]>(transaction.objectStore(RECORDS).getAll()),
      committed(transaction),
    ])
    return records
  })
  const query = filters.search?.trim().toLowerCase()
  return records.filter(record => {
    if (filters.type && record.input_type !== filters.type) return false
    if (query && ![record.input_type, record.original_name || '', ...record.classes]
      .some(value => value.toLowerCase().includes(query))) return false
    const date = localDate(record.created_at)
    return (!filters.from || date >= filters.from) && (!filters.to || date <= filters.to)
  }).sort((a, b) => b.created_at.localeCompare(a.created_at))
}

export function getBrowserHistoryPreview(id: string): Promise<Blob | undefined> {
  return withHistory(async db => {
    const transaction = db.transaction(PREVIEWS, 'readonly')
    const [preview] = await Promise.all([
      resultOf<Blob | undefined>(transaction.objectStore(PREVIEWS).get(id)),
      committed(transaction),
    ])
    return preview
  })
}

export async function deleteBrowserHistory(id: string): Promise<void> {
  await withHistory(async db => {
    const transaction = db.transaction([RECORDS, PREVIEWS], 'readwrite')
    transaction.objectStore(RECORDS).delete(id)
    transaction.objectStore(PREVIEWS).delete(id)
    await committed(transaction)
  })
  window.dispatchEvent(new Event(BROWSER_HISTORY_CHANGED))
}
