// Downloads the pinned "Everyday objects" (Objects365) ONNX export into
// public/models/ so the browser can run it without a backend.
//
// This runs in `predev`/`prebuild`. It must never fail the build: a temporary
// Hugging Face outage (or a missing file) used to abort `npm run build`, which
// took the whole site down until someone reran the deploy. When the download
// cannot be completed the app still builds and simply reports that this
// detector's weights are unavailable, offering a retry.
import { createHash } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { mkdir, mkdtemp, rename, rm, stat, writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { join } from 'node:path'
import { EnvHttpProxyAgent, fetch } from 'undici'

const modelDirectory = fileURLToPath(new URL('../public/models/', import.meta.url))
const target = join(modelDirectory, 'objects365.onnx')
const expectedSize = 38_000_000
const source = 'https://huggingface.co/wide-video/yoloe-11-Objects365-v1.0.0/resolve/93d84d557920494a1070fe0dbd71a87be339d89a/yoloe-11s-seg-object365.onnx'

async function digest(path) {
  const hash = createHash('sha256')
  for await (const chunk of createReadStream(path)) hash.update(chunk)
  return hash.digest('hex')
}

/** A previously downloaded file is reused when it is a plausible ONNX blob. */
async function usable(path) {
  try {
    const info = await stat(path)
    if (info.size < 1_000_000) return false
    // ONNX files are protobuf; an HTML error page is not.
    const head = []
    for await (const chunk of createReadStream(path, { start: 0, end: 3 })) head.push(...chunk)
    return head[0] !== 0x3c // '<'
  } catch { return false }
}

async function prepare() {
  await mkdir(modelDirectory, { recursive: true })
  if (await usable(target)) {
    if (process.env.ODS_VERIFY_MODELS === '1') {
      console.log(`Everyday objects detector present (SHA-256 ${await digest(target)}).`)
    }
    return
  }

  const tempDir = await mkdtemp(join(modelDirectory, '.download-'))
  const temp = join(tempDir, 'model')
  const proxy = new EnvHttpProxyAgent()
  try {
    console.log('Downloading the Everyday objects detector (~38 MB)…')
    const response = await fetch(source, { dispatcher: proxy, signal: AbortSignal.timeout(180_000) })
    if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`)
    await writeFile(temp, response.body)
    if (!(await usable(temp))) throw new Error('the downloaded file is not an ONNX model')
    const { size } = await stat(temp)
    if (size < expectedSize * 0.8) throw new Error(`the download looks truncated (${size} bytes)`)
    await rename(temp, target)
    console.log('Everyday objects detector ready.')
  } catch (error) {
    // Warn, do not throw: a failed optional download must not break the build,
    // and the app surfaces this detector's unavailability at runtime.
    console.warn(
      `\n[prepare-image-model] Could not download the Everyday objects detector: ${error.message}\n` +
      '  The site still builds. That one mode will show "weights unavailable" in the UI\n' +
      '  and can be retried there, or rerun `npm run build` with network access.\n',
    )
  } finally {
    await proxy.close()
    await rm(tempDir, { recursive: true, force: true })
  }
}

await prepare()
