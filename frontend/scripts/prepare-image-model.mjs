import { createHash } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { mkdir, mkdtemp, rename, rm, writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { join } from 'node:path'
import { EnvHttpProxyAgent, fetch } from 'undici'

const modelDirectory = fileURLToPath(new URL('../public/models/', import.meta.url))
const target = join(modelDirectory, 'objects365.onnx')
const expected = '4bd6925933413ac257ddb8dbd2c9ae7091150baffe6f7b41dcac580e889edb19'
const source = 'https://huggingface.co/wide-video/yoloe-11-Objects365-v1.0.0/resolve/93d84d557920494a1070fe0dbd71a87be339d89a/yoloe-11s-seg-object365.onnx'

async function digest(path) {
  const hash = createHash('sha256')
  for await (const chunk of createReadStream(path)) hash.update(chunk)
  return hash.digest('hex')
}

async function prepare() {
  await mkdir(modelDirectory, { recursive: true })
  try {
    if (await digest(target) === expected) return
  } catch (error) {
    if (error.code !== 'ENOENT') throw error
  }

  const tempDir = await mkdtemp(join(modelDirectory, '.download-'))
  const temp = join(tempDir, 'model')
  const proxy = new EnvHttpProxyAgent()
  try {
    console.log('Downloading the Everyday objects detector (38 MB)…')
    const response = await fetch(source, { dispatcher: proxy, signal: AbortSignal.timeout(180_000) })
    if (!response.ok || !response.body) throw new Error(`Model download failed (HTTP ${response.status})`)
    await writeFile(temp, response.body)
    if (await digest(temp) !== expected) throw new Error('Everyday objects model failed its SHA-256 check')
    await rename(temp, target)
  } finally {
    await proxy.close()
    await rm(tempDir, { recursive: true, force: true })
  }
}

await prepare()
