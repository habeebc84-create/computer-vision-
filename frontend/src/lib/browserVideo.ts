import { detectInBrowser, type BrowserDetectResult } from './browserDetection'

function waitForVideo(video: HTMLVideoElement, event: string, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    signal.throwIfAborted()
    const cleanup = () => {
      clearTimeout(timer)
      video.removeEventListener(event, ready)
      video.removeEventListener('error', failed)
      signal.removeEventListener('abort', aborted)
    }
    const ready = () => { cleanup(); resolve() }
    const failed = () => { cleanup(); reject(new Error('Could not decode this video. Try an MP4 with H.264 video or a WebM file.')) }
    const aborted = () => { cleanup(); reject(signal.reason) }
    const timer = setTimeout(failed, 15_000)
    video.addEventListener(event, ready)
    video.addEventListener('error', failed)
    signal.addEventListener('abort', aborted)
  })
}

export interface VideoSample {
  time: number
  result: BrowserDetectResult
}

export async function sampleVideo(
  file: File,
  signal: AbortSignal,
  onFrame: (sample: VideoSample, done: number, total: number) => void,
) {
  const video = document.createElement('video')
  video.muted = true
  video.preload = 'auto'
  const url = URL.createObjectURL(file)
  try {
    const loaded = waitForVideo(video, 'loadeddata', signal)
    video.src = url
    await loaded
    if (!Number.isFinite(video.duration) || video.duration <= 0 || !video.videoWidth || !video.videoHeight) {
      throw new Error('The video has no readable frames or duration.')
    }
    const total = Math.min(24, Math.max(1, Math.ceil(video.duration)))
    for (let i = 0; i < total; i++) {
      const time = video.duration * (i + 0.5) / total
      if (Math.abs(video.currentTime - time) > 0.001) {
        const sought = waitForVideo(video, 'seeked', signal)
        video.currentTime = time
        await sought
      }
      signal.throwIfAborted()
      // Bound the resolution of the retained annotated previews.
      const scale = Math.min(1, 960 / Math.max(video.videoWidth, video.videoHeight))
      const frame = document.createElement('canvas')
      frame.width = Math.max(1, Math.round(video.videoWidth * scale))
      frame.height = Math.max(1, Math.round(video.videoHeight * scale))
      frame.getContext('2d')!.drawImage(video, 0, 0, frame.width, frame.height)
      const result = await detectInBrowser(frame, frame.width, frame.height, undefined, 'broad')
      signal.throwIfAborted()
      onFrame({ time, result }, i + 1, total)
    }
  } finally {
    video.removeAttribute('src')
    video.load()
    URL.revokeObjectURL(url)
  }
}
