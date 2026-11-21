import { suppressDuplicates, type Detection } from './yoloPost'

const modelRoot = 'https://huggingface.co/talmago/yoloe.onnx/resolve/4ea526e5f1962d756e5cc7e13360ba5008a0d672/'
export const EXPANDED_MODEL_URL = modelRoot + 'yoloe-11l-seg-pf.onnx'
export const EXPANDED_LABEL_COUNT = 4585

let labelRequest: Promise<string[]> | undefined

export function expandedLabels() {
  labelRequest ??= fetch(modelRoot + 'yoloe.names').then(async response => {
    if (!response.ok) throw new Error('Could not download expanded object labels.')
    const lines = (await response.text()).trim().split(/\r?\n/)
    const labels = lines.map((line, index) => {
      const prefix = `${index}: `
      return line.startsWith(prefix) ? line.slice(prefix.length).trim() : ''
    })
    if (labels.length !== EXPANDED_LABEL_COUNT || labels.some(label => !label)) {
      throw new Error('The expanded model label list is invalid.')
    }
    return labels
  }).catch(error => {
    labelRequest = undefined
    throw error
  })
  return labelRequest
}

/** Read the exported rows: x1, y1, x2, y2, confidence, class ID, mask coefficients. */
export function parseExpandedDetections(
  output: Float32Array, dims: readonly number[], labels: string[],
  width: number, height: number, scale: number, padX: number, padY: number,
  confidenceThreshold: number, iouThreshold: number,
): Detection[] {
  const [batch, rows, stride] = dims
  if (dims.length !== 3 || batch !== 1 || stride !== 38 || output.length !== rows * stride) {
    throw new Error('The expanded detection model returned an unexpected output.')
  }
  const detections: Detection[] = []
  for (let offset = 0; offset < output.length; offset += stride) {
    const confidence = output[offset + 4]
    const classId = output[offset + 5]
    if (!Number.isFinite(confidence) || confidence < confidenceThreshold || !Number.isInteger(classId) || !labels[classId]) continue
    const x1 = Math.max(0, (output[offset] - padX) / scale)
    const y1 = Math.max(0, (output[offset + 1] - padY) / scale)
    const x2 = Math.min(width, (output[offset + 2] - padX) / scale)
    const y2 = Math.min(height, (output[offset + 3] - padY) / scale)
    if (!(x2 > x1 && y2 > y1)) continue
    detections.push({ class_id: classId, class_name: labels[classId], confidence, bbox: [x1, y1, x2, y2] })
  }
  return suppressDuplicates(detections, iouThreshold, 0.9)
}
