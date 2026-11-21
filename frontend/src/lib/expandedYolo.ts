import { type Detection, type Fit } from './yoloPost'

const modelRoot = 'https://huggingface.co/talmago/yoloe.onnx/resolve/4ea526e5f1962d756e5cc7e13360ba5008a0d672/'
export const EXPANDED_MODEL_URL = modelRoot + 'yoloe-11l-seg-pf.onnx'
export const EXPANDED_LABEL_COUNT = 4585

let labelRequest: Promise<string[]> | undefined

/**
 * Label list for the prompt-free YOLOE-11l export. Lines are formatted
 * `"<id>: <name>"`; the id prefix is stripped. The count is validated because
 * a short/partial list would mislabel every detection.
 */
export function expandedLabels(): Promise<string[]> {
  labelRequest ??= fetch(modelRoot + 'yoloe.names')
    .then(async response => {
      if (!response.ok) throw new Error(`Could not download the expanded object labels (HTTP ${response.status}).`)
      const lines = (await response.text()).trim().split(/\r?\n/).filter(line => line.trim())
      const labels = lines.map((line, index) => {
        const match = /^\s*(\d+)\s*:\s*(.*)$/.exec(line)
        return match ? match[2].trim() : line.trim()
      })
      if (labels.length < EXPANDED_LABEL_COUNT) {
        throw new Error(`The expanded model label list is incomplete (${labels.length} of ${EXPANDED_LABEL_COUNT}).`)
      }
      return labels.slice(0, EXPANDED_LABEL_COUNT)
    })
    .catch(error => {
      labelRequest = undefined
      throw error
    })
  return labelRequest
}

/**
 * Kept for backwards compatibility. The actual decoding lives in
 * yoloPost.parseRowDetections() so both decoders share one implementation and
 * one set of layout checks.
 */
export type { Detection, Fit }
