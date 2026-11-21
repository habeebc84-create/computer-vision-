import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import ts from 'typescript'

const source = await readFile(new URL('../src/lib/yoloPost.ts', import.meta.url), 'utf8')
const js = ts.transpile(source, { module: ts.ModuleKind.ES2022, target: ts.ScriptTarget.ES2022 })
const { isRowOutput, parseDetections, fitParams } = await import(`data:text/javascript;base64,${Buffer.from(js).toString('base64')}`)
assert.equal(isRowOutput([1, 84, 8400], 80), false)
assert.equal(isRowOutput([1, 8400, 84], 80), false)
assert.equal(isRowOutput([1, 300, 6], 80), true)
const data = new Float32Array(84 * 2)
data[0] = 320; data[2] = 320; data[4] = 100; data[6] = 100; data[8] = 0.9
const boxes = parseDetections(data, 640, 640, fitParams(640, 640, 640, 'letterbox'), 0.25, 0.45, Array.from({ length: 80 }, (_, i) => i === 0 ? 'person' : String(i)), [1, 84, 2])
assert.equal(boxes.length, 1)
assert.equal(boxes[0].class_name, 'person')
console.log('YOLO output layout tests passed')
