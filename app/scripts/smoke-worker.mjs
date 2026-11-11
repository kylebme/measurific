#!/usr/bin/env node
// Starts the inspection worker exactly as the app does and analyzes a demo image.
//
//   node scripts/smoke-worker.mjs                  # bundled runtime in build/
//   node scripts/smoke-worker.mjs --resources DIR  # a packaged app's resources folder
import { spawn } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const guiRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const windows = process.platform === 'win32'
const at = process.argv.indexOf('--resources')
const layout = at > 0
  ? (() => {
    const resources = path.resolve(process.argv[at + 1])
    return { runtime: path.join(resources, 'python'), worker: path.join(resources, 'worker', 'worker.py'),
      model: path.join(resources, 'models', 'contour'), edgeModel: path.join(resources, 'models', 'edge'),
      demo: path.join(resources, 'demo-data') }
  })()
  : { runtime: path.join(guiRoot, 'build', 'python-runtime'), worker: path.join(guiRoot, 'python', 'worker.py'),
    model: path.join(guiRoot, '..', 'packages', 'contour-model', 'release-v6', 'onnx'),
    edgeModel: path.join(guiRoot, '..', 'packages', 'edge-model', 'models', 'sdf_v5-onnx'), demo: path.join(guiRoot, 'demo-data') }
const python = windows ? path.join(layout.runtime, 'python.exe') : path.join(layout.runtime, 'bin', 'python3')
const env = { ...process.env, PYTHONNOUSERSITE: '1', PYTHONDONTWRITEBYTECODE: '1' }
delete env.PYTHONPATH
delete env.PYTHONHOME

const child = spawn(python, [layout.worker, '--model', layout.model, '--edge-model', layout.edgeModel], { env, stdio: ['pipe', 'pipe', 'inherit'] })
const requests = [
  { id: 1, command: 'health' },
  // Arbitrary files exercise edge-model extraction; synthetic demos carry their contours.
  { id: 2, command: 'analyze', path: path.join(layout.demo, 'extras', '000000003.image.png') },
  { id: 3, command: 'analyze', path: path.join(layout.demo, 'drilled_plate.png') },
]
let buffered = ''
const results = new Map()
child.stdout.setEncoding('utf8')
child.stdout.on('data', (chunk) => {
  buffered += chunk
  for (let newline; (newline = buffered.indexOf('\n')) >= 0;) {
    const response = JSON.parse(buffered.slice(0, newline))
    buffered = buffered.slice(newline + 1)
    if (response.error) { console.error(`request ${response.id} failed: ${response.error}`); process.exit(1) }
    results.set(response.id, response.result)
    if (results.size === requests.length) child.stdin.end()
  }
})
child.on('exit', (code) => {
  if (results.size !== requests.length) { console.error(`worker exited (${code}) after ${results.size} responses`); process.exit(1) }
  const health = results.get(1)
  // The default edge method: PAE seeded by the SDF export's coarse model (exercises numba).
  if (health.edgeModel.format !== 'edgesr-coarse-pae-v1') { console.error('expected PAE edges with the coarse ONNX model'); process.exit(1) }
  console.log(`python ${health.python} · ${health.model.format} on ${health.model.providers.join(', ')} · ${health.edgeModel.format} on ${health.edgeModel.providers.join(', ')} · available ${health.availableProviders.join(', ')}`)
  if (health.packages.torch) { console.error('torch must not be bundled'); process.exit(1) }
  for (const id of [2, 3]) {
    const result = results.get(id)
    const features = result.prediction?.features?.length ?? 0
    console.log(`${result.name}: ${features} features, ${JSON.stringify(result.timingsMs)}`)
    if (!features) { console.error(`${result.name}: no features recognized`); process.exit(1) }
  }
  console.log('worker smoke test passed')
})
for (const request of requests) child.stdin.write(`${JSON.stringify(request)}\n`)
