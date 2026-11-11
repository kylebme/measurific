const assert = require('node:assert/strict')
const { spawn } = require('node:child_process')
const path = require('node:path')
const { CameraPacketDecoder } = require('../../app/electron/camera-protocol.cjs')

const executable = process.argv[2]
if (!executable) throw new Error('Usage: node smoke.cjs /path/to/inspection-camera-bridge')

const fake = process.argv.includes('--fake')
const camera = process.argv.includes('--camera')
const exercise = fake || camera
const child = spawn(path.resolve(executable), [], {
  stdio: ['pipe', 'pipe', 'pipe'],
  env: { ...process.env, ...(fake ? { INSPECTION_CAMERA_ENABLE_FAKE: '1' } : {}) },
})
const responses = new Map()
let preview = null
let snapshot = null
const decoder = new CameraPacketDecoder((packet) => {
  const { header } = packet
  if (packet.kind === 2) {
    preview = packet
    child.stdin.write('pause\t4\n')
    return
  }
  if (packet.kind === 3) {
    snapshot = packet
    child.stdin.write('quit\t5\n')
    return
  }
  if (header.requestId != null) responses.set(header.requestId, header)
  if (header.requestId === 1 && exercise) {
    const id = header.result.devices[0]?.id
    assert.ok(id, fake ? 'Aravis fake camera was not discovered' : 'No camera was discovered')
    child.stdin.write(`open\t2\t${encodeURIComponent(id)}\n`)
  } else if (header.requestId === 2) {
    // Color cameras deliver one Bayer site; check red before streaming.
    if (header.result?.camera?.channels?.includes('red')) child.stdin.write('set\t6\tchannel\tred\n')
    else child.stdin.write('start\t3\n')
  } else if (header.requestId === 6) {
    child.stdin.write('start\t3\n')
  } else if (header.requestId === 3 && !preview) {
    // The stream thread will deliver the preview and trigger pause.
  }
})
child.stdout.on('data', (chunk) => decoder.push(chunk))
child.stderr.pipe(process.stderr)
child.stdin.write('discover\t1\n')
if (!exercise) child.stdin.end('quit\t2\n')

const timeout = setTimeout(() => {
  child.kill()
  throw new Error('Camera bridge smoke test timed out')
}, 5000)

child.on('exit', (code) => {
  clearTimeout(timeout)
  assert.equal(code, 0)
  assert.equal(responses.get(1)?.ok, true)
  assert.ok(Array.isArray(responses.get(1)?.result?.devices))
  if (exercise) {
    assert.equal(responses.get(2)?.ok, true)
    assert.equal(responses.get(3)?.ok, true)
    assert.ok(preview?.payload.length > 0)
    assert.equal(preview.header.storageBits, 8)
    assert.ok(snapshot?.payload.length >= preview.payload.length)
    assert.ok([8, 16].includes(snapshot.header.storageBits))
    const opened = responses.get(2).result.camera
    if (opened.channels.includes('red')) {
      assert.equal(responses.get(6)?.ok, true)
      assert.equal(snapshot.header.channel, 'red')
      assert.equal(snapshot.header.sampleStep, 2)
      assert.equal(snapshot.header.width, Math.ceil((opened.sensorWidth - snapshot.header.sampleOffsetX) / 2))
      assert.equal(snapshot.header.height, Math.ceil((opened.sensorHeight - snapshot.header.sampleOffsetY) / 2))
    } else {
      assert.equal(snapshot.header.channel, 'camera')
    }
    assert.equal(responses.get(5)?.ok, true)
  } else {
    assert.equal(responses.get(2)?.ok, true)
  }
  console.log(`camera bridge smoke test passed (${responses.get(1).result.devices.length} device(s))`)
})
