const test = require('node:test')
const assert = require('node:assert/strict')
const { FocusTracker } = require('./focus.cjs')
const { CameraService } = require('./camera.cjs')
const { PacketKind } = require('./camera-protocol.cjs')
const ok = (sequence, score = 10) => ({ sequence, score, count: 100, status: 'ok' })

function service() {
  const events = [], requests = []
  const camera = new CameraService({ app: { isPackaged: false }, guiRoot: '.', projectRoot: '.', send: (...args) => events.push(args) })
  camera.state.phase = 'streaming'
  camera.request = async (...args) => { requests.push(args); return ok(requests.length) }
  return { camera, events, requests }
}

test('time-based smoothing retains a peak, excludes duplicate frames, and resets', () => {
  const tracker = new FocusTracker()
  assert.equal(tracker.update(ok(1), 0).score, 10)
  assert.equal(tracker.update(ok(1, 100), 100), null)
  const lower = tracker.update(ok(2, 5), 200)
  assert.ok(Math.abs(lower.score - (5+5/Math.E)) < 1e-9)
  assert.equal(lower.peak, 10)
  assert.equal(tracker.update({ sequence: 3, status: 'low-detail' }, 400).score, null)
  assert.equal(tracker.update(ok(4, 8), 600).score, 8, 'invalid data clears smoothing')
  tracker.reset()
  assert.equal(tracker.update(ok(1, 3), 800).peak, 3)
})

test('native focus receives the selected ROI and late replies are discarded on ROI change or pause', async () => {
  for (const change of ['roi', 'pause']) {
    const { camera, events, requests } = service()
    camera.setRoi({ shape: 'circle', cx: .6, cy: .4, r: .2 })
    let resolve
    camera.request = (...args) => { requests.push(args); return new Promise(done => { resolve = done }) }
    const pending = camera.measureFocus()
    assert.deepEqual(requests[0], ['focus', 'circle', .6, .4, .2, 0])
    if (change === 'roi') camera.setRoi(null)
    else camera.publish({ phase: 'pausing' })
    resolve(ok(1))
    await pending
    assert.equal(events.filter(([name, value]) => name === 'inspection:camera-focus' && value).length, 0)
  }
})

test('focus computation is throttled, never queued, and absent for paused images', async () => {
  const { camera, requests } = service()
  const packet = { kind: PacketKind.preview, header: { sequence: 1 }, payload: Buffer.alloc(0) }
  let resolve
  camera.request = (...args) => { requests.push(args); return new Promise(done => { resolve = done }) }
  await camera.onPacket(packet)
  await camera.onPacket(packet)
  assert.equal(requests.length, 1)
  resolve(ok(1)); await camera.focusPending
  await camera.onPacket(packet)
  assert.equal(requests.length, 1, '200 ms rate cap')
  camera.publish({ phase: 'paused' })
  camera.nextFocusAt = 0
  await camera.onPacket(packet)
  assert.equal(requests.length, 1, 'no measurement in paused view')
})

test('focus failure leaves the camera streaming and reports unavailable feedback', async () => {
  const { camera, events } = service()
  camera.request = async () => { throw new Error('Unknown command') }
  await camera.measureFocus()
  assert.equal(camera.state.phase, 'streaming')
  assert.equal(events.at(-1)[1].status, 'unavailable')
})

test('acquisition changes also discard focus replies requested while a setting was applying', async () => {
  const { camera, events } = service()
  let resolveSet, resolveFocus
  camera.request = async (command) => new Promise(resolve => {
    if (command === 'set') resolveSet = resolve
    else resolveFocus = resolve
  })
  const setting = camera.setControl('gain', 2)
  const measurement = camera.measureFocus()
  resolveSet({ controls: null }); await setting
  resolveFocus(ok(99)); await measurement
  assert.equal(events.filter(([name, value]) => name === 'inspection:camera-focus' && value).length, 0)
})
