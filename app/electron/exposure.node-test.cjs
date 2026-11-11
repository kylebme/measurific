const { test } = require('node:test')
const assert = require('node:assert/strict')
const { nextExposure } = require('./exposure.cjs')
const { CameraService } = require('./camera.cjs')
const control = { value: 40000, min: 12, max: 1000000 }
test('clipping overrides a safe percentile, including a tiny saturated region', () => {
  assert.equal(nextExposure(control, { count: 1000000, clipped: 1, peak: 1, percentile: .8 }), 32000)
})
test('settled exposure stays fixed; increases are damped and bounded', () => {
  assert.equal(nextExposure(control, { count: 100, clipped: 0, peak: .96, percentile: .9 }), 40000)
  assert.equal(nextExposure(control, { count: 100, clipped: 0, peak: .2, percentile: .1 }), 48000)
  assert.equal(nextExposure({ ...control, value: 12 }, { count: 100, clipped: 100, peak: 1, percentile: 1 }), 12)
})
test('ROI is sent to native metering and changes exposure with vendor auto disabled', async () => {
  const calls = []
  const service = new CameraService({ app: {}, guiRoot: '.', projectRoot: '.', send() {} })
  service.state = { phase: 'streaming', controls: { exposure: { ...control, available: true, autoAvailable: true }, gain: { auto: false } } }
  service.request = async (...args) => {
    calls.push(args)
    if (args[0] === 'meter') return { count: 1000, clipped: 10, peak: 1, percentile: 1 }
    return { controls: service.state.controls }
  }
  service.setRoi({ shape: 'circle', cx: .6, cy: .4, r: .2 })
  await service.setControl('exposureAuto', true)
  await service.adjustExposure()
  assert.deepEqual(calls[0], ['set', 'exposureAuto', false])
  assert.deepEqual(calls[1], ['meter', 'circle', .6, .4, .2, 0])
  assert.deepEqual(calls[2], ['set', 'exposure', 32000])
  assert.equal(service.publicState().controls.exposure.auto, true)
})
test('pause stops further automatic writes from pending metering', async () => {
  const service = new CameraService({ app: {}, guiRoot: '.', projectRoot: '.', send() {} })
  service.state = { phase: 'streaming', controls: { exposure: control } }; service.autoExposure = true
  let release
  const calls = []
  service.request = async (...args) => {
    calls.push(args)
    if (args[0] === 'meter') return new Promise(resolve => { release = resolve })
    return {}
  }
  service.autoPending = service.adjustExposure()
  const paused = service.pause(null)
  release({ count: 100, clipped: 100, peak: 1, percentile: 1 })
  await paused
  assert.equal(calls.some(call => call[0] === 'set'), false)
  assert.equal(service.state.phase, 'paused')
})

test('changing ROI discards in-flight statistics from the old region', async () => {
  const service = new CameraService({ app: {}, guiRoot: '.', projectRoot: '.', send() {} })
  service.state = { phase: 'streaming', controls: { exposure: control } }; service.autoExposure = true
  let release
  service.request = async command => {
    assert.equal(command, 'meter')
    return new Promise(resolve => { release = resolve })
  }
  const pending = service.adjustExposure()
  service.setRoi({ shape: 'rectangle', x0: .2, y0: .2, x1: .8, y1: .8 })
  release({ count: 100, clipped: 100, peak: 1, percentile: 1 })
  await pending
  assert.equal(service.nextMeterAt, 0)
  assert.throws(() => service.setRoi({ shape: 'circle', cx: .5, cy: .5, r: 0 }))
})
