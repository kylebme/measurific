const test = require('node:test')
const assert = require('node:assert/strict')
const { CameraService } = require('./camera.cjs')

const numeric = { available: true, autoAvailable: true, auto: false, value: 1, min: 0, max: 2 }
const controls = { exposure: numeric, gain: numeric }

function service(camera, channel) {
  const sent = []
  const cameraService = new CameraService({ app: { isPackaged: false }, guiRoot: '.', projectRoot: '.', send: () => {}, channel })
  cameraService.child = {} // an open bridge
  cameraService.state = { ...cameraService.state, phase: 'streaming', camera, controls }
  cameraService.request = async (...args) => {
    sent.push(args)
    return { camera: { ...cameraService.state.camera, channel: args[2] }, controls }
  }
  return { cameraService, sent }
}

const color = { id: 'basler', channel: 'camera', channels: ['camera', 'red', 'green', 'blue'] }

test('an open color camera switches to the selected channel', async () => {
  const { cameraService, sent } = service(color, 'red')
  await cameraService.applyChannel()
  assert.deepEqual(sent, [['set', 'channel', 'red']])
  assert.equal(cameraService.state.camera.channel, 'red')
  await cameraService.setChannel('red')
  assert.equal(sent.length, 1, 'an unchanged channel sends nothing')
  await cameraService.setChannel('camera')
  assert.deepEqual(sent.at(-1), ['set', 'channel', 'camera'])
})

test('a camera without raw Bayer output keeps its own image', async () => {
  const { cameraService, sent } = service({ ...color, channels: ['camera'] }, 'red')
  await cameraService.setChannel('blue')
  assert.deepEqual(sent, [])
  assert.equal(cameraService.channel, 'blue', 'the preference is kept for the next camera')
})

test('the choice is remembered while no camera is open', async () => {
  const { cameraService, sent } = service(null, 'red')
  cameraService.child = null
  await cameraService.setChannel('green')
  assert.equal(cameraService.channel, 'green')
  assert.deepEqual(sent, [])
})
