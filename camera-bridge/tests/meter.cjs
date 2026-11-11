// Native integration check: meter exactly the pixels in the frozen snapshot.
const assert = require('node:assert/strict')
const { spawn } = require('node:child_process')
const path = require('node:path')
const { CameraPacketDecoder } = require('../../app/electron/camera-protocol.cjs')
const child = spawn(path.resolve(process.argv[2]), [], { env: { ...process.env, INSPECTION_CAMERA_ENABLE_FAKE: '1' }, stdio: ['pipe', 'pipe', 'inherit'] })
let next = 1, firstFrame, snapshot
const pending = new Map(), ready = new Promise(resolve => { firstFrame = resolve })
function request(command, ...args) {
  const id = next++
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject })
    child.stdin.write([command, id, ...args.map(v => encodeURIComponent(String(v)))].join('\t')+'\n')
  })
}
const decoder = new CameraPacketDecoder(p => {
  if (p.kind === 2) { firstFrame(); return }
  const slot = pending.get(p.header.requestId)
  if (!slot) return
  pending.delete(p.header.requestId)
  if (p.kind === 3) { snapshot = p; slot.resolve(p); return }
  if (p.header.ok === false) slot.reject(new Error(p.header.error))
  else slot.resolve(p.header.result)
})
child.stdout.on('data', data => decoder.push(data))
const timeout = setTimeout(() => { child.kill(); process.exitCode = 1 }, 10000)
;(async () => {
 try {
  const { devices } = await request('discover')
  const fake = devices.find(d => /fake/i.test(d.id+' '+d.model+' '+d.protocol))
  assert.ok(fake, 'Aravis fake camera must be available')
  await request('open', fake.id); await request('start'); await ready; await request('pause')
  const { width, height, storageBits, sensorBits } = snapshot.header
  const maximum = sensorBits <= 8 ? 255 : ((2**sensorBits-1) * 2**(16-sensorBits))
  for (const [shape, a, b, c, d] of [['rectangle', .2, .3, .8, .7], ['circle', .6, .45, .2, 0]]) {
    const stats = await request('meter', shape, a, b, c, d)
    const values = []
    for (let y=0; y<height; y++) for (let x=0; x<width; x++) {
      const inside = shape === 'circle' ? (x+.5-a*width)**2+(y+.5-b*height)**2 <= (c*Math.min(width,height))**2 : x+.5>=a*width && x+.5<c*width && y+.5>=b*height && y+.5<d*height
      if (inside) values.push(storageBits === 8 ? snapshot.payload[y*width+x] : snapshot.payload.readUInt16LE(2*(y*width+x)))
    }
    values.sort((a,b)=>a-b)
    assert.equal(stats.count, values.length)
    assert.equal(stats.clipped, values.filter(v=>v>=maximum).length)
    assert.ok(Math.abs(stats.peak-values.at(-1)/maximum)<1e-9)
    assert.ok(Math.abs(stats.percentile-values[Math.ceil(values.length*.999)-1]/maximum)<1e-9)
    assert.equal(stats.sequence, snapshot.header.sequence)
  }
  for (const roi of [['rectangle', .2, .3, .8, .7], ['circle', .6, .45, .2, 0]]) {
    const focus = await request('focus', ...roi)
    assert.ok(['ok', 'low-detail', 'too-small'].includes(focus.status))
    assert.ok(Number.isFinite(focus.score) && focus.score >= 0)
    assert.ok(focus.count > 0 && focus.count <= 65536)
    assert.equal(focus.sequence, snapshot.header.sequence)
    assert.deepEqual(await request('focus', ...roi), focus, 'frozen frames give repeatable scores')
  }
  await assert.rejects(request('focus', 'rectangle', .8, .2, .1, .9), /Empty/)
  await assert.rejects(request('meter', 'rectangle', .8, .2, .1, .9), /Empty/)
  console.log('Native rectangle/circle metering matches frozen sensor samples; focus is bounded and repeatable')
 } finally { await request('quit'); clearTimeout(timeout) }
})().catch(error => { console.error(error); child.kill(); clearTimeout(timeout); process.exitCode = 1 })
