const assert = require('node:assert/strict')
const test = require('node:test')
const { CameraPacketDecoder, PacketKind, cropGrayscaleFrame, encodeGrayscalePng } = require('./camera-protocol.cjs')

function packet(kind, header, payload = Buffer.alloc(0)) {
  const json = Buffer.from(JSON.stringify(header))
  const prefix = Buffer.alloc(16)
  prefix.write('ICAM')
  prefix[4] = 1
  prefix[5] = kind
  prefix.writeUInt32LE(json.length, 8)
  prefix.writeUInt32LE(payload.length, 12)
  return Buffer.concat([prefix, json, payload])
}

test('packet decoder handles split and adjacent packets', () => {
  const found = []
  const decoder = new CameraPacketDecoder((value) => found.push(value))
  const bytes = Buffer.concat([
    packet(PacketKind.json, { event: 'ready' }),
    packet(PacketKind.preview, { width: 2, height: 1 }, Buffer.from([3, 9])),
  ])
  decoder.push(bytes.subarray(0, 11))
  decoder.push(bytes.subarray(11, 29))
  decoder.push(bytes.subarray(29))
  assert.equal(found.length, 2)
  assert.equal(found[1].header.width, 2)
  assert.deepEqual(found[1].payload, Buffer.from([3, 9]))
})

test('PNG encoder preserves 8 and 16-bit grayscale declarations', () => {
  const png8 = encodeGrayscalePng({ width: 2, height: 1, storageBits: 8 }, Buffer.from([0, 255]))
  const png16 = encodeGrayscalePng({ width: 2, height: 1, storageBits: 16 }, Buffer.from([0x34, 0x12, 0xcd, 0xab]))
  assert.deepEqual(png8.subarray(1, 4), Buffer.from('PNG'))
  assert.equal(png8[24], 8)
  assert.equal(png16[24], 16)
  assert.throws(() => encodeGrayscalePng({ width: 2, height: 1, storageBits: 16 }, Buffer.alloc(2)), /size/)
})

test('camera ROI crops full-depth pixels and creates a circular analysis mask', () => {
  const bytes = Buffer.from(Array.from({ length: 192 }, (_, value) => value))
  const rectangle = cropGrayscaleFrame(
    { width: 16, height: 12, storageBits: 8 }, bytes,
    { shape: 'rectangle', x0: .25, y0: 1 / 6, x1: .75, y1: 5 / 6 },
  )
  assert.deepEqual([rectangle.meta.width, rectangle.meta.height], [8, 8])
  assert.deepEqual(rectangle.pixels.subarray(0, 8), Buffer.from([36, 37, 38, 39, 40, 41, 42, 43]))
  assert.equal(rectangle.analysisMask, null)

  const sixteen = Buffer.alloc(20 * 20 * 2)
  for (let i = 0; i < 400; i += 1) sixteen.writeUInt16LE(i, i * 2)
  const circle = cropGrayscaleFrame(
    { width: 20, height: 20, storageBits: 16 }, sixteen,
    { shape: 'circle', cx: .5, cy: .5, r: .25 },
  )
  assert.deepEqual([circle.meta.width, circle.meta.height], [10, 10])
  assert.deepEqual(circle.analysisMask, { shape: 'circle', cx: 5, cy: 5, r: 5 })
  assert.equal(circle.pixels.readUInt16LE(0), 0xffff)
  assert.equal(circle.pixels.readUInt16LE((5 * 10 + 5) * 2), 210)
})
