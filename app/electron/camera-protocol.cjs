const { deflateSync } = require('node:zlib')

const MAGIC = Buffer.from('ICAM')
const HEADER_BYTES = 16
const PROTOCOL_VERSION = 1

const PacketKind = Object.freeze({ json: 1, preview: 2, snapshot: 3 })

class CameraPacketDecoder {
  constructor(onPacket) {
    this.onPacket = onPacket
    this.buffer = Buffer.alloc(0)
  }

  push(chunk) {
    this.buffer = this.buffer.length ? Buffer.concat([this.buffer, chunk]) : Buffer.from(chunk)
    while (this.buffer.length >= HEADER_BYTES) {
      if (!this.buffer.subarray(0, 4).equals(MAGIC)) {
        const next = this.buffer.indexOf(MAGIC, 1)
        this.buffer = next >= 0 ? this.buffer.subarray(next) : this.buffer.subarray(Math.max(0, this.buffer.length - 3))
        if (this.buffer.length < HEADER_BYTES) return
      }
      const version = this.buffer.readUInt8(4)
      if (version !== PROTOCOL_VERSION) throw new Error(`Unsupported camera bridge protocol ${version}`)
      const kind = this.buffer.readUInt8(5)
      const headerLength = this.buffer.readUInt32LE(8)
      const payloadLength = this.buffer.readUInt32LE(12)
      if (headerLength > 1024 * 1024 || payloadLength > 1024 * 1024 * 1024) {
        throw new Error('Camera bridge sent an invalid packet length')
      }
      const packetLength = HEADER_BYTES + headerLength + payloadLength
      if (this.buffer.length < packetLength) return
      const headerStart = HEADER_BYTES
      const payloadStart = headerStart + headerLength
      const header = JSON.parse(this.buffer.toString('utf8', headerStart, payloadStart))
      const payload = Buffer.from(this.buffer.subarray(payloadStart, packetLength))
      this.buffer = this.buffer.subarray(packetLength)
      this.onPacket({ kind, header, payload })
    }
  }
}

function crc32(buffer) {
  let crc = 0xffffffff
  for (const byte of buffer) {
    crc ^= byte
    for (let bit = 0; bit < 8; bit += 1) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1))
  }
  return (crc ^ 0xffffffff) >>> 0
}

function pngChunk(name, data) {
  const type = Buffer.from(name, 'ascii')
  const result = Buffer.allocUnsafe(12 + data.length)
  result.writeUInt32BE(data.length, 0)
  type.copy(result, 4)
  data.copy(result, 8)
  result.writeUInt32BE(crc32(Buffer.concat([type, data])), 8 + data.length)
  return result
}

function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value))
}

/** Crop a tightly packed grayscale frame using image-normalized ROI coordinates. */
function cropGrayscaleFrame(meta, pixels, roi) {
  if (!roi) return { meta, pixels, analysisMask: null }
  const { width, height, storageBits } = meta
  if (!Number.isInteger(width) || !Number.isInteger(height) || ![8, 16].includes(storageBits)) throw new Error('Invalid camera frame metadata')
  const bytesPerPixel = storageBits / 8
  if (pixels.length !== width * height * bytesPerPixel) throw new Error('Camera frame size does not match its dimensions')

  let left, top, right, bottom
  if (roi.shape === 'rectangle') {
    const values = [roi.x0, roi.y0, roi.x1, roi.y1]
    if (!values.every(Number.isFinite)) throw new Error('Invalid rectangular ROI')
    left = Math.floor(clamp(Math.min(roi.x0, roi.x1), 0, 1) * width)
    top = Math.floor(clamp(Math.min(roi.y0, roi.y1), 0, 1) * height)
    right = Math.ceil(clamp(Math.max(roi.x0, roi.x1), 0, 1) * width)
    bottom = Math.ceil(clamp(Math.max(roi.y0, roi.y1), 0, 1) * height)
  } else if (roi.shape === 'circle') {
    if (![roi.cx, roi.cy, roi.r].every(Number.isFinite) || roi.r <= 0) throw new Error('Invalid circular ROI')
    const radius = roi.r * Math.min(width, height)
    const centerX = roi.cx * width, centerY = roi.cy * height
    left = Math.floor(clamp(centerX - radius, 0, width))
    top = Math.floor(clamp(centerY - radius, 0, height))
    right = Math.ceil(clamp(centerX + radius, 0, width))
    bottom = Math.ceil(clamp(centerY + radius, 0, height))
  } else {
    throw new Error('Unknown camera ROI shape')
  }
  left = clamp(left, 0, width - 1); top = clamp(top, 0, height - 1)
  right = clamp(right, left + 1, width); bottom = clamp(bottom, top + 1, height)
  const croppedWidth = right - left, croppedHeight = bottom - top
  if (croppedWidth < 8 || croppedHeight < 8) throw new Error('Camera ROI is too small')
  const output = Buffer.allocUnsafe(croppedWidth * croppedHeight * bytesPerPixel)
  const sourceStride = width * bytesPerPixel, targetStride = croppedWidth * bytesPerPixel
  for (let y = 0; y < croppedHeight; y += 1) {
    pixels.copy(output, y * targetStride, (top + y) * sourceStride + left * bytesPerPixel, (top + y) * sourceStride + right * bytesPerPixel)
  }

  let analysisMask = null
  if (roi.shape === 'circle') {
    const radius = roi.r * Math.min(width, height)
    const centerX = roi.cx * width - left, centerY = roi.cy * height - top
    const fill = storageBits === 8 ? 0xff : 0xffff
    const squaredRadius = radius * radius
    for (let y = 0; y < croppedHeight; y += 1) {
      for (let x = 0; x < croppedWidth; x += 1) {
        const dx = x + .5 - centerX, dy = y + .5 - centerY
        if (dx * dx + dy * dy <= squaredRadius) continue
        const offset = (y * croppedWidth + x) * bytesPerPixel
        output[offset] = fill & 0xff
        if (bytesPerPixel === 2) output[offset + 1] = fill >> 8
      }
    }
    analysisMask = { shape: 'circle', cx: centerX, cy: centerY, r: radius }
  }
  return {
    meta: { ...meta, width: croppedWidth, height: croppedHeight, roi: { ...roi }, roiOffsetX: left, roiOffsetY: top },
    pixels: output,
    analysisMask,
  }
}

/** Encode the bridge's tightly packed grayscale frame without discarding 16-bit data. */
function encodeGrayscalePng(meta, pixels) {
  const { width, height, storageBits } = meta
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) throw new Error('Invalid camera frame dimensions')
  if (storageBits !== 8 && storageBits !== 16) throw new Error(`Unsupported grayscale storage depth ${storageBits}`)
  const bytesPerPixel = storageBits / 8
  if (pixels.length !== width * height * bytesPerPixel) throw new Error('Camera frame size does not match its dimensions')
  const stride = width * bytesPerPixel
  const scanlines = Buffer.allocUnsafe((stride + 1) * height)
  for (let y = 0; y < height; y += 1) {
    const target = y * (stride + 1)
    scanlines[target] = 0
    if (storageBits === 8) {
      pixels.copy(scanlines, target + 1, y * stride, (y + 1) * stride)
    } else {
      // The bridge protocol is little-endian; PNG grayscale samples are network order.
      for (let x = 0; x < stride; x += 2) {
        scanlines[target + 1 + x] = pixels[y * stride + x + 1]
        scanlines[target + 2 + x] = pixels[y * stride + x]
      }
    }
  }
  const ihdr = Buffer.alloc(13)
  ihdr.writeUInt32BE(width, 0)
  ihdr.writeUInt32BE(height, 4)
  ihdr[8] = storageBits
  ihdr[9] = 0 // grayscale
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    pngChunk('IHDR', ihdr),
    pngChunk('IDAT', deflateSync(scanlines, { level: 1 })),
    pngChunk('IEND', Buffer.alloc(0)),
  ])
}

module.exports = { CameraPacketDecoder, PacketKind, cropGrayscaleFrame, encodeGrayscalePng }
