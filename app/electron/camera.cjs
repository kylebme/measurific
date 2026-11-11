const { spawn } = require('node:child_process')
const { mkdir, writeFile } = require('node:fs/promises')
const path = require('node:path')
const { FocusTracker } = require('./focus.cjs')
const { nextExposure } = require('./exposure.cjs')
const { CameraPacketDecoder, PacketKind, cropGrayscaleFrame, encodeGrayscalePng } = require('./camera-protocol.cjs')

class CameraService {
  constructor({ app, guiRoot, projectRoot, send, channel = 'red' }) {
    this.app = app
    this.channel = channel
    this.guiRoot = guiRoot
    this.projectRoot = projectRoot
    this.send = send
    this.child = null
    this.nextId = 1
    this.pending = new Map()
    this.snapshotRoi = null
    this.meterRoi = null
    this.meterRevision = 0
    this.autoExposure = false
    this.autoPending = null
    this.nextMeterAt = 0
    this.focusTracker = new FocusTracker()
    this.focusRevision = 0
    this.focusPending = null
    this.nextFocusAt = 0
    this.state = { phase: 'idle', devices: [], camera: null, controls: null, error: null }
  }

  executablePath() {
    if (process.env.INSPECTION_CAMERA_BRIDGE) return process.env.INSPECTION_CAMERA_BRIDGE
    if (this.app.isPackaged) return path.join(process.resourcesPath, 'camera-bridge', 'bin', process.platform === 'win32' ? 'inspection-camera-bridge.exe' : 'inspection-camera-bridge')
    const platform = process.platform === 'darwin' ? 'macos' : process.platform
    return path.join(this.projectRoot, 'camera-bridge', 'dist', `${platform}-${process.arch}`, 'bin', process.platform === 'win32' ? 'inspection-camera-bridge.exe' : 'inspection-camera-bridge')
  }

  publicState() {
    const controls = this.state.controls ? { ...this.state.controls, exposure: {
      ...this.state.controls.exposure, autoAvailable: this.state.controls.exposure.available, auto: this.autoExposure,
    } } : null
    return { ...this.state, controls, bridgePath: this.executablePath() }
  }

  setRoi(roi) {
    if (roi) {
      const values = roi.shape === 'circle' ? [roi.cx, roi.cy, roi.r] : roi.shape === 'rectangle' ? [roi.x0, roi.y0, roi.x1, roi.y1] : []
      if (!values.length || !values.every(v => Number.isFinite(v) && v >= 0 && v <= 1) ||
          (roi.shape === 'circle' ? roi.r <= 0 : roi.x1 <= roi.x0 || roi.y1 <= roi.y0)) throw new Error('Invalid metering ROI')
    }
    this.resetFocus()
    this.meterRevision += 1
    this.meterRoi = roi ? { ...roi } : null
    this.nextMeterAt = 0
  }

  async adjustExposure() {
    const revision = this.meterRevision
    const roi = this.meterRoi || { shape: 'rectangle', x0: 0, y0: 0, x1: 1, y1: 1 }
    const args = roi.shape === 'circle' ? [roi.shape, roi.cx, roi.cy, roi.r, 0] : [roi.shape, roi.x0, roi.y0, roi.x1, roi.y1]
    const meter = await this.request('meter', ...args)
    if (!this.autoExposure || this.state.phase !== 'streaming' || revision !== this.meterRevision) return
    const control = this.state.controls.exposure
    const value = nextExposure(control, meter)
    if (Math.abs(value - control.value) > Math.max(.1, control.value * .001)) {
      this.resetFocus()
      const result = await this.request('set', 'exposure', value)
      this.resetFocus()
      this.publish({ controls: result.controls, error: null })
    }
    // Allow in-flight camera buffers and the sensor exposure to settle.
    this.nextMeterAt = Date.now() + Math.max(500, value / 1000 * 10)
  }

  resetFocus() {
    this.focusRevision += 1
    this.focusTracker.reset()
    this.nextFocusAt = 0
    this.send('inspection:camera-focus', null)
  }

  async measureFocus() {
    const revision = this.focusRevision
    const roi = this.meterRoi || { shape: 'rectangle', x0: 0, y0: 0, x1: 1, y1: 1 }
    const args = roi.shape === 'circle' ? [roi.shape, roi.cx, roi.cy, roi.r, 0] : [roi.shape, roi.x0, roi.y0, roi.x1, roi.y1]
    try {
      const sample = await this.request('focus', ...args)
      if (revision !== this.focusRevision || this.state.phase !== 'streaming') return
      const feedback = this.focusTracker.update(sample, Date.now())
      if (feedback) this.send('inspection:camera-focus', feedback)
    } catch {
      if (revision === this.focusRevision && this.state.phase === 'streaming')
        this.send('inspection:camera-focus', { score: null, peak: null, status: 'unavailable', updatedAt: Date.now() })
    }
  }

  publish(patch) {
    if (patch.phase && patch.phase !== this.state.phase) this.resetFocus()
    this.state = { ...this.state, ...patch }
    this.send('inspection:camera-state', this.publicState())
  }

  ensureBridge() {
    if (this.child) return
    const executable = this.executablePath()
    const libraryDir = path.resolve(executable, '..', '..', 'lib')
    const environment = { ...process.env }
    if (process.platform === 'darwin') environment.DYLD_LIBRARY_PATH = [libraryDir, environment.DYLD_LIBRARY_PATH].filter(Boolean).join(':')
    if (process.platform === 'linux') environment.LD_LIBRARY_PATH = [libraryDir, environment.LD_LIBRARY_PATH].filter(Boolean).join(':')
    this.child = spawn(executable, [], { cwd: path.dirname(executable), env: environment, stdio: ['pipe', 'pipe', 'pipe'] })
    const decoder = new CameraPacketDecoder((packet) => void this.onPacket(packet))
    this.child.stdout.on('data', (chunk) => {
      try { decoder.push(chunk) } catch (error) { this.fail(error) }
    })
    this.child.stderr.setEncoding('utf8')
    this.child.stderr.on('data', (chunk) => console.error(`[inspection-camera] ${chunk.trimEnd()}`))
    this.child.on('error', (error) => this.bridgeStopped(new Error(`Camera bridge could not start: ${error.message}`)))
    this.child.stdin.on('error', (error) => this.bridgeStopped(new Error(`Camera bridge input failed: ${error.message}`)))
    this.child.on('close', (code, signal) => {
      const expected = this.state.phase === 'closed'
      this.child = null
      if (!expected) this.bridgeStopped(new Error(`Camera bridge stopped (${signal || `exit ${code}`})`))
    })
  }

  bridgeStopped(error) {
    for (const { reject } of this.pending.values()) reject(error)
    this.pending.clear()
    this.fail(error)
  }

  fail(error) {
    const message = error instanceof Error ? error.message : String(error)
    this.publish({ phase: 'error', error: message })
  }

  request(command, ...args) {
    try { this.ensureBridge() } catch (error) { return Promise.reject(error) }
    const id = this.nextId++
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      const encoded = args.map((arg) => encodeURIComponent(String(arg)))
      this.child.stdin.write([command, id, ...encoded].join('\t') + '\n')
    })
  }

  async onPacket({ kind, header, payload }) {
    if (kind === PacketKind.preview) {
      // Preview frames are capped by the native process and are always 8-bit.
      this.send('inspection:camera-frame', header, payload)
      if (this.state.phase === 'streaming' && !this.focusPending && Date.now() >= this.nextFocusAt) {
        this.nextFocusAt = Date.now() + 200
        this.focusPending = this.measureFocus().finally(() => { this.focusPending = null })
      }
      if (this.autoExposure && this.state.phase === 'streaming' && !this.autoPending && Date.now() >= this.nextMeterAt) {
        this.autoPending = this.adjustExposure().catch((error) => {
          this.autoExposure = false
          this.publish({ error: `Automatic exposure stopped: ${error.message}` })
        }).finally(() => { this.autoPending = null })
      }
      return
    }
    if (kind === PacketKind.snapshot) {
      try {
        const directory = path.join(this.app.getPath('temp'), 'measurific')
        await mkdir(directory, { recursive: true })
        const snapshotPath = path.join(directory, `camera-${Date.now()}.png`)
        const cropped = cropGrayscaleFrame(header, payload, this.snapshotRoi)
        const png = encodeGrayscalePng(cropped.meta, cropped.pixels)
        await writeFile(snapshotPath, png)
        this.resolve(header.requestId, {
          snapshotPath,
          imageUrl: `data:image/png;base64,${png.toString('base64')}`,
          frame: cropped.meta,
          analysisMask: cropped.analysisMask,
        })
      } catch (error) {
        this.reject(header.requestId, error)
      }
      return
    }
    if (kind !== PacketKind.json) return
    if (header.event === 'stream-error') this.publish({ phase: 'error', error: header.error || 'Camera stream failed' })
    if (header.requestId != null) {
      if (header.ok === false) this.reject(header.requestId, new Error(header.error || 'Camera command failed'))
      else this.resolve(header.requestId, header.result)
    }
  }

  resolve(id, value) {
    const slot = this.pending.get(Number(id))
    if (!slot) return
    this.pending.delete(Number(id)); slot.resolve(value)
  }

  reject(id, error) {
    const slot = this.pending.get(Number(id))
    if (!slot) return
    this.pending.delete(Number(id)); slot.reject(error instanceof Error ? error : new Error(String(error)))
  }

  async discover() {
    const result = await this.request('discover')
    this.publish({ devices: result.devices || [], error: null })
    return result.devices || []
  }

  async start(deviceId) {
    this.publish({ phase: 'starting', error: null })
    try {
      const devices = await this.discover()
      const wanted = deviceId || this.state.camera?.id || devices[0]?.id
      if (!wanted) throw new Error('No USB3 Vision or GigE Vision camera was found')
      if (this.state.camera?.id !== wanted) {
        const opened = await this.request('open', wanted)
        this.publish({ camera: opened.camera, controls: opened.controls })
        await this.applyChannel()
        if (opened.controls.exposure.auto) await this.setControl('exposureAuto', true)
      }
      const started = await this.request('start')
      this.publish({ phase: 'streaming', camera: started.camera || this.state.camera, controls: started.controls || this.state.controls, error: null })
      return this.publicState()
    } catch (error) {
      this.fail(error)
      throw error
    }
  }

  async pause(roi) {
    if (this.state.phase !== 'streaming') throw new Error('Camera is not streaming')
    this.publish({ phase: 'pausing' })
    this.snapshotRoi = roi || null
    try {
      if (this.autoPending) await this.autoPending
      // The native process answers this request with a full-resolution snapshot packet.
      const result = await this.request('pause')
      this.publish({ phase: 'paused', error: null })
      return result
    } catch (error) {
      this.fail(error)
      throw error
    } finally {
      this.snapshotRoi = null
    }
  }

  /** Select the delivered image: 'red' | 'green' | 'blue' (one Bayer site at half
   * resolution) or 'camera' (the camera's own grayscale output). Cameras without raw
   * Bayer output only offer 'camera'. Takes effect immediately on an open camera. */
  async setChannel(channel) {
    this.channel = channel
    if (!this.child || !this.state.camera) return this.publicState()
    try {
      await this.applyChannel()
      return this.publicState()
    } catch (error) {
      this.fail(error)
      throw error
    }
  }

  async applyChannel() {
    const camera = this.state.camera
    const wanted = camera.channels?.includes(this.channel) ? this.channel : 'camera'
    if (camera.channel === wanted) return
    if (this.autoPending) await this.autoPending
    this.resetFocus()
    this.meterRevision += 1 // metering now reads different samples
    const result = await this.request('set', 'channel', wanted)
    this.resetFocus()
    this.nextMeterAt = 0
    this.publish({ camera: result.camera, controls: result.controls, error: null })
  }

  async setControl(name, value) {
    try {
      if (!['exposureAuto', 'exposure', 'gainAuto', 'gain'].includes(name)) throw new Error('Unknown camera control')
      this.resetFocus()
      if (name === 'exposureAuto') {
        this.autoExposure = false
        if (this.autoPending) await this.autoPending
        if (this.state.controls?.exposure.autoAvailable) await this.request('set', 'exposureAuto', false)
        if (value && this.state.controls?.gain.auto) {
          const gain = await this.request('set', 'gainAuto', false)
          this.state.controls = gain.controls
        }
        this.autoExposure = Boolean(value)
        this.resetFocus()
        this.nextMeterAt = 0
        this.publish({ error: null })
        return this.publicState()
      }
      if (name === 'exposure' || (name === 'gainAuto' && value)) {
        this.autoExposure = false
        if (this.autoPending) await this.autoPending
      }
      const result = await this.request('set', name, value)
      this.resetFocus()
      this.publish({ controls: result.controls, error: null })
      return this.publicState()
    } catch (error) {
      this.fail(error)
      throw error
    }
  }

  async close() {
    if (!this.child) return
    this.resetFocus()
    this.state.phase = 'closed'
    this.autoExposure = false
    if (this.autoPending) await this.autoPending
    try { await Promise.race([this.request('quit'), new Promise((resolve) => setTimeout(resolve, 300))]) } catch {}
    if (this.child) this.child.kill()
    this.child = null
  }
}

module.exports = { CameraService }
