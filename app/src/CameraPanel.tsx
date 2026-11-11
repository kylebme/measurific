import { useEffect, useState } from 'react'
import { Camera, RefreshCw } from 'lucide-react'
import type { CameraChannel, CameraFocus, CameraNumericControl, CameraRoi, CameraState, PersistentStateView } from './types'

const channelLabels: Record<CameraChannel, string> = {
  red: 'Red channel only', green: 'Green channel only', blue: 'Blue channel only', camera: 'Camera grayscale',
}

function delivered(camera: NonNullable<CameraState['camera']>) {
  return camera.channel === 'camera'
    ? `camera grayscale · ${camera.sensorBits}-bit`
    : `${camera.channel} sites · half resolution · ${camera.sensorBits}-bit`
}

/** The persisted choice; applied to the open camera by the main process. */
function ChannelControl({ camera }: { camera: NonNullable<CameraState['camera']> }) {
  const [view, setView] = useState<PersistentStateView | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => { void window.inspection.getState().then(setView) }, [])
  useEffect(() => window.inspection.onState(setView), [])
  if (!view) return null
  const locked = view.settings.overridden.includes('cameraChannel')
  const choose = async (cameraChannel: CameraChannel) => {
    setError(null)
    try { setView(await window.inspection.updateSettings({ cameraChannel })) }
    catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)) }
  }
  const colorCamera = camera.channels.length > 1
  return <section className="camera-control">
    <label className="camera-channel"><span>Image</span>
      <select value={view.settings.effective.cameraChannel} disabled={locked || !colorCamera}
        onChange={(event) => { void choose(event.target.value as CameraChannel) }}>
        {(['red', 'green', 'blue', 'camera'] as const).map((channel) => <option key={channel} value={channel}>
          {channelLabels[channel]}{channel === 'red' ? ' (default)' : ''}</option>)}
      </select>
    </label>
    <p className="hint">{!colorCamera ? 'This camera has no raw color output; its own grayscale image is used.'
      : locked ? 'Set by INSPECTION_CAMERA_CHANNEL.'
      : 'Single channels use one color site of the sensor, on a regular grid at half resolution, without demosaicing or color processing. Camera grayscale is demosaiced and color-processed in the camera.'}</p>
    {error && <div className="inline-error">{error}</div>}
  </section>
}

function formatNumber(value: number) {
  if (!Number.isFinite(value)) return '—'
  if (Math.abs(value) >= 1000) return value.toFixed(0)
  if (Math.abs(value) >= 10) return value.toFixed(2)
  return value.toFixed(3)
}

function NumericControl({ label, units, control, autoName, valueName, onSet }: {
  label: string
  units: string
  control: CameraNumericControl
  autoName: 'exposureAuto' | 'gainAuto'
  valueName: 'exposure' | 'gain'
  onSet: (name: 'exposureAuto' | 'exposure' | 'gainAuto' | 'gain', value: boolean | number) => void
}) {
  const span = Math.max(0, control.max - control.min)
  const step = span ? Math.max(span / 1000, Number.EPSILON) : 1
  return <section className="camera-control">
    <div className="camera-control-heading"><span>{label}</span><b>{formatNumber(control.value)} {units}</b></div>
    {control.autoAvailable && <label className="camera-auto"><span>Automatic</span><input type="checkbox" checked={control.auto}
      onChange={(event) => onSet(autoName, event.target.checked)}/></label>}
    <input type="range" min={control.min} max={control.max} step={step} value={control.value}
      disabled={!control.available || control.auto}
      onChange={(event) => onSet(valueName, Number(event.target.value))}/>
    <div className="range-labels"><span>{formatNumber(control.min)}</span><span>{formatNumber(control.max)} {units}</span></div>
  </section>
}

function SharpnessIndicator({ state, roi }: { state: CameraState; roi: CameraRoi | null }) {
  const [focus, setFocus] = useState<CameraFocus | null>(null)
  const [now, setNow] = useState(Date.now())
  useEffect(() => window.inspection.onCameraFocus(setFocus), [])
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 200)
    return () => window.clearInterval(timer)
  }, [])
  const stale = Boolean(focus && now-focus.updatedAt > 1000)
  const valid = state.phase === 'streaming' && focus?.status === 'ok' && !stale
  const percent = valid && focus?.score != null && focus.peak ? Math.min(100, 100*focus.score/focus.peak) : 0
  const message = state.phase !== 'streaming' ? 'Waiting for live frames'
    : stale ? 'Waiting for a new frame'
    : focus?.status === 'too-small' ? 'Choose a larger region'
    : focus?.status === 'low-detail' ? 'Include a clear edge in the region'
    : focus?.status === 'unavailable' ? 'Sharpness unavailable'
    : !focus ? 'Measuring sharpness…' : 'Higher is sharper'
  return <section className="panel-section focus-panel" aria-label="Live sharpness">
    <div className="focus-heading"><div className="eyebrow">Sharpness</div>
      <button className="focus-reset" title="Reset the remembered peak after moving or replacing the part"
        onClick={() => { setFocus(null); void window.inspection.resetCameraFocus() }}
        aria-label="Reset sharpness peak"><RefreshCw size={12}/>Reset peak</button></div>
    <div className="focus-reading"><strong>{valid && focus?.score != null ? focus.score.toFixed(2) : '—'}</strong>
      <span>{message}</span></div>
    <div className="focus-track" role="meter" aria-label="Sharpness relative to session peak" aria-valuemin={0}
      aria-valuemax={100} aria-valuenow={Math.round(percent)} aria-valuetext={valid ? `${Math.round(percent)} percent of session peak` : message}>
      <span style={{ width: `${percent}%` }}/></div>
    <div className="focus-reference"><span>{valid ? `${Math.round(percent)}% of peak` : 'No current score'}</span>
      <span>Peak {focus?.peak != null ? focus.peak.toFixed(2) : '—'}</span></div>
    <p>Turn focus slowly to maximize the score. {roi ? 'Measured inside your ROI.' : 'Select an ROI around the edge you want to focus.'} Keep the part still; reset the peak after moving it.</p>
  </section>
}

export function CameraPanel({ state, roi, onDiscover, onSet }: {
  state: CameraState
  roi: CameraRoi | null
  onDiscover: () => void
  onSet: (name: 'exposureAuto' | 'exposure' | 'gainAuto' | 'gain', value: boolean | number) => void
}) {
  const camera = state.camera
  return <aside className="inspector-panel camera-panel">
    <section className="panel-section camera-summary">
      <div className="eyebrow">Live camera</div>
      <div className="camera-title"><Camera size={19}/><div><h3>{camera ? `${camera.vendor} ${camera.model}`.trim() : 'Searching for camera'}</h3>
        <small>{camera?.serial || camera?.id || 'USB3 Vision / GigE Vision'}</small></div></div>
      {camera && <dl>
        <div><dt>Transport</dt><dd>{camera.protocol || 'GenICam'}</dd></div>
        <div><dt>Frame</dt><dd>{camera.width} × {camera.height}</dd></div>
        <div><dt>Pixel format</dt><dd>{camera.pixelFormat}</dd></div>
        <div><dt>Delivered</dt><dd>{delivered(camera)}</dd></div>
      </dl>}
      <button className="secondary-button" onClick={onDiscover}><RefreshCw size={13}/>Rescan cameras</button>
      {state.error && <div className="inline-error">{state.error}</div>}
    </section>
    <SharpnessIndicator state={state} roi={roi}/>
    <section className="panel-section camera-roi-summary">
      <div className="eyebrow">Region of interest</div>
      <strong>{roi ? `${roi.shape === 'rectangle' ? 'Rectangular' : 'Circular'} ROI active` : 'Full camera frame'}</strong>
      <p>Choose a rectangle or circle from the toolbar, then drag over the live image. Sharpness and automatic exposure use this ROI. Pause applies it at full sensor resolution.</p>
    </section>
    {state.controls && <section className="panel-section">
      <div className="eyebrow">Acquisition</div>
      {camera && <ChannelControl camera={camera}/>}
      {state.controls.exposure.available
        ? <NumericControl label="Exposure time" units="µs" control={state.controls.exposure} autoName="exposureAuto" valueName="exposure" onSet={onSet}/>
        : <p className="hint">This camera does not expose a standard GenICam exposure control.</p>}
      {state.controls.gain.available
        ? <NumericControl label="Gain" units="dB" control={state.controls.gain} autoName="gainAuto" valueName="gain" onSet={onSet}/>
        : <p className="hint">This camera does not expose a standard GenICam gain control.</p>}
    </section>}
    <section className="panel-section camera-note">
      <div className="eyebrow">Measurement lock</div>
      <p>Measurements are disabled while frames are moving. Pause to capture the latest full-resolution frame and analyze it.</p>
    </section>
  </aside>
}
