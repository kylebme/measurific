/// <reference types="vite/client" />
// Standalone visual regression surface. Served by Vite, excluded from the app entry.
import { Component, useEffect, useRef, useState, type ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import { MeasurementEvidence } from './MeasurementEvidence'
import { MeasurementAnnotations } from './MeasurementAnnotations'
import type { FeatureRecord, MeasurementRecord, Point } from './types'
import './styles.css'
import thread25Fixture from './fixtures/thread25.json'
import threadFixture from './fixtures/threadMeasurements.json'
import angleCrashFixture from './fixtures/drilledPlateAngle.json'
import slottedBracketFixture from './fixtures/slottedBracket.json'
import gearTeethFixture from './fixtures/gearTeeth.json'
import rackTeethFixture from './fixtures/rackTeeth.json'

const line = (id: string, a: Point, b: Point): FeatureRecord => ({ id, type: 'line', status: 'fitted', support: [[a, b]], geometry: { center: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], direction: [b[0] - a[0], b[1] - a[1]] } })
const circle = (id: string, center: Point, radius: number): FeatureRecord => ({ id, type: 'circle', status: 'fitted', geometry: { center, radius }, support: [Array.from({ length: 97 }, (_, i) => [center[0] + radius * Math.cos(i * Math.PI / 48), center[1] + radius * Math.sin(i * Math.PI / 48)] as Point)] })
const distance = (id: string, a: Point, b: Point): MeasurementRecord => ({ id, operator: 'distance_construction', feature_refs: ['a', 'b'], status: 'measured', value: Math.hypot(b[0] - a[0], b[1] - a[1]) / 5, units: 'mm', witness: [a, b] })
const angle: MeasurementRecord = { id: 'angle', operator: 'angle', feature_refs: ['a', 'b'], status: 'measured', value: 45, units: 'deg' }
const polygon = (points: Point[]): FeatureRecord => ({ id: 'outline', type: 'polyline', status: 'fitted', support: [points], closed: [true] })
const plate = polygon([[180, 220], [470, 220], [470, 140], [620, 140], [620, 500], [360, 500], [360, 450], [180, 450], [180, 220]])
const datumFeatures = [
  polygon([[170, 140], [660, 140], [660, 510], [170, 510], [170, 140]]),
  line('datum-line', [210, 190], [610, 190]), line('parallel-target', [230, 440], [600, 441]),
  line('perpendicular-target', [600, 210], [600, 440]),
  circle('datum-center', [320, 315], 58), circle('concentric-target', [323, 313], 24),
]
type Fixture = { features: FeatureRecord[]; measurements: MeasurementRecord[]; size?: [number, number] }
// Worker results on demo images, at their image size.
const imageFixture = (data: unknown, size: [number, number]): Fixture => ({ ...(data as Fixture), size })
const fixtures: Record<string, Fixture> = {
  'Thread image 25': thread25Fixture as unknown as { features: FeatureRecord[]; measurements: MeasurementRecord[] },
  'Thread measurements': threadFixture as unknown as { features: FeatureRecord[]; measurements: MeasurementRecord[] },
  'Drilled plate angle regression': angleCrashFixture as unknown as { features: FeatureRecord[]; measurements: MeasurementRecord[] },
  'Mechanical plate': { features: [plate, circle('hole1', [240, 280], 18), circle('hole2', [240, 390], 18), circle('hole3', [550, 200], 23), circle('bore', [420, 350], 78)],
    measurements: [distance('width', [180, 450], [620, 450]), distance('step', [360, 500], [620, 500]), distance('height', [620, 140], [620, 500]), distance('top', [470, 140], [620, 140]), distance('holes', [240, 280], [240, 390]), distance('diagonal', [240, 280], [550, 200])] },
  'Datum controls': { features: datumFeatures, measurements: [
    { id: 'datum-a', operator: 'datum', feature_refs: ['datum-line'], status: 'measured', value: 0, units: 'deg', datum_label: 'A' },
    { id: 'parallel-a', operator: 'parallelism', feature_refs: ['datum-line', 'parallel-target'], status: 'measured', value: .155, units: 'deg', datum_label: 'A' },
    { id: 'perpendicular-a', operator: 'perpendicularity', feature_refs: ['datum-line', 'perpendicular-target'], status: 'measured', value: .018, units: 'deg', datum_label: 'A' },
    { id: 'datum-b', operator: 'datum', feature_refs: ['datum-center'], status: 'measured', value: 0, datum_label: 'B' },
    { id: 'concentric-b', operator: 'concentricity', feature_refs: ['datum-center', 'concentric-target'], status: 'measured', value: 3.606, units: 'px', datum_label: 'B' },
  ] },
  'Mixed annotations': { features: [polygon([[220, 170], [610, 170], [610, 470], [220, 470], [220, 170]]), line('a', [220, 170], [570, 170]), line('b', [220, 170], [430, 380])],
    measurements: [distance('short', [260, 170], [390, 170]), distance('long', [220, 170], [580, 170]), angle, { ...angle, id: 'parallel', operator: 'parallelism', value: .002 }] },
  'Diagonal / short / zero': { features: [polygon([[180, 240], [460, 520], [500, 480], [220, 200], [180, 240]])],
    measurements: [distance('diagonal', [220, 280], [420, 480]), distance('short', [180, 240], [190, 230]), distance('zero', [460, 520], [460, 520])] },
  'Distant intersection': { features: [line('a', [220, 200], [550, 200]), line('b', [220, 350], [550, 351])], measurements: [{ ...angle, value: Math.atan(1 / 330) * 180 / Math.PI }] },
  'Slot and size (slotted bracket)': imageFixture(slottedBracketFixture, [1600, 900]),
  'Tooth profile (gear)': imageFixture(gearTeethFixture, [1920, 1080]),
  'Tooth profile (rack)': imageFixture(rackTeethFixture, [1920, 1080]),
  'Crowded': { features: [plate], measurements: Array.from({ length: 14 }, (_, i) => distance(`distance-${i.toString().padStart(2, '0')}`, [190 + i * 9, 220], [420 + i * 10, 220])) },
}
// Catch a rendering regression on the fixture page without losing its failure report.
class RegressionBoundary extends Component<{ children: ReactNode }, { error: string | null }> {
  state = { error: null as string | null }
  static getDerivedStateFromError(error: Error) { return { error: error.message } }
  render() { return this.state.error ? <p role="alert">Annotation regression failed: {this.state.error}</p> : this.props.children }
}

function Preview() {
  const ref = useRef<HTMLDivElement>(null)
  const [stageSize, setStageSize] = useState({ width: 0, height: 0 })
  useEffect(() => {
    const element = ref.current!
    const observer = new ResizeObserver(() => { const b = element.getBoundingClientRect(); setStageSize({ width: b.width, height: b.height }) })
    observer.observe(element)
    return () => observer.disconnect()
  }, [])
  const requestedFixture = new URLSearchParams(window.location.search).get('fixture')
  const [name, setName] = useState(requestedFixture && fixtures[requestedFixture] ? requestedFixture : 'Mechanical plate'), [hidden, setHidden] = useState<string[]>([])
  const [view, setView] = useState({ scale: 1, x: 0, y: 0 }), [card, setCard] = useState(false), [narrow, setNarrow] = useState(false)
  const fixture = fixtures[name]
  const isAngleRegression = name === 'Drilled plate angle regression'
  const [regressionStatus, setRegressionStatus] = useState('')
  useEffect(() => {
    if (!isAngleRegression) { setRegressionStatus(''); return }
    if (hidden.includes('crash-angle') || view.scale !== 1 || view.x !== 0 || view.y !== 0) {
      setRegressionStatus('Reset view and visibility to run the angle regression')
      return
    }
    setRegressionStatus('Checking angle regression…')
    let frame = 0, attempts = 0, previous = ''
    const check = () => {
      const stage = ref.current!, label = stage.querySelector<HTMLElement>('[data-measurement-id="crash-angle"]')
      if (label && getComputedStyle(label).visibility === 'visible') {
        const b = label.getBoundingClientRect(), s = stage.getBoundingClientRect()
        const signature = JSON.stringify([b.x, b.y, b.width, b.height])
        const text = label.querySelector('strong')!
        if (signature === previous && b.left >= s.left && b.right <= s.right && b.top >= s.top && b.bottom <= s.bottom && text.scrollWidth <= text.clientWidth) {
          setRegressionStatus('Angle regression passed: stable, fully visible label')
          return
        }
        previous = signature
      }
      if (++attempts >= 10) { setRegressionStatus('Angle regression FAILED: label did not settle or text is clipped'); return }
      frame = requestAnimationFrame(check)
    }
    frame = requestAnimationFrame(check)
    return () => cancelAnimationFrame(frame)
  }, [isAngleRegression, stageSize, view, hidden])
  const width = fixture.size?.[0] ?? (name === 'Drilled plate angle regression' ? 1600 : 900)
  const height = fixture.size?.[1] ?? (name === 'Drilled plate angle regression' ? 900 : 650)
  return <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, padding: 12, alignItems: 'center' }}>
      <label>Fixture <select aria-label="Fixture" value={name} onChange={e => { setName(e.target.value); setHidden([]); setView({ scale: 1, x: 0, y: 0 }) }}>{Object.keys(fixtures).map(key => <option key={key}>{key}</option>)}</select></label>
      <button style={{ width: 'auto', margin: 0 }} className="primary-button" onClick={() => setView({ scale: 1.8, x: -240, y: -150 })}>Zoom and pan</button>
      <button style={{ width: 'auto', margin: 0 }} className="primary-button" onClick={() => { setView({ scale: 1, x: 0, y: 0 }); setHidden([]) }}>Reset view and visibility</button>
      <button style={{ width: 'auto', margin: 0 }} className="primary-button" onClick={() => setCard(!card)}>Toggle pinned card</button>
      <button style={{ width: 'auto', margin: 0 }} className="primary-button" onClick={() => setNarrow(!narrow)}>Toggle narrow viewport</button>
      <span>{hidden.length} hidden</span>
      {isAngleRegression && <span role="status">{regressionStatus}</span>}
    </div>
    <div ref={ref} className="image-stage" style={{ flex: isAngleRegression ? '0 0 951px' : 1, width: isAngleRegression ? 1366 : narrow ? 560 : '100%', background: '#f6f6f4', alignSelf: 'center' }}>
      <div className="stage-content" style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})` }}>
        <svg viewBox={`0 0 ${width} ${height}`} width="100%" height="100%">
          {fixture.features.flatMap(f => f.support?.map((run, i) => <polyline key={`${f.id}-${i}`} points={run.map(p => p.join(',')).join(' ')} fill={f.closed?.[i] ? '#e0e1df' : 'none'} stroke="#283139" strokeWidth="1.8"/>))}
          <MeasurementEvidence measurements={fixture.measurements.filter(m => !hidden.includes(m.id))}/>
        </svg>
      </div>
      <MeasurementAnnotations stageRef={ref} imageWidth={width} imageHeight={height} features={fixture.features} measurements={fixture.measurements} hiddenIds={hidden} view={view} deformationFactor={1} onHide={id => setHidden(ids => [...ids, id])}/>
      {name === 'Drilled plate angle regression' && fixture.features.filter(f => ['base_00003', 'base_00005'].includes(f.id)).map(f => {
        const center = f.geometry!.center as Point
        const scale = Math.min(stageSize.width / width, stageSize.height / height)
        const x = ((stageSize.width - width * scale) / 2 + center[0] * scale) * view.scale + view.x
        const y = ((stageSize.height - height * scale) / 2 + center[1] * scale) * view.scale + view.y
        return <div key={f.id} className="floating-tooltip pinned-tooltip" style={{ left: Math.max(126, Math.min(stageSize.width - 126, x)), top: Math.max(20, Math.min(stageSize.height - 20, y)) }}>
          <div className="feature-tooltip"><div className="tooltip-title"><span>Line</span><span className="mono">{f.id}</span></div><div className="tooltip-meta">base · fitted · 100% score (uncalibrated)</div></div>
        </div>
      })}
      {card && <div className="floating-tooltip pinned-tooltip" style={{ left: '50%', top: 160, background: '#15202c', padding: 20 }}>Pinned feature card<br/>This rectangle reserves layout space.</div>}
    </div>
  </div>
}
const root = createRoot(document.getElementById('root')!)
root.render(<RegressionBoundary><Preview/></RegressionBoundary>)
if (import.meta.hot) import.meta.hot.dispose(() => root.unmount())
