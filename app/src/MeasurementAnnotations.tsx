import { EyeOff } from 'lucide-react'
import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type RefObject } from 'react'
import type { FeatureRecord, MeasurementRecord, Point } from './types'
import {
  expandCompoundAnnotations, metricTables, isMetricTable, tableMetrics, annotationNames, annotationSymbols, formattedValue, isAnnotatedMeasurement, layoutAnnotations,
  type Box, type Size,
} from './annotationLayout'

export { isAnnotatedMeasurement } from './annotationLayout'

interface Props {
  stageRef: RefObject<HTMLDivElement | null>
  imageWidth: number
  imageHeight: number
  features: FeatureRecord[]
  measurements: MeasurementRecord[]
  hiddenIds: string[]
  view: { scale: number; x: number; y: number }
  deformationFactor: number
  onHide: (id: string) => void
}

export function MeasurementAnnotations({
  stageRef, imageWidth, imageHeight, features, measurements, hiddenIds, view, deformationFactor, onHide,
}: Props) {
  const refs = useRef(new Map<string, HTMLDivElement>())
  const [layout, setLayout] = useState<ReturnType<typeof layoutAnnotations>>({ drawings: [], unplaced: [] })
  const [, invalidate] = useState(0)
  const previous = useRef<{ key: string; features: FeatureRecord[]; measurements: MeasurementRecord[] }>()
  const visible = expandCompoundAnnotations(measurements.filter(m => isAnnotatedMeasurement(m) && !hiddenIds.includes(m.id)))

  // The parent stage ref is attached after descendant layout effects on mount.
  // A passive observer ensures the first image gets a layout without a click.
  useEffect(() => {
    const stage = stageRef.current
    if (!stage) return
    const observer = new ResizeObserver(() => invalidate(v => v + 1))
    observer.observe(stage)
    for (const label of refs.current.values()) observer.observe(label)
    return () => observer.disconnect()
  }, [stageRef, measurements, hiddenIds])

  // Read unrotated DOM dimensions after commit, then solve synchronously. There
  // are no asynchronous per-label writers racing pan/zoom or each other.
  useLayoutEffect(() => {
    const stage = stageRef.current
    if (!stage || imageWidth <= 0 || imageHeight <= 0) return
    const rect = stage.getBoundingClientRect()
    if (!rect.width || !rect.height) return
    const sizes = new Map<string, Size>()
    for (const m of visible) {
      const element = refs.current.get(m.id)
      if (element) sizes.set(m.id, { width: element.offsetWidth, height: element.offsetHeight })
    }
    const obstacles: Box[] = Array.from(stage.querySelectorAll<HTMLElement>('.pinned-tooltip, .deformation-warning, .control-layer-note, .reset-view-button'))
      .map(element => {
        const b = element.getBoundingClientRect()
        return { x: b.left - rect.left, y: b.top - rect.top, width: b.width, height: b.height }
      })
    const key = JSON.stringify([rect.width, rect.height, imageWidth, imageHeight, view, deformationFactor, hiddenIds, [...sizes], obstacles])
    if (previous.current?.key === key && previous.current.features === features && previous.current.measurements === measurements) return
    previous.current = { key, features, measurements }
    const scale = Math.min(rect.width / imageWidth, rect.height / imageHeight)
    const ox = (rect.width - imageWidth * scale) / 2, oy = (rect.height - imageHeight * scale) / 2
    // Match the feature overlay's integer pixel centers to the raster image.
    const toStage = ([x, y]: Point): Point => [(ox + (x + .5) * scale) * view.scale + view.x, (oy + (y + .5) * scale) * view.scale + view.y]
    setLayout(layoutAnnotations({ measurements: visible, features, toStage, sizes, obstacles, deformationFactor,
      // Leave a small footer for explicit placement failures; it never obscures a dimension.
      stage: { x: 12, y: 12, width: rect.width - 24, height: rect.height - 48 } }))
  })

  const drawings = new Map(layout.drawings.map(d => [d.measurement.id, d]))
  return <>
    <svg className="measurement-leader-layer mechanical-dimension-layer" aria-hidden="true">
      {layout.drawings.filter(d => !hiddenIds.includes(d.measurement.id)).map(d => <g key={d.measurement.id} data-drawing={d.kind}>
        {d.segments.map(({ start, end, role }, i) => {
          // Arc samples participate in occupancy, but SVG renders the true arc.
          if (d.arc && i < 16) return null
          const path = `M ${start[0]} ${start[1]} L ${end[0]} ${end[1]}`
          return <g key={i}><path className="leader-halo" d={path}/><path className={`leader-stroke ${role === 'leader' ? 'annotation-connector' : ''}`} d={path}/></g>
        })}
        {d.arc && <g><path className="leader-halo" d={d.arc.path}/><path className="leader-stroke" d={d.arc.path}/></g>}
        {d.arrows.map((path, i) => <g key={`arrow-${i}`}><path className="dimension-arrow-halo" d={path}/><path className="dimension-arrow" d={path}/></g>)}
        {d.witnesses.map(([x, y], i) => d.measurement.operator === 'datum'
          ? <path key={i} className="datum-triangle" d={`M ${x} ${y} l -5 -9 l 10 0 Z`}/>
          : <circle key={i} cx={x} cy={y} r="3"/>)}
      </g>)}
    </svg>
    {visible.map(m => {
      const drawing = drawings.get(m.id), name = m.annotation_label || annotationNames[m.operator], symbol = annotationSymbols[m.operator]
      const isDatumControl = ['concentricity', 'parallelism', 'perpendicularity'].includes(m.operator)
      return <div key={m.id} ref={element => { if (element) refs.current.set(m.id, element); else refs.current.delete(m.id) }}
        className={`mechanical-value-label ${m.operator === 'distance_construction' && m.selected_statistic === 'average' ? 'distance-mean-label' : ''} ${m.operator === 'datum' ? 'datum-value-label' : symbol ? 'gdt-value-label' : 'linear-value-label'}`}
        data-measurement-annotation={m.operator} data-measurement-id={m.id} data-placement={drawing?.kind}
        aria-label={isMetricTable(m) ? name : m.annotation_label ? formattedValue(m) : `${name}: ${m.operator === 'datum' ? (m.datum_label || '?') : formattedValue(m)}`}
        title={drawing?.kind === 'callout' && m.operator === 'angle' ? 'Angle between the indicated fitted lines' : undefined}
        style={{ left: drawing?.label[0] ?? 0, top: drawing?.label[1] ?? 0, visibility: drawing ? 'visible' : 'hidden',
          '--dimension-label-angle': `${drawing?.rotation ?? 0}deg` } as CSSProperties}>
        {symbol && <span className="measurement-symbol" aria-hidden="true">{symbol}</span>}
        {isMetricTable(m) ? <MetricTable measurement={m}/> : <strong>{m.operator === 'datum' ? (m.datum_label || '?') : <>{m.operator === 'angle' || m.operator === 'angular_span' ? '∠ ' : ''}{formattedValue(m)}</>}</strong>}
        {m.operator === 'distance_construction' && m.selected_statistic === 'average' && <span className="distance-annotation-note">Average over highlighted area · {((m.coverage || 0)*100).toFixed(1)}% measured{m.definition?.extend ? ' · extended geometry' : ''}</span>}
        {isDatumControl && <span className="datum-reference" title={`Datum ${m.datum_label || '?'}`}>{m.datum_label || '?'}</span>}
        <button title="Hide annotation" aria-label={`Hide ${name.toLowerCase()} annotation`} onClick={() => onHide(m.annotation_parent_id || m.id)}><EyeOff size={13}/></button>
      </div>
    })}
    {layout.unplaced.length > 0 && <div className="annotation-layout-notice" role="status"
      title={layout.unplaced.map(m => `${annotationNames[m.operator]}: ${formattedValue(m)}`).join('\n')}>
      {layout.unplaced.length} annotation{layout.unplaced.length === 1 ? '' : 's'} cannot fit in this view. Pan or reset the view; values remain in Results.
    </div>}
  </>
}

/** Min/max/average table for thread, slot and tooth profile measurements. */
export function MetricTable({ measurement }: { measurement: MeasurementRecord }) {
  const table = metricTables[measurement.operator], metrics = tableMetrics(measurement)
  const value = (v: number | null, units?: string) => v == null ? '—' : units === 'count' ? v.toFixed(Number.isInteger(v) ? 0 : 1) : v.toFixed(3)
  return <div className="metric-table"><strong>{table.title}</strong>
    <table><thead><tr><th>Measurement</th><th>Min</th><th>Max</th><th>Avg</th></tr></thead>
      <tbody>{Object.entries(table.names).map(([key, label]) => {
        const metric = metrics[key]
        return <tr key={key} title={metric?.reason || table.hints[key]}><th>{label}{metric ? ` (${metric.units})` : ''}</th>
          <td>{value(metric?.minimum ?? null, metric?.units)}</td><td>{value(metric?.maximum ?? null, metric?.units)}</td><td>{value(metric?.average ?? null, metric?.units)}</td></tr>
      })}</tbody></table>
    <small>{table.note.map((line, i) => <span key={i}>{i > 0 && <br/>}{line}</span>)}</small>
  </div>
}
