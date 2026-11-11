import type { FeatureRecord, MeasurementRecord, Point } from './types'

const path = (points: Point[]) => points.map(([x, y], i) => `${i ? 'L' : 'M'} ${x} ${y}`).join(' ')
/** Evidence stays in original image coordinates even when deformation is exaggerated. */
export function MeasurementEvidence({ measurements, previewId, features = [] }: { measurements: MeasurementRecord[]; previewId?: string; features?: FeatureRecord[] }) {
  return <g className="measurement-evidence" pointerEvents="none">
    {measurements.map(m => <g key={m.id} opacity={previewId === m.id ? .7 : 1}>
      {previewId === m.id && m.feature_refs.map((id, i) => {
        const f = features.find(f => f.id === id)
        const q = f?.support?.flat() || []
        const p = (f?.geometry?.center || q[Math.floor(q.length/2)]) as Point | undefined
        return p && <g key={`feature-${id}`} className="measurement-feature-badge" transform={`translate(${p[0]},${p[1]-18})`}>
          <rect x="-10" y="-11" width="20" height="22" rx="4"/><text textAnchor="middle" y="4">{i ? 'B' : 'A'}</text>
        </g>
      })}
      {previewId === m.id && m.section_direction && (() => {
        const d = m.section_direction!, n: Point = [d[1], -d[0]]
        const q = features.find(f => f.id === m.feature_refs[0])?.support?.flat() || []
        if (!q.length || !d.every(Number.isFinite)) return null
        const p = q.reduce((a, b) => a[0]*n[0]+a[1]*n[1] < b[0]*n[0]+b[1]*n[1] ? a : b)
        const arrow = (v: Point) => { const tip: Point = [p[0]+v[0]*32, p[1]+v[1]*32]; return `${path([p, tip])} ${path([[tip[0]-v[0]*6-v[1]*3, tip[1]-v[1]*6+v[0]*3], tip, [tip[0]-v[0]*6+v[1]*3, tip[1]-v[1]*6-v[0]*3]])}` }
        return <g className="measurement-position-guide"><path className="section-direction" d={arrow(d)}><title>Measurement direction</title></path><path className="position-direction" d={arrow(n)}><title>Position along A</title></path><text x={p[0]-6} y={p[1]-8}>0</text></g>
      })()}
      {m.envelope_runs?.map((q, i) => <path key={`envelope-${i}`} d={path(q)} className="virtual-envelope"><title>{i < 2 ? 'Crest' : 'Root'} envelope · constructed between resolved extrema</title></path>)}
      {m.axis_witness && <path d={path(m.axis_witness)} className="measurement-axis"><title>Thread axis · diameters measured perpendicular to this axis</title></path>}
      {m.field_bands?.map((band, i) => <g key={i}>
        {(m.selected_statistic === 'average' || previewId === m.id) && <path className="gap-band" d={`${path([...band.map(pair => pair[0]), ...band.map(pair => pair[1]).reverse()])} Z`}><title>Contributing supported sections · no data across gaps</title></path>}
        {m.definition?.basis === 'fitted' && [0, 1].map(side => <path key={side} d={path(band.map(pair => pair[side]))} className="virtual-envelope"><title>{m.definition?.extend ? 'Virtual extended geometry' : 'Bounded fitted geometry'}</title></path>)}
      </g>)}
      {m.operator === 'distance_construction' && <>
        {(['minimum', 'maximum'] as const).map(stat => {
          const witness = m[`${stat}_witness`]
          if (!witness || (stat === 'maximum' && Math.abs((m.maximum || 0)-(m.minimum || 0)) < 1e-9)) return null
          return <path key={stat} d={path(witness)} className="gap-section"><title>{stat} gap: {m[stat]?.toFixed(3)} px</title></path>
        })}
        {m.witness && (m.dimension_direction ? (() => {
          // Axis components: the measured leg runs along the axis; the offset leg is only a guide.
          const [a, b] = m.witness!, d = m.dimension_direction!, t = (b[0]-a[0])*d[0]+(b[1]-a[1])*d[1]
          const corner: Point = [a[0]+t*d[0], a[1]+t*d[1]]
          return <><path d={path([a, corner])} className="gap-section"/><path d={path([corner, b])} className="dimension-projection"/></>
        })() : <path d={path(m.witness)} className="gap-section"/>)}
        {m.virtual_segments?.map((q, i) => <path key={`virtual-${i}`} d={path(q)} className="virtual-envelope"><title>Best-fit line extended beyond the visible edge</title></path>)}
        {m.section_direction && m.minimum_witness && (() => {
          const p = m.minimum_witness![0], d = m.section_direction!, n: Point = [-d[1], d[0]]
          const tip: Point = [p[0]+d[0]*20, p[1]+d[1]*20]
          return <path className="section-direction" d={`${path([p, tip])} ${path([[tip[0]-d[0]*5+n[0]*3, tip[1]-d[1]*5+n[1]*3], tip, [tip[0]-d[0]*5-n[0]*3, tip[1]-d[1]*5-n[1]*3]])}`}><title>Positive section direction</title></path>
        })()}
      </>}
    </g>)}
  </g>
}
