import { useEffect, useState } from 'react'
import { ArrowLeftRight } from 'lucide-react'
import { useInspectionStore } from './store'
import { anchorNames, availableConstructions, constructionHints, constructionNames, describeDistance, featureNoun, isRound, shortFeatureId } from './distanceText'
import type { Analysis, DistanceAnchor, DistanceDefinition, FeatureRecord, MeasurementRecord, Point } from './types'

type Construction = DistanceDefinition['construction']
const pointConstructions: Construction[] = ['center_distance', 'center_to_line', 'tangent_clearance']
const lineExtensionError = 'Center projection falls outside supported line spans'

const friendlyError = (message: string) => ({
  'Multiple surface intersections: choose a near/far or negative/positive branch': 'More than one edge lies along the measurement direction. Choose which side of each feature to measure.',
  'Near/far is ambiguous at coincident projected centers; choose negative/positive': 'These feature centers line up along the measurement direction. Choose the side toward or opposite the arrow.',
  'No common supported station interval; choose a different direction or explicitly extend fitted geometry': 'The selected edges do not overlap in this measuring direction. Change the direction or enable extensions for best-fit geometry.',
  'No transverse support for this section direction': 'This direction runs along an edge. Choose a direction that crosses both edges.',
  'Section has no common supported surface intersections': 'Both edges are not visible at this position. Move the measurement to where they overlap.',
  'Section is outside the requested station range': 'This position is outside the selected measurement area.',
  'Section lies on a discontinuity; choose an interior station': 'The edge changes abruptly at this position. Move the measurement slightly.',
  [lineExtensionError]: 'The perpendicular misses the visible part of the line. Turn on “Measure to the line’s extension” to use the extended best-fit line.',
  'Tangent side is ambiguous when the center lies on the line': 'The circle’s center lies on the line, so it has no near or far side. Measure to the center instead.',
  'Centers coincide, so near/far edge points are undefined; choose a measurement direction': 'The centers coincide, so near and far sides have no direction. Choose horizontal, vertical or an angle.',
} as Record<string, string>)[message] || message.replace(/^The (near|far) edge point of (A|B|the circle) lies outside the visible arc$/,
  (_, side, who) => `The ${side} side of ${who === 'the circle' ? 'the arc' : `arc ${who}`} is not on the visible part of the arc. Choose the other side or the center.`)
// Electron IPC wraps worker errors as "Error invoking remote method '…': Error: <message>".
const backendMessage = (e: unknown) => (e instanceof Error ? e.message : String(e)).replace(/^Error invoking remote method '[^']*': (?:\w*Error: )?/, '')
const sideName = (branch: string) => ({ auto: 'automatic', near: 'facing other feature', far: 'opposite side', negative: 'opposite arrow', positive: 'toward arrow' }[branch] || branch)

// Position controls use distance from the beginning of A, while the backend
// retains absolute projected image coordinates for reproducible requests.
export function distanceDirection(features: FeatureRecord[], definition: DistanceDefinition): Point {
  const mode = definition.direction_mode || (features[0].type === 'line' ? 'normal_a' : 'normal_b')
  let d = definition.direction || [0, 1]
  if (mode === 'x') d = [1, 0]
  if (mode === 'y') d = [0, 1]
  if (mode.startsWith('normal')) {
    const u = features[mode === 'normal_a' ? 0 : 1]?.geometry?.direction as [number, number] | undefined
    d = u ? [-u[1], u[0]] : [0, 1]
    const ca = features[0].geometry?.center as [number, number] | undefined
    const cb = features[1].geometry?.center as [number, number] | undefined
    if (ca && cb && (cb[0]-ca[0])*d[0]+(cb[1]-ca[1])*d[1] < 0) d = [-d[0], -d[1]]
  }
  const size = Math.hypot(...d) || 1
  return [d[0]/size, d[1]/size]
}
export function distancePositionRange(features: FeatureRecord[], definition: DistanceDefinition): Point {
  const d = distanceDirection(features, definition)
  const positions = features[0].support?.flat().map(p => p[0]*d[1]-p[1]*d[0]) || [0, 100]
  return [Math.min(...positions), Math.max(...positions)]
}

/** A fresh definition for a construction; gap statistics carry over between choices. */
export function definitionFor(construction: Construction, features: FeatureRecord[], statistic: DistanceDefinition['statistic'] = 'minimum'): DistanceDefinition {
  if (pointConstructions.includes(construction)) return { construction, basis: 'fitted', extend: false, statistic, anchors: ['center', 'center'] }
  if (construction === 'closest_clearance') return { construction, basis: 'observed', extend: false, statistic }
  const next: DistanceDefinition = { construction, basis: 'observed', extend: false, statistic, branches: ['auto', 'auto'],
    direction_mode: features[0].type === 'line' ? 'normal_a' : features[1].type === 'line' ? 'normal_b' : 'y' }
  if (construction === 'section') { const [lo, hi] = distancePositionRange(features, next); next.station = (lo+hi)/2 }
  return next
}
export const defaultDistanceDefinition = (features: FeatureRecord[]) => definitionFor(availableConstructions(features)[0], features)

// Saved tangent clearances are edited as point-to-line with a near-side anchor; the value is identical.
function editableDefinition(d: DistanceDefinition, features: FeatureRecord[]): DistanceDefinition {
  if (d.construction === 'tangent_clearance') return { ...d, construction: 'center_to_line', anchors: features.map(f => f.type === 'line' ? 'center' : 'near') }
  return pointConstructions.includes(d.construction) && !d.anchors ? { ...d, anchors: ['center', 'center'] } : d
}

export function DistanceInspector() {
  const { analysis, draft, editingMeasurementId, relationships } = useInspectionStore()
  const editing = relationships.find(m => m.id === editingMeasurementId)
  const refs = editing?.feature_refs || draft
  if (!analysis) return <p className="hint">Open an image to measure distances.</p>
  if (refs.length !== 2) return <p className="hint">Click two features: edges, circles or corners. {refs.length === 1 ? `A: ${featureNoun(analysis.report.features.find(f => f.id === refs[0]))}. Now click B.` : 'The options that fit your selection appear next.'}</p>
  return <DistanceEditor key={`${analysis.path}:${editing?.id || refs.join('|')}`} features={refs.map(id => analysis.report.features.find(f => f.id === id)!)} editing={editing}/>
}

function Segmented<T extends string>({ label, value, options, onChange }: { label: string; value: T; options: [T, string][]; onChange: (v: T) => void }) {
  return <div className="segmented" role="radiogroup" aria-label={label}>
    {options.map(([v, text]) => <button key={v} type="button" role="radio" aria-checked={v === value} className={v === value ? 'active' : ''} onClick={() => onChange(v)}>{text}</button>)}
  </div>
}

function DistanceEditor({ features, editing }: { features: FeatureRecord[]; editing?: MeasurementRecord }) {
  const { analysis, relationships, setDistancePreview, saveDistance, clearSelections } = useInspectionStore()
  const [definition, setDefinition] = useState<DistanceDefinition>(() => editing?.definition ? editableDefinition(editing.definition, features) : defaultDistanceDefinition(features))
  const [swapped, setSwapped] = useState(false)
  const ordered = swapped ? [...features].reverse() : features
  const refs = ordered.map(f => f.id)
  const [response, setResponse] = useState<{ key: string; analysis: Analysis | null; result?: MeasurementRecord; error?: string } | null>(null)
  const key = JSON.stringify([refs, definition])
  const result = response?.key === key && response.analysis === analysis ? response.result : undefined
  const rawError = response?.key === key && response.analysis === analysis ? response.error : undefined
  const error = rawError && friendlyError(rawError)
  const construction = definition.construction
  const field = construction === 'normal_gap' || construction === 'section'
  const pointBased = pointConstructions.includes(construction)
  const positionRange = distancePositionRange(ordered, definition)
  const positionOrigin = positionRange[0]
  const datums = relationships.filter(m => m.operator === 'datum' && m.datum?.type === 'line')
  const datumLabel = datums.find(m => m.feature_refs[0] === definition.datum_feature_ref)?.datum_label
  const choices = availableConstructions(ordered)
  if (!choices.includes(construction)) choices.push(construction)
  const patch = (values: Partial<DistanceDefinition>) => setDefinition(old => ({ ...old, ...values }))
  const setAnchor = (i: number, anchor: DistanceAnchor) => patch({ anchors: [0, 1].map(j => j === i ? anchor : definition.anchors?.[j] || 'center') })

  useEffect(() => {
    let cancelled = false
    setDistancePreview({ id: editing?.id || 'distance_preview', operator: 'distance_construction', feature_refs: refs, definition, status: 'preview', value: null, ...(field ? { section_direction: distanceDirection(ordered, definition) } : {}) })
    const timer = window.setTimeout(async () => {
      try {
        const measured = await window.inspection.measureRelationship({ id: editing?.id || 'distance_preview', operator: 'distance_construction', featureRefs: refs, params: definition })
        if (cancelled || useInspectionStore.getState().analysis !== analysis) return
        setResponse({ key, analysis, result: measured }); setDistancePreview(measured)
      } catch (e) {
        if (!cancelled && useInspectionStore.getState().analysis === analysis) setResponse({ key, analysis, error: backendMessage(e) })
      }
    }, 180)
    return () => { cancelled = true; window.clearTimeout(timer); setDistancePreview(null) }
    // The serialized request is the dependency; arrays are rebuilt during render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, analysis, editing?.id, setDistancePreview])

  const directionSelect = (point: boolean) => <label className="field-row">{point ? 'Direction' : 'Measurement direction'}<select aria-label={point ? 'Point distance direction' : 'Section direction'}
    value={definition.datum_feature_ref ? `datum:${definition.datum_axis || 'normal'}:${definition.datum_feature_ref}` : definition.direction_mode || (point ? 'direct' : 'normal_a')}
    onChange={e => {
      const [kind, axis, ref] = e.target.value.split(':')
      if (kind === 'datum') {
        const datum = datums.find(m => m.feature_refs[0] === ref)!.datum!
        patch({ direction_mode: 'custom', direction: axis === 'along' ? datum.direction : datum.normal, datum_feature_ref: ref, datum_axis: axis as 'normal' | 'along' })
      } else patch({ direction_mode: kind === 'direct' ? undefined : kind as DistanceDefinition['direction_mode'], direction: kind === 'custom' ? [1, 0] : point ? undefined : [0, 1], datum_feature_ref: undefined, datum_axis: undefined })
    }}>
    {point && <option value="direct">Straight line (true distance)</option>}
    {!point && ordered[0].type === 'line' && <option value="normal_a">Perpendicular to A</option>}
    {!point && ordered[1].type === 'line' && <option value="normal_b">Perpendicular to B</option>}
    <option value="x">Horizontal{point ? ' (ΔX)' : ' →'}</option><option value="y">Vertical{point ? ' (ΔY)' : ' ↓'}</option><option value="custom">At an angle…</option>
    {datums.map(m => <optgroup key={m.id} label={`Datum ${m.datum_label}`}>
      <option value={`datum:normal:${m.feature_refs[0]}`}>Perpendicular to datum {m.datum_label}</option>
      <option value={`datum:along:${m.feature_refs[0]}`}>Parallel to datum {m.datum_label}</option>
    </optgroup>)}
  </select></label>
  const angleInput = definition.direction_mode === 'custom' && !definition.datum_feature_ref && <label className="field-row">Angle from horizontal (°)<input aria-label="Measurement angle in degrees" type="number" step="any" value={Math.round(Math.atan2(definition.direction?.[1] ?? 1, definition.direction?.[0] ?? 0)*180/Math.PI*1000)/1000} onChange={e => {
    const angle = e.target.valueAsNumber*Math.PI/180; patch({ direction: [Math.cos(angle), Math.sin(angle)] })
  }}/></label>

  return <form className="distance-editor relationship-options" onSubmit={event => {
    event.preventDefault()
    if (result) saveDistance({ ...result, id: editing?.id || `user_distance_${Date.now()}` })
  }}>
    <div className="distance-pair">
      {ordered.map((f, i) => <div className="selection-slot filled" key={f.id} style={{ order: i*2 }}>
        <span>{'AB'[i]}</span><div><b>{featureNoun(f)}</b><small>{shortFeatureId(f.id)}</small></div>
      </div>)}
      <button type="button" className="icon-button distance-swap" title="Swap A and B" aria-label="Swap A and B" onClick={() => {
        setSwapped(v => !v); patch({ datum_feature_ref: undefined, datum_axis: undefined, ...(definition.anchors ? { anchors: [...definition.anchors].reverse() } : {}) })
      }}><ArrowLeftRight size={13}/></button>
    </div>

    <div className="field-stack">
      <label className="field-row">Measure<select aria-label="Distance construction" value={construction} onChange={e => setDefinition(definitionFor(e.target.value as Construction, ordered, definition.statistic))}>
        {choices.map(c => <option key={c} value={c}>{constructionNames[c]}</option>)}
      </select></label>
      <p className="field-hint">{constructionHints[construction]}</p>
    </div>

    {pointBased && <div className="distance-options">
      {ordered.map((f, i) => isRound(f) && <div className="anchor-row" key={f.id}>
        <span>Point on {'AB'[i]}</span>
        <Segmented label={`Point on ${'AB'[i]}`} value={definition.anchors?.[i] || 'center'} options={(['center', 'near', 'far'] as const).map(a => [a, anchorNames[a]])} onChange={a => setAnchor(i, a)}/>
      </div>)}
      {ordered.some(isRound) && <p className="hint">Near side faces the other feature{construction === 'center_distance' ? ' along the measuring direction' : ''}; far side is opposite.</p>}
      {construction === 'center_distance' && <>{directionSelect(true)}{angleInput}</>}
      {construction === 'center_to_line' && (definition.extend || rawError === lineExtensionError) && <label className="distance-check">
        <input type="checkbox" checked={definition.extend} onChange={e => patch({ extend: e.target.checked })}/>Measure to the line’s extension
      </label>}
    </div>}

    {field && <div className="distance-options">
      <label className="field-row">Measure from<select aria-label="Distance geometry" value={definition.basis} onChange={e => patch({ basis: e.target.value as DistanceDefinition['basis'], extend: false })}>
        <option value="observed">Actual edges</option><option value="fitted">Best-fit lines and circles</option>
      </select></label>
      {directionSelect(false)}{angleInput}
      {ordered.map((f, i) => (f.type !== 'line' || rawError?.includes('branch') || definition.branches?.[i] !== 'auto') && <label className="field-row" key={f.id}>Which side of {'AB'[i]}?<select aria-label={`Surface branch ${'AB'[i]}`} value={definition.branches?.[i] || 'auto'} onChange={e => {
        const branches = [...(definition.branches || ['auto', 'auto'])]; branches[i] = e.target.value; patch({ branches })
      }}><option value="auto">Choose automatically</option><option value="near">Facing the other feature</option><option value="far">Opposite side</option><option value="negative">Side opposite the arrow</option><option value="positive">Side toward the arrow</option></select></label>)}
      {construction === 'section' ? <label className="field-stack">Position along A (px)<input aria-label="Position along A" type="number" step="any" value={Math.round(((definition.station ?? positionOrigin)-positionOrigin)*1000)/1000} onChange={e => patch({ station: e.target.valueAsNumber+positionOrigin })}/><input aria-label="Move measurement along A" type="range" min={0} max={positionRange[1]-positionOrigin} step={.1} value={(definition.station ?? positionOrigin)-positionOrigin} onChange={e => patch({ station: e.target.valueAsNumber+positionOrigin })}/><small>0 is the beginning of A along the position arrow.</small></label>
        : <div className="field">Label on image<Segmented label="Label on image" value={definition.statistic} options={[['minimum', 'Smallest'], ['maximum', 'Largest'], ['average', 'Average']]} onChange={statistic => patch({ statistic })}/></div>}
      <details className="distance-more-options"><summary>Measurement area and extensions</summary>
        <p className="hint">Only the area where both features are visible is measured. You can narrow this area, or extend best-fit geometry beyond the visible edges.</p>
        <label className="distance-check"><input type="checkbox" checked={Boolean(definition.station_range)} onChange={e => patch({ station_range: e.target.checked ? result?.requested_span || positionRange : undefined, extend: false })}/>Limit the measurement area</label>
        {definition.station_range && <div className="distance-input-row">{[0, 1].map(i => <label key={i}>{i ? 'End' : 'Start'} along A (px)<input aria-label={`Station ${i ? 'end' : 'start'}`} type="number" step="any" value={Math.round((definition.station_range![i]-positionOrigin)*1000)/1000} onChange={e => {
          const v = [...definition.station_range!] as [number, number]; v[i] = e.target.valueAsNumber+positionOrigin; patch({ station_range: v })
        }}/></label>)}</div>}
        {definition.basis === 'fitted' && <label className="distance-check"><input type="checkbox" checked={definition.extend} onChange={e => patch({ extend: e.target.checked, station_range: e.target.checked ? definition.station_range || result?.requested_span || positionRange : definition.station_range })}/>Extend best-fit lines / arcs into this area</label>}
      </details>
      <p className="hint">The arrow shows the measuring direction from A. A negative gap means B lies on the opposite side of that arrow.</p>
    </div>}

    <p className="distance-summary">{describeDistance(definition, ordered, datumLabel)}</p>
    <div className="distance-preview" aria-live="polite">
      {error ? <div className="inline-error">{error}</div> : !result ? <p className="hint">Calculating preview…</p> : <DistanceResult measurement={result} describe={false}/>}
    </div>
    <div className="distance-actions">
      <button className="primary-button" type="submit" disabled={!result}>{editing ? 'Save changes' : 'Add measurement'}</button>
      <button className="open-button" type="button" onClick={clearSelections}>Cancel</button>
    </div>
  </form>
}

const negativeMeaning = (m: MeasurementRecord) => ({
  center_to_line: 'the circle crosses the line', tangent_clearance: 'the circle crosses the line',
  center_distance: 'the sides overlap along the measuring direction',
} as Record<string, string>)[m.definition?.construction || ''] || 'B lies on the opposite side of the arrow'

export function DistanceResult({ measurement: m, describe = true }: { measurement: MeasurementRecord; describe?: boolean }) {
  const { analysis, relationships } = useInspectionStore()
  const d = m.definition
  const construction = d?.construction || 'normal_gap'
  const features = m.feature_refs.map(id => analysis?.report.features.find(f => f.id === id))
  const datumLabel = relationships.find(r => r.operator === 'datum' && r.feature_refs[0] === d?.datum_feature_ref)?.datum_label
  const field = construction === 'normal_gap' || construction === 'section'
  return <div className="distance-result">
    <b>{constructionNames[construction]}{m.selected_statistic ? ` · ${({ minimum: 'smallest', maximum: 'largest', average: 'average' }[m.selected_statistic])}` : ''}</b>
    <strong>{typeof m.value === 'number' ? m.value.toFixed(3) : '—'} {m.units}</strong>
    {m.minimum != null && <dl>{(['minimum', 'maximum', 'average'] as const).map(k => <div key={k}><dt>{k === 'average' ? 'Average' : k === 'minimum' ? 'Smallest' : 'Largest'}</dt><dd>{m[k]?.toFixed(3)} px</dd></div>)}</dl>}
    {describe && d && <small>{describeDistance(d, features, datumLabel)}</small>}
    {m.signed && typeof m.value === 'number' && m.value < 0 && !field && <small>Negative: {negativeMeaning(m)}.</small>}
    {field && <small>{d?.basis === 'fitted' ? 'Best-fit geometry' : 'Actual edges'}{d?.extend ? ' · extended beyond visible edges' : ' · visible edges only'}</small>}
    {m.coverage != null && <small title="Percentage of the selected area where both edges can be measured">Measured area: {(100*m.coverage).toFixed(1)}%{m.crosses ? ' · edges cross' : ''}</small>}
    {field && d?.branches?.some(b => b !== 'auto') && <small>A: {sideName(d.branches[0])}; B: {sideName(d.branches[1])}</small>}
  </div>
}
