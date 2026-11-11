import type { FeatureRecord, MeasurementRecord, Point } from './types'
import { distanceShortLabel } from './distanceText'

export interface Box { x: number; y: number; width: number; height: number }
export interface Segment { start: Point; end: Point; role?: 'dimension' | 'leader' | 'extension' }
export interface Size { width: number; height: number }
export interface AnnotationDrawing {
  measurement: MeasurementRecord
  kind: 'distance' | 'angle' | 'callout'
  label: Point
  rotation: number
  box: Box
  size: Size
  segments: Segment[]
  arrows: string[]
  witnesses: Point[]
  arc?: { center: Point; radius: number; start: number; sweep: number; path: string }
}
interface Candidate extends AnnotationDrawing { cost: number }
interface Job { measurement: MeasurementRecord; size: Size; anchor: Point; targets: Point[][]; ports: Point[][]; candidates: Candidate[] }
export interface LayoutInput {
  measurements: MeasurementRecord[]
  features: FeatureRecord[]
  toStage: (point: Point) => Point
  stage: Box
  sizes?: ReadonlyMap<string, Size>
  obstacles?: Box[]
  deformationFactor?: number
}

export const annotationSymbols: Record<string, string> = {
  concentricity: '◎', perpendicularity: '⊥', parallelism: '∥', circularity_ls: '○', straightness: '—',
}
export const annotationNames: Record<string, string> = {
  distance_construction: 'Distance', thread: 'Thread measurements', compound_dimension: 'Compound dimension', angle: 'Angle', datum: 'Datum feature', concentricity: 'Concentricity',
  perpendicularity: 'Perpendicularity', parallelism: 'Parallelism', circularity_ls: 'Circularity', straightness: 'Straightness',
  least_squares_diameter: 'Diameter', radius: 'Radius', length: 'Length', angular_span: 'Arc angle',
  slot: 'Slot measurements', tooth_profile: 'Tooth profile measurements',
}
export const isAnnotatedMeasurement = (m: MeasurementRecord) => m.status === 'measured' && m.operator in annotationNames
// Drawing convention: size symbols lead the value without a separator.
const valueSymbols: Record<string, string> = { least_squares_diameter: 'Ø', radius: 'R' }
export function formattedValue(m: MeasurementRecord) {
  if (typeof m.value !== 'number' || !Number.isFinite(m.value)) return '—'
  const prefix = m.annotation_label || (m.operator === 'distance_construction' ? distanceShortLabel(m) : '')
  if (m.units === 'count') return `${prefix ? prefix + ': ' : ''}${m.value.toFixed(Number.isInteger(m.value) ? 0 : 1)}`
  return `${prefix ? prefix + ': ' : ''}${valueSymbols[m.operator] ?? ''}${m.value.toFixed(m.operator !== 'distance_construction' && Math.abs(m.value) >= 100 ? 1 : 3)}${m.units ? ` ${m.units}` : ''}`
}

/** Compound measurements reported as one table of min/max/average metrics. */
export interface MetricTable {
  title: string
  /** Table row names, in display order. */
  names: Record<string, string>
  /** Short dimension labels on the image. */
  labels: Record<string, string>
  hints: Record<string, string>
  note: string[]
}
export const metricTables: Record<string, MetricTable> = {
  thread: {
    title: 'Thread measurements',
    names: { minor_diameter: 'Minor diameter', major_diameter: 'Major diameter', root_radius: 'Root radius',
      crest_radius: 'Crest radius', pitch: 'Pitch', number_of_threads: 'Number of threads' },
    labels: { minor_diameter: 'Minor Ø', major_diameter: 'Major Ø', root_radius: 'Root R', crest_radius: 'Crest R', number_of_threads: 'Threads' },
    hints: { pitch: 'Axial spacing of corresponding same-direction flanks at a fixed radial section',
      root_radius: 'Radii of arcs reported at thread roots or crests', crest_radius: 'Radii of arcs reported at thread roots or crests' },
    note: ['Dashed: estimated crest/root boundaries', 'Diameters perpendicular to axis · unresolved radii: —', 'Visible crests counted per side'],
  },
  slot: {
    title: 'Slot measurements',
    names: { width: 'Width', overall_length: 'Overall length', end_radius: 'End radius', end_center_spacing: 'End center spacing' },
    labels: { width: 'Width', overall_length: 'Slot length', end_radius: 'End R', end_center_spacing: 'Center spacing' },
    hints: { width: 'Normal gap between the observed flats, or the fitted stadium width when the flats are not two lines',
      end_radius: 'Radii of the end arcs, or the fitted stadium radius when the ends are not arcs',
      overall_length: 'Fitted stadium length, end to end', end_center_spacing: 'Distance between the fitted end centers' },
    note: ['Width between flats · length and spacing from stadium fit'],
  },
  tooth_profile: {
    title: 'Tooth profile measurements',
    names: { number_of_teeth: 'Number of teeth', pitch: 'Pitch', tooth_height: 'Tooth height', tip_diameter: 'Tip diameter',
      root_diameter: 'Root diameter', tip_radius: 'Tip radius', root_radius: 'Root radius' },
    labels: { number_of_teeth: 'Teeth', tooth_height: 'Height', tip_diameter: 'Tip Ø', root_diameter: 'Root Ø', tip_radius: 'Tip R', root_radius: 'Root R' },
    hints: { pitch: 'Spacing of corresponding same-direction flanks at mid height; degrees about the center for round profiles',
      tooth_height: 'Tip height above the mean of its adjacent roots', tip_diameter: 'Twice the tip distance from the fitted center',
      root_diameter: 'Twice the root distance from the fitted center', tip_radius: 'Radii of arcs reported at tooth tips or roots',
      root_radius: 'Radii of arcs reported at tooth tips or roots' },
    note: ['Visible teeth counted per connected run', 'Diameters only for round profiles · unresolved radii: —'],
  },
}
export const isMetricTable = (m: { operator: string }) => m.operator in metricTables
export const tableMetrics = (m: MeasurementRecord) => m.thread_metrics ?? m.compound_metrics ?? {}

// Derived dimensions share their parent's visibility and lifetime, but use the
// same witness, collision, arrow, and placement machinery as other dimensions.
export function expandCompoundAnnotations(measurements: MeasurementRecord[]): MeasurementRecord[] {
  return measurements.flatMap(m => {
    if (!isMetricTable(m) || m.status !== 'measured') return [m]
    const table = metricTables[m.operator]
    const dimensions = Object.entries(tableMetrics(m)).flatMap(([key, metric]) =>
      (['minimum', 'maximum'] as const).flatMap(stat => {
        const equal = metric.minimum != null && metric.maximum != null && Math.abs(metric.minimum - metric.maximum) < 1e-9
        if (equal && stat === 'maximum') return []
        // An angular spacing has no straight dimension line; it stays in the table.
        if (metric.units === 'deg') return []
        const value = metric[stat], witness = metric[`${stat}_witness`]
        if (metric.status !== 'measured' || value == null || !witness || witness.length !== 2) return []
        return [{ ...m, id: `${m.id}:${key}:${stat}`, operator: 'compound_dimension',
          value, units: metric.units, witness, selected_statistic: stat,
          minimum_witness: witness, maximum_witness: witness,
          annotation_label: `${table.labels[key] || table.names[key] || key} ${equal ? 'min/max' : stat === 'minimum' ? 'min' : 'max'}`,
          annotation_parent_id: m.id, annotation_metric: key, thread_metrics: undefined, compound_metrics: undefined }]
      }))
    return [m, ...dimensions]
  })
}

// Compound extrema are display-only dimensions, independent of distance operators.
const isDistance = (m: MeasurementRecord) => ['distance_construction', 'compound_dimension', 'least_squares_diameter', 'radius', 'length'].includes(m.operator) && Boolean(selectedWitness(m))

const isPoint = (p: unknown): p is Point => Array.isArray(p) && p.length === 2 && p.every(Number.isFinite)
const add = (a: Point, b: Point): Point => [a[0] + b[0], a[1] + b[1]]
const sub = (a: Point, b: Point): Point => [a[0] - b[0], a[1] - b[1]]
const mul = (p: Point, s: number): Point => [p[0] * s, p[1] * s]
const dot = (a: Point, b: Point) => a[0] * b[0] + a[1] * b[1]
const cross = (a: Point, b: Point) => a[0] * b[1] - a[1] * b[0]
const norm = (p: Point) => Math.hypot(...p)
const unit = (p: Point): Point => norm(p) > 1e-9 ? mul(p, 1 / norm(p)) : [1, 0]
const midpoint = (a: Point, b: Point) => mul(add(a, b), .5)
const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x))
const compare = (a: string, b: string) => a < b ? -1 : a > b ? 1 : 0
export const expanded = (b: Box, p: number): Box => ({ x: b.x - p, y: b.y - p, width: b.width + p * 2, height: b.height + p * 2 })
export const boxesIntersect = (a: Box, b: Box) => a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height
export const containsBox = (a: Box, b: Box) => b.x >= a.x && b.y >= a.y && b.x + b.width <= a.x + a.width && b.y + b.height <= a.y + a.height
const containsPoint = (b: Box, p: Point) => p[0] >= b.x && p[0] <= b.x + b.width && p[1] >= b.y && p[1] <= b.y + b.height
export function boundsFor(points: Point[]): Box {
  let x = Infinity, y = Infinity, right = -Infinity, bottom = -Infinity
  for (const p of points) { x = Math.min(x, p[0]); y = Math.min(y, p[1]); right = Math.max(right, p[0]); bottom = Math.max(bottom, p[1]) }
  return { x, y, width: right - x, height: bottom - y }
}
function labelBox(label: Point, size: Size, rotation = 0): Box {
  const a = rotation * Math.PI / 180
  const width = Math.abs(Math.cos(a)) * size.width + Math.abs(Math.sin(a)) * size.height
  const height = Math.abs(Math.sin(a)) * size.width + Math.abs(Math.cos(a)) * size.height
  return { x: label[0] - width / 2, y: label[1] - height / 2, width, height }
}
function segmentHitsLabel(s: Segment, c: AnnotationDrawing, padding: number) {
  const a = -c.rotation * Math.PI / 180
  const local = (p: Point): Point => {
    const d = sub(p, c.label)
    return [d[0] * Math.cos(a) - d[1] * Math.sin(a), d[0] * Math.sin(a) + d[1] * Math.cos(a)]
  }
  return segmentHitsBox({ start: local(s.start), end: local(s.end) }, expanded(labelBox([0, 0], c.size), padding))
}
export function segmentsCross(a: Segment, b: Segment) {
  const r = sub(a.end, a.start), s = sub(b.end, b.start), d = cross(r, s)
  if (Math.abs(d) < 1e-8) return false
  const t = cross(sub(b.start, a.start), s) / d, u = cross(sub(b.start, a.start), r) / d
  return t > 1e-6 && t < 1 - 1e-6 && u > 1e-6 && u < 1 - 1e-6
}
// Liang–Barsky clipping also detects collinear/tangent segments and endpoints.
export function segmentHitsBox(s: Segment, b: Box) {
  const d = sub(s.end, s.start)
  let lo = 0, hi = 1
  for (const [p, q] of [[-d[0], s.start[0] - b.x], [d[0], b.x + b.width - s.start[0]], [-d[1], s.start[1] - b.y], [d[1], b.y + b.height - s.start[1]]]) {
    if (Math.abs(p) < 1e-9) { if (q < 0) return false; continue }
    if (p < 0) lo = Math.max(lo, q / p)
    else hi = Math.min(hi, q / p)
    if (lo > hi) return false
  }
  return true
}
function closest(points: Point[], target: Point) {
  return points.reduce((a, b) => norm(sub(a, target)) <= norm(sub(b, target)) ? a : b)
}
function dock(box: Box, target: Point): Point {
  const center: Point = [box.x + box.width / 2, box.y + box.height / 2], d = sub(target, center)
  const k = Math.min(d[0] ? box.width / 2 / Math.abs(d[0]) : Infinity, d[1] ? box.height / 2 / Math.abs(d[1]) : Infinity)
  return Number.isFinite(k) ? add(center, mul(d, k)) : center
}
function arrow(tip: Point, towardsBase: Point) {
  const u = unit(towardsBase), n: Point = [-u[1], u[0]], base = add(tip, mul(u, 8))
  const a = add(base, mul(n, 3)), b = add(base, mul(n, -3))
  return `M ${tip[0]} ${tip[1]} L ${a[0]} ${a[1]} L ${b[0]} ${b[1]} Z`
}
export function selectedWitness(m: MeasurementRecord): Point[] | undefined {
  const w = m.witness ?? (m.selected_statistic === 'maximum' ? m.maximum_witness : m.selected_statistic === 'average' ? undefined : m.minimum_witness)
  return w?.length === 2 && w.every(isPoint) ? w : undefined
}

/** Exact L1 isotonic placement with variable label widths. Independently implemented
 * from the PAVA method discussed in Draftwright layout.py: shift away minimum gaps,
 * merge descending median pools, clamp the common feasible interval, unshift.
 * Stable lower medians make equal-cost layouts reproducible. null means no room.
 */
export function solveStrip(naturals: number[], extents: number[], lo: number, hi: number, gap = 8): number[] | null {
  if (!naturals.length) return []
  const offsets = [extents[0] / 2]
  for (let i = 1; i < naturals.length; i++) offsets.push(offsets[i - 1] + (extents[i - 1] + extents[i]) / 2 + gap)
  const upper = hi - offsets.at(-1)! - extents.at(-1)! / 2
  if (upper < lo) return null
  const blocks: { values: number[]; median: number }[] = []
  for (let i = 0; i < naturals.length; i++) {
    let block = { values: [naturals[i] - offsets[i]], median: naturals[i] - offsets[i] }
    while (blocks.length && blocks.at(-1)!.median > block.median) {
      const values = [...blocks.pop()!.values, ...block.values].sort((a, b) => a - b)
      block = { values, median: values[Math.floor((values.length - 1) / 2)] }
    }
    blocks.push(block)
  }
  return blocks.flatMap(b => b.values.map(() => clamp(b.median, lo, upper))).map((p, i) => p + offsets[i])
}

function featureRuns(f: FeatureRecord, toStage: LayoutInput['toStage'], factor: number): Point[][] {
  return (f.support || []).map((run, r) => run.map((p, i) => {
    const ideal = f.ideal_support?.[r]?.[i]
    return toStage(ideal && factor !== 1 ? add(ideal, mul(sub(p, ideal), factor)) : p)
  }).filter(isPoint)).filter(run => run.length)
}

// Only the contour inside the measured span constrains an aligned dimension.
// Clipping segments to that span avoids the far, empty corner of a global AABB.
function localEnvelope(p1: Point, p2: Point, axis: Point, normal: Point, contour: Segment[]) {
  const lo = dot(p1, axis), hi = dot(p2, axis)
  let min = Math.min(dot(p1, normal), dot(p2, normal)), max = Math.max(dot(p1, normal), dot(p2, normal))
  const include = (p: Point) => { const v = dot(p, normal); min = Math.min(min, v); max = Math.max(max, v) }
  for (const s of contour) {
    const a = dot(s.start, axis), b = dot(s.end, axis), delta = b - a
    if (Math.max(a, b) < lo || Math.min(a, b) > hi) continue
    if (Math.abs(delta) < 1e-9) { include(s.start); include(s.end) }
    else {
      const t1 = clamp((lo - a) / delta, 0, 1), t2 = clamp((hi - a) / delta, 0, 1)
      include(add(s.start, mul(sub(s.end, s.start), t1)))
      include(add(s.start, mul(sub(s.end, s.start), t2)))
    }
  }
  return [min, max]
}
function distanceCandidates(job: Job, witness: Point[], contour: Segment[]): Candidate[] {
  let [p1, p2] = witness
  // Axis-projected distances dimension along their measuring direction, with
  // extension lines from the true points; others align with the witness.
  const fixed = job.measurement.dimension_direction
  let axis = isPoint(fixed) && norm(fixed) > 1e-9 ? unit(fixed) : unit(sub(p2, p1))
  // Keep nearly vertical text reading bottom-to-top even when fitted x noise
  // changes sign. This only reverses direction/order; it never snaps geometry.
  if (Math.abs(axis[0]) < .001 ? axis[1] > 0 : axis[0] < 0) axis = [0 - axis[0], 0 - axis[1]] // 0 - x avoids -0
  if (dot(sub(p2, p1), axis) < 0) [p1, p2] = [p2, p1]
  const span = dot(sub(p2, p1), axis)
  if (!(span >= .5)) return [] // Coincident/subpixel points need a callout, not a made-up axis.
  const normal: Point = [-axis[1], axis[0]]
  const rotation = Math.atan2(axis[1], axis[0]) * 180 / Math.PI
  const envelope = localEnvelope(p1, p2, axis, normal, contour), candidates: Candidate[] = []
  for (const side of [-1, 1]) for (let lane = 0; lane < 4; lane++) {
    const projection = envelope[side < 0 ? 0 : 1] + side * (14 + job.size.height / 2 + lane * (job.size.height + 12))
    const q1 = add(p1, mul(normal, projection - dot(p1, normal))), q2 = add(p2, mul(normal, projection - dot(p2, normal)))
    const short = span < job.size.width + 24
    for (const end of short ? [-1, 1] : [0]) {
      const label = end ? add(end < 0 ? q1 : q2, mul(axis, end * (job.size.width / 2 + 18))) : midpoint(q1, q2)
      const extension = (p: Point, q: Point): Segment => {
        const u = unit(sub(q, p))
        return { start: add(p, mul(u, Math.min(5, norm(sub(q, p)) / 2))), end: add(q, mul(u, 5)), role: 'extension' }
      }
      const segments = [extension(p1, q1), extension(p2, q2), { start: short ? add(q1, mul(axis, -14)) : q1, end: short ? add(q2, mul(axis, 14)) : q2 }]
      if (end) segments.push({ start: end < 0 ? q1 : q2, end: label })
      candidates.push({ measurement: job.measurement, kind: 'distance', label, rotation, box: labelBox(label, job.size, rotation), size: job.size,
        segments, arrows: [arrow(q1, mul(axis, short ? -1 : 1)), arrow(q2, mul(axis, short ? 1 : -1))], witnesses: [p1, p2],
        cost: norm(sub(q1, p1)) + norm(sub(q2, p2)) + lane * 12 + (short ? 24 : 0) })
    }
  }
  return candidates
}

function angleCandidates(job: Job, features: Map<string, FeatureRecord>, toStage: LayoutInput['toStage'], stage: Box): Candidate[] {
  const lines = job.measurement.feature_refs.map(id => features.get(id)?.geometry)
  if (lines.length !== 2 || lines.some(g => !isPoint(g?.center) || !isPoint(g?.direction) || norm(g.direction) < 1e-9)) return []
  const a = lines[0]!, b = lines[1]!, u = unit(a.direction as Point)
  let v = unit(b.direction as Point)
  if (dot(u, v) < 0) v = mul(v, -1)
  const determinant = cross(u, v)
  if (Math.abs(determinant) < 1e-6) return []
  const vertex = add(a.center as Point, mul(u, cross(sub(b.center as Point, a.center as Point), v) / determinant))
  const center = toStage(vertex)
  // A distant virtual sharp must never be replaced by a plausible-looking fake arc.
  if (!isPoint(center) || !containsPoint(stage, center)) return []
  const sweep = Math.acos(clamp(dot(u, v), -1, 1))
  if (sweep < Math.PI / 90) return []
  const candidates: Candidate[] = []
  for (const sign of [1, -1]) {
    const startVector = mul(determinant > 0 ? u : v, sign), start = Math.atan2(startVector[1], startVector[0])
    for (const radius of [40, 60, 84, 112, 144]) {
      const radial = (a: number, r = radius): Point => add(center, [Math.cos(a) * r, Math.sin(a) * r])
      const middle = start + sweep / 2
      const labelClearance = Math.abs(Math.cos(middle)) * job.size.width / 2 + Math.abs(Math.sin(middle)) * job.size.height / 2 + 10
      const p1 = radial(start), p2 = radial(start + sweep), label = radial(middle, radius + labelClearance)
      const path = `M ${p1[0]} ${p1[1]} A ${radius} ${radius} 0 0 1 ${p2[0]} ${p2[1]}`
      const segments: Segment[] = []
      // Sample the actual arc for shared occupancy, not just its label.
      for (let i = 0; i < 16; i++) segments.push({ start: radial(start + sweep * i / 16), end: radial(start + sweep * (i + 1) / 16) })
      const targets = determinant > 0 ? job.targets : [...job.targets].reverse()
      for (const [i, endpoint] of [p1, p2].entries()) {
        const target = closest(targets[i], endpoint)
        // Project observed support onto the fitted line; this is an extension of
        // the measured line, not a ray at an arbitrary position on the part.
        const direction = unit(sub(endpoint, center)), projected = add(center, mul(direction, dot(sub(target, center), direction)))
        segments.push({ start: projected, end: add(endpoint, mul(direction, 6)) })
      }
      segments.push({ start: radial(middle), end: dock(labelBox(label, job.size), radial(middle)) })
      candidates.push({ measurement: job.measurement, kind: 'angle', label, rotation: 0, box: labelBox(label, job.size), size: job.size, segments,
        arrows: [arrow(p1, [-Math.sin(start), Math.cos(start)]), arrow(p2, [Math.sin(start + sweep), -Math.cos(start + sweep)])],
        witnesses: [], arc: { center, radius, start, sweep, path },
        cost: 40 + radius * .3 + segments.slice(16, 18).reduce((s, line) => s + norm(sub(line.end, line.start)), 0) })
    }
  }
  return candidates
}

function callout(job: Job, label: Point, cost: number): Candidate {
  const box = labelBox(label, job.size)
  const witnesses = job.ports.map(points => closest(points, label))
  const segments: Segment[] = witnesses.map(target => ({ start: dock(box, target), end: target, role: 'leader' }))
  // Keep a direct, exact witness segment in a distance fallback, including zero.
  if (isDistance(job.measurement)) segments.push({ start: witnesses[0], end: witnesses[1], role: 'dimension' })
  return { measurement: job.measurement, kind: 'callout', label, rotation: 0, box, size: job.size, segments, arrows: [], witnesses, cost: cost + norm(sub(label, job.anchor)) }
}

// Compound extrema are a coordinated set of horizontal callouts. Draw the
// measured witness once and attach one short leader, instead of extending both
// endpoints to independent dimension lanes around the entire part.
function compoundCallout(job: Job, label: Point): Candidate {
  const box = labelBox(label, job.size)
  const witnesses = job.targets.map(points => points[0])
  const target = midpoint(witnesses[0], witnesses[1])
  const start: Point = [target[0] < label[0] ? box.x : box.x + box.width, label[1]]
  const elbow: Point = [start[0] + Math.sign(target[0] - start[0]) * 12, start[1]]
  const count = job.measurement.units === 'count'
  const segments: Segment[] = [{ start, end: elbow, role: 'leader' }, { start: elbow, end: target, role: 'leader' }]
  if (!count) segments.push({ start: witnesses[0], end: witnesses[1], role: 'dimension' })
  const delta = sub(witnesses[1], witnesses[0])
  return { measurement: job.measurement, kind: 'callout', label, rotation: 0, box, size: job.size,
    segments, witnesses, arrows: !count && norm(delta) > 18 ? [arrow(witnesses[0], delta), arrow(witnesses[1], mul(delta, -1))] : [],
    cost: norm(sub(target, elbow)) + 20 }
}

function compoundCandidates(jobs: Job[], runs: Map<string, Point[][]>, stage: Box) {
  const parents = jobs.filter(j => isMetricTable(j.measurement))
  for (const parent of parents) {
    const children = jobs.filter(j => j.measurement.annotation_parent_id === parent.measurement.id)
    if (!children.length) continue
    const points = parent.measurement.feature_refs.flatMap(id => runs.get(id)?.flat() || [])
    if (!points.length) continue
    const part = boundsFor(points)
    // Solve the same ordered set in both gutters. The shared occupancy solver
    // can use either side, but labels retain horizontal alignment and order.
    for (const right of [false, true]) {
      const port = (j: Job) => midpoint(j.targets[0][0], j.targets[1][0])
      const ordered = [...children].sort((a, b) => port(a)[1] - port(b)[1] || compare(a.measurement.id, b.measurement.id))
      const positions = solveStrip(ordered.map(j => port(j)[1]), ordered.map(j => j.size.height), stage.y, stage.y + stage.height, 8)
      if (!positions) continue
      const width = Math.max(...ordered.map(j => j.size.width))
      const edge = right ? Math.min(part.x + part.width + 32, stage.x + stage.width - width)
        : Math.max(part.x - 32, stage.x + width)
      ordered.forEach((job, i) => {
        const x = right ? edge + job.size.width / 2 : edge - job.size.width / 2
        job.candidates.push(compoundCallout(job, [x, positions[i]]))
      })
    }
    // Give the large summary its own proposals; do not let it displace the
    // extrema while partitioning the generic boundary strips.
    boundaryCandidates([parent], part, stage)
  }
}

function boundaryCandidates(jobs: Job[], part: Box, stage: Box) {
  for (const side of ['top', 'bottom', 'left', 'right']) {
    const horizontal = side === 'top' || side === 'bottom', axis = horizontal ? 0 : 1
    const ordered = [...jobs].sort((a, b) => a.anchor[axis] - b.anchor[axis] || compare(a.measurement.id, b.measurement.id))
    const lo = horizontal ? stage.x : stage.y, hi = lo + (horizontal ? stage.width : stage.height)
    // Partition overfull strips into bounded rows, preserving feature order.
    let row: Job[] = [], used = 0
    const rows: Job[][] = []
    for (const job of ordered) {
      const extent = horizontal ? job.size.width : job.size.height
      if (row.length && used + extent + 8 > hi - lo) { rows.push(row); row = []; used = 0 }
      row.push(job); used += extent + (row.length > 1 ? 8 : 0)
    }
    if (row.length) rows.push(row)
    rows.forEach((members, tier) => {
      const positions = solveStrip(members.map(j => j.anchor[axis]), members.map(j => horizontal ? j.size.width : j.size.height), lo, hi)
      if (!positions) return
      members.forEach((job, i) => {
        const { width: w, height: h } = job.size
        for (let lane = 0; lane < 2; lane++) {
          const offset = 12 + (tier + lane) * ((horizontal ? h : w) + 10)
          const normal = side === 'top' ? part.y - h / 2 - offset : side === 'bottom' ? part.y + part.height + h / 2 + offset
            : side === 'left' ? part.x - w / 2 - offset : part.x + part.width + w / 2 + offset
          const label: Point = horizontal ? [positions[i], clamp(normal, stage.y + h / 2, stage.y + stage.height - h / 2)]
            : [clamp(normal, stage.x + w / 2, stage.x + stage.width - w / 2), positions[i]]
          job.candidates.push(callout(job, label, (job.measurement.operator in annotationSymbols ? 0 : 1000) + lane * 20))
        }
      })
    })
  }
}

function candidatesConflict(a: Candidate, b: Candidate) {
  if (boxesIntersect(expanded(a.box, 5), b.box)) return true
  // A fallback must not turn a dense view into a web of crossing leaders.
  // Extension-line crossings between actual dimensions remain a soft drafting
  // penalty; callout crossings are an explicit capacity failure.
  if ((a.kind === 'callout' || b.kind === 'callout') && a.segments.some(s => b.segments.some(t => segmentsCross(s, t)))) return true
  return a.segments.some(s => segmentHitsBox(s, expanded(b.box, 3))) || b.segments.some(s => segmentHitsBox(s, expanded(a.box, 3)))
}

/** One collect → solve → emit pass for distances, angles and GD&T.
 * Boundary proposals use feature-ordered PAVA; finite aligned/radial alternatives
 * share the same occupancy. A bounded deterministic beam can revisit earlier
 * choices. Bounds, label/line collisions and callout crossings are hard;
 * extension-line crossings and length are soft.
 * This is a bounded candidate solver, not a claim of global optimality.
 */
export function layoutAnnotations(input: LayoutInput): { drawings: AnnotationDrawing[]; unplaced: MeasurementRecord[] } {
  const { toStage, stage, sizes, obstacles = [] } = input
  const visible = input.measurements.filter(m => isAnnotatedMeasurement(m) && typeof m.value === 'number' && Number.isFinite(m.value))
    .sort((a, b) => compare(a.id, b.id))
  const featureMap = new Map(input.features.map(f => [f.id, f]))
  const runs = new Map(input.features.map(f => [f.id, featureRuns(f, toStage, input.deformationFactor ?? 1)]))
  const allRuns = [...runs.values()].flat(), points = allRuns.flat()
  if (!points.length || stage.width <= 0 || stage.height <= 0) return { drawings: [], unplaced: visible }
  const part = boundsFor(points)
  const contour: Segment[] = allRuns.flatMap(run => run.slice(1).map((end, i) => ({ start: run[i], end })))
  const jobs: Job[] = [], invalid: MeasurementRecord[] = []
  for (const measurement of visible) {
    const witness = selectedWitness(measurement)?.map(toStage)
    const featureRefs = measurement.operator === 'datum' ? measurement.feature_refs.slice(0, 1)
      : ['concentricity', 'parallelism', 'perpendicularity'].includes(measurement.operator) ? measurement.feature_refs.slice(1)
        : measurement.feature_refs
    const targets = isDistance(measurement) ? witness?.map(p => [p])
      : measurement.field_bands?.length && measurement.selected_statistic === 'average' ? [measurement.field_bands.flatMap(band => band.flat()) .map(toStage)]
        : featureRefs.map(id => runs.get(id)?.flat() || [])
    if (!targets?.length || targets.some(t => !t.length || !t.every(isPoint))) { invalid.push(measurement); continue }
    const anchor = targets.map(t => { const b = boundsFor(t); return [b.x + b.width / 2, b.y + b.height / 2] as Point }).reduce(add)
    // Line callouts attach inside each fitted feature. Using nearest endpoints
    // can collapse both leaders onto one shared corner and conceal the pairing.
    const ports = targets.map((points, i) => {
      const f = featureMap.get(featureRefs[i])
      return !measurement.field_bands && !isDistance(measurement) && f?.type === 'line' && isPoint(f.geometry?.center)
        ? [toStage(f.geometry.center)] : points
    })
    const job: Job = { measurement, targets, ports, anchor: mul(anchor, 1 / targets.length),
      size: sizes?.get(measurement.id) ?? { width: Math.max(104, formattedValue(measurement).length * 7 + 46 + (measurement.operator in annotationSymbols ? 29 : 0)), height: 28 }, candidates: [] }
    if (isDistance(measurement) && !measurement.annotation_parent_id) job.candidates = distanceCandidates(job, witness!, contour)
    if (measurement.operator === 'angle') job.candidates = angleCandidates(job, featureMap, toStage, stage)
    jobs.push(job)
  }
  compoundCandidates(jobs, runs, stage)
  boundaryCandidates(jobs.filter(j => !j.measurement.annotation_parent_id && !isMetricTable(j.measurement)), part, stage)
  for (const job of jobs) {
    job.candidates = job.candidates.filter(c => containsBox(stage, c.box) && !obstacles.some(b => boxesIntersect(expanded(c.box, 4), b) || c.segments.some(s => segmentHitsBox(s, b)))
      && (c.kind === 'callout' || c.segments.every(s => containsPoint(stage, s.start) && containsPoint(stage, s.end))))
    job.candidates = job.candidates.filter(c => !contour.some(s => segmentHitsLabel(s, c, 4)))
    for (const c of job.candidates) {
      // Outside the part is preferred. Local contour checks preserve useful bays
      // in concave parts where the global bounding rectangle would be misleading.
      const lineHits = c.kind === 'distance' ? contour.filter(s => segmentsCross(c.segments[2], s)).length : 0
      c.cost += lineHits * 250
      if (c.kind === 'callout' && boxesIntersect(c.box, expanded(part, 5))) c.cost += 2000
    }
    job.candidates.sort((a, b) => a.cost - b.cost)
  }
  // Short dimensions occupy inner lanes before enclosing spans. Stable geometry
  // keys make reversing selection or insertion order immaterial.
  jobs.sort((a, b) => {
    const span = (j: Job) => isDistance(j.measurement) ? norm(sub(j.targets[1][0], j.targets[0][0])) : Infinity
    return Number(isMetricTable(b.measurement)) - Number(isMetricTable(a.measurement)) || span(a) - span(b) || a.anchor[0] - b.anchor[0] || a.anchor[1] - b.anchor[1] || compare(a.measurement.id, b.measurement.id)
  })
  type State = { chosen: Candidate[]; unplaced: MeasurementRecord[]; cost: number }
  const pairCache = new WeakMap<Candidate, WeakMap<Candidate, { conflict: boolean; crossings: number }>>()
  const pair = (a: Candidate, b: Candidate) => {
    let cache = pairCache.get(a)
    if (!cache) { cache = new WeakMap(); pairCache.set(a, cache) }
    let result = cache.get(b)
    if (!result) {
      const conflict = candidatesConflict(a, b)
      const crossings = conflict ? 0 : a.segments.reduce((n, s) => n + b.segments.filter(t => segmentsCross(s, t)).length, 0)
      result = { conflict, crossings }; cache.set(b, result)
    }
    return result
  }
  let states: State[] = [{ chosen: [], unplaced: invalid, cost: 0 }]
  for (const job of jobs) {
    const next: State[] = []
    for (const state of states) {
      for (const c of job.candidates) {
        if (state.chosen.some(other => pair(c, other).conflict)) continue
        const crossings = state.chosen.reduce((n, other) => n + pair(c, other).crossings, 0)
        next.push({ chosen: [...state.chosen, c], unplaced: state.unplaced, cost: state.cost + c.cost + crossings * 100 })
      }
      next.push({ chosen: state.chosen, unplaced: [...state.unplaced, job.measurement], cost: state.cost })
    }
    const missed = (state: State) => state.unplaced.reduce((sum, m) => sum + (isMetricTable(m) ? 100 : 1), 0)
    next.sort((a, b) => missed(a) - missed(b) || a.cost - b.cost)
    states = next.slice(0, 24)
  }
  const best = states[0]
  return { drawings: best.chosen, unplaced: best.unplaced }
}
