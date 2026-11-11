import thread25Fixture from './fixtures/thread25.json'
import threadFixture from './fixtures/threadMeasurements.json'
import { describe, expect, it } from 'vitest'
import { expandCompoundAnnotations, boxesIntersect, containsBox, expanded, formattedValue, layoutAnnotations, segmentHitsBox, segmentsCross, solveStrip, type AnnotationDrawing, type LayoutInput } from './annotationLayout'
import type { FeatureRecord, MeasurementRecord, Point } from './types'

const stage = { x: 12, y: 12, width: 876, height: 626 }
const identity = (p: Point): Point => p
const outline: FeatureRecord = { id: 'outline', type: 'polyline', status: 'fitted', support: [[[180, 160], [640, 160], [640, 470], [180, 470], [180, 160]]] }
const distance = (id: string, a: Point, b: Point): MeasurementRecord => ({ id, operator: 'distance_construction', feature_refs: ['a', 'b'], status: 'measured', value: Math.hypot(b[0] - a[0], b[1] - a[1]), units: 'px', witness: [a, b] })
const line = (id: string, a: Point, b: Point): FeatureRecord => ({ id, type: 'line', status: 'fitted', support: [[a, b]], geometry: { center: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], direction: [b[0] - a[0], b[1] - a[1]] } })
const angle: MeasurementRecord = { id: 'angle', operator: 'angle', feature_refs: ['a', 'b'], status: 'measured', value: 45, units: 'deg' }
const run = (measurements: MeasurementRecord[], overrides: Partial<LayoutInput> = {}) => layoutAnnotations({ measurements, features: [outline], stage, toStage: identity, ...overrides })
function clear(drawings: AnnotationDrawing[]) {
  for (const [i, a] of drawings.entries()) {
    expect(containsBox(stage, a.box)).toBe(true)
    for (const b of drawings.slice(i + 1)) {
      expect(boxesIntersect(expanded(a.box, 4), b.box)).toBe(false)
      expect(a.segments.some(s => segmentHitsBox(s, expanded(b.box, 2)))).toBe(false)
      expect(b.segments.some(s => segmentHitsBox(s, expanded(a.box, 2)))).toBe(false)
    }
  }
}

describe('boundary strip placement', () => {
  it('preserves feature order, heterogeneous widths, gaps and bounds', () => {
    const widths = [60, 110, 70], positions = solveStrip([10, 30, 190], widths, 0, 300)!
    expect(positions[0] - widths[0] / 2).toBeGreaterThanOrEqual(0)
    expect(positions[2] + widths[2] / 2).toBeLessThanOrEqual(300)
    for (let i = 1; i < positions.length; i++) expect(positions[i] - positions[i - 1]).toBeGreaterThanOrEqual((widths[i] + widths[i - 1]) / 2 + 8)
    expect(solveStrip([10, 30], [100, 100], 0, 180)).toBeNull()
  })
  it('minimizes displacement against a brute-force small placement oracle', () => {
    for (const naturals of [[2, 2, 3], [0, 6, 6], [1, 2, 8], [5, 5, 5]]) {
      const actual = solveStrip(naturals, [2, 2, 2], 0, 10, 1)!
      let optimum = Infinity
      for (let a = 1; a <= 9; a++) for (let b = a + 3; b <= 9; b++) for (let c = b + 3; c <= 9; c++) {
        optimum = Math.min(optimum, [a, b, c].reduce((sum, p, i) => sum + Math.abs(p - naturals[i]), 0))
      }
      expect(actual.reduce((sum, p, i) => sum + Math.abs(p - naturals[i]), 0)).toBe(optimum)
    }
  })
})

describe('mechanical annotation geometry and occupancy', () => {
  it('keeps aligned distance extensions perpendicular and arrow tips on exact witnesses', () => {
    const m = distance('horizontal', [220, 160], [580, 160]), result = run([m])
    expect(result.unplaced).toEqual([])
    const d = result.drawings[0]
    expect(d.kind).toBe('distance')
    expect(d.witnesses).toEqual(m.witness)
    expect(d.segments[0].start[0]).toBe(220)
    expect(d.segments[1].start[0]).toBe(580)
    expect(d.segments[2].start[1]).toBe(d.segments[2].end[1])
    expect(d.label[1]).toBeLessThan(160)
    clear(result.drawings)
  })
  it('does not send a diagonal dimension to the empty corner of the part bounding box', () => {
    const features: FeatureRecord[] = [{ ...outline, support: [[[180, 240], [460, 520], [500, 480], [220, 200], [180, 240]]] }]
    const m = distance('diagonal', [220, 280], [420, 480]), d = run([m], { features }).drawings[0]
    expect(d.kind).toBe('distance')
    expect(d.rotation).toBeCloseTo(45)
    expect(Math.hypot(d.label[0] - 320, d.label[1] - 380)).toBeLessThan(60)
  })
  it('makes selection direction immaterial, including vertical arrow direction', () => {
    for (const [a, b] of [[[220, 160], [580, 160]], [[180, 200], [180, 420]], [[220, 200], [520, 400]]] as [Point, Point][]) {
      const geometry = (m: MeasurementRecord) => run([m]).drawings.map(({ measurement: _measurement, ...drawing }) => drawing)
      expect(geometry(distance('d', a, b))).toEqual(geometry(distance('d', b, a)))
    }
  })
  it('moves a short-span label beyond the arrowheads', () => {
    const d = run([distance('short', [220, 160], [238, 160])]).drawings[0]
    expect(d.kind).toBe('distance')
    expect(d.label[0] < 210 || d.label[0] > 248).toBe(true)
    expect(d.arrows[0]).toContain('L 212 ')
    expect(d.arrows[1]).toContain('L 246 ')
  })
  it('does not flip vertical text when fitted x noise changes sign', () => {
    const rotations = [-.01, .01].map(noise => run([distance('vertical', [180, 200], [180 + noise, 420])]).drawings[0].rotation)
    expect(Math.abs(rotations[0] - rotations[1])).toBeLessThan(.01)
    expect(rotations[0]).toBeCloseTo(-90, 1)
  })
  it('uses the selected statistic when the combined witness field is absent', () => {
    const m = { ...distance('maximum', [200, 160], [600, 160]), witness: undefined,
      selected_statistic: 'maximum' as const, minimum_witness: [[300, 160], [310, 160]] as Point[], maximum_witness: [[200, 160], [600, 160]] as Point[] }
    expect(run([m]).drawings[0].witnesses).toEqual(m.maximum_witness)
  })
  it('uses a zero-distance callout without manufacturing a dimension axis', () => {
    const result = run([distance('zero', [220, 160], [220, 160])])
    expect(result.drawings[0].kind).toBe('callout')
    expect(result.drawings[0].arrows).toEqual([])
    expect(result.drawings[0].witnesses).toEqual([[220, 160], [220, 160]])
  })
  it('draws an angle about its real intersection even outside the part envelope', () => {
    const result = run([angle], { features: [line('a', [250, 150], [500, 150]), line('b', [250, 300], [400, 450])] })
    const d = result.drawings[0]
    expect(d.kind).toBe('angle')
    expect(d.arc!.center[0]).toBeCloseTo(100)
    expect(d.arc!.center[1]).toBeCloseTo(150)
    expect(d.arc!.sweep * 180 / Math.PI).toBeCloseTo(45)
    expect(d.segments.slice(0, 16).some(s => segmentHitsBox(s, d.box))).toBe(false)
    expect(d.segments[16].start[1]).toBeCloseTo(150)
    expect(d.segments[17].start[1] - d.segments[17].start[0]).toBeCloseTo(50)
  })
  it('never invents an angle vertex for parallel or nearly parallel separated lines', () => {
    for (const rise of [0, .001, 1]) {
      const d = run([{ ...angle, value: 0 }], { features: [line('a', [220, 200], [550, 200]), line('b', [220, 350], [550, 350 + rise])] }).drawings[0]
      expect(d.kind).toBe('callout')
      expect(d.arc).toBeUndefined()
      expect(d.witnesses).toHaveLength(2)
    }
  })
  it('packs distance, angle and GD&T together and is stable under insertion order', () => {
    const features = [outline, line('a', [220, 160], [560, 160]), line('b', [220, 160], [450, 390])]
    const measurements = [distance('small', [260, 160], [390, 160]), distance('large', [220, 160], [580, 160]), angle,
      { ...angle, id: 'parallel', operator: 'parallelism', value: .002 }]
    const result = run(measurements, { features })
    expect(result.unplaced).toEqual([])
    expect(result.drawings).toHaveLength(4)
    clear(result.drawings)
    expect(run([...measurements].reverse(), { features })).toEqual(result)
  })
  it('uses a separate datum marker and only points a datum-based control at its target', () => {
    const features = [outline, line('a', [220, 160], [560, 160]), line('b', [220, 360], [560, 360])]
    const datum: MeasurementRecord = { id: 'datum-a', operator: 'datum', feature_refs: ['a'], status: 'measured', value: 0, units: 'deg', datum_label: 'A' }
    const parallel: MeasurementRecord = { id: 'parallel', operator: 'parallelism', feature_refs: ['a', 'b'], status: 'measured', value: .002, units: 'deg', datum_label: 'A' }
    const result = run([datum, parallel], { features })
    expect(result.unplaced).toEqual([])
    const datumDrawing = result.drawings.find((drawing) => drawing.measurement.id === datum.id)!
    const controlDrawing = result.drawings.find((drawing) => drawing.measurement.id === parallel.id)!
    expect(datumDrawing.witnesses).toHaveLength(1)
    expect(controlDrawing.witnesses).toHaveLength(1)
    expect(controlDrawing.witnesses[0][1]).toBeCloseTo(360)
  })
  it('treats pinned cards as fixed obstacles', () => {
    const measurements = [distance('d', [220, 160], [580, 160])], first = run(measurements).drawings[0]
    const result = run(measurements, { obstacles: [expanded(first.box, 8)] })
    expect(result.unplaced).toEqual([])
    expect(boxesIntersect(result.drawings[0].box, expanded(first.box, 8))).toBe(false)
  })
  it('uses measured text dimensions and reserves rotated bounds', () => {
    const result = run([distance('wide', [220, 160], [580, 160])], { sizes: new Map([['wide', { width: 260, height: 34 }]]) })
    expect(result.drawings[0].box.width).toBe(260)
    expect(result.drawings[0].box.height).toBe(34)
  })
  it('reports an overfull viewport instead of choosing an off-screen least-bad label', () => {
    const result = run([distance('d', [220, 160], [580, 160])], { stage: { x: 0, y: 0, width: 50, height: 20 } })
    expect(result.drawings).toEqual([])
    expect(result.unplaced.map(m => m.id)).toEqual(['d'])
  })
  it('reports crowding instead of drawing crossing fallback leaders', () => {
    const measurements = Array.from({ length: 12 }, (_, i) => distance(`dense-${i}`, [200 + i * 6, 160], [480 + i * 8, 160]))
    const result = run(measurements)
    expect(result.drawings.length + result.unplaced.length).toBe(measurements.length)
    clear(result.drawings)
    for (const a of result.drawings.filter(d => d.kind === 'callout')) for (const b of result.drawings.filter(d => d !== a)) {
      expect(a.segments.some(s => b.segments.some(t => segmentsCross(s, t)))).toBe(false)
    }
  })
  it('retains exact witnesses through pan and zoom while keeping text inside the viewport', () => {
    const transform = ([x, y]: Point): Point => [x * 2 - 350, y * 2 - 180]
    const m = distance('zoom', [220, 160], [580, 160]), result = run([m], { toStage: transform })
    expect(result.unplaced).toEqual([])
    expect(result.drawings[0].witnesses).toEqual(m.witness!.map(transform))
    clear(result.drawings)
  })
  it('ignores failed measurements and prevents nonfinite SVG coordinates', () => {
    const result = run([{ ...distance('bad', [NaN, 0], [100, 100]) , value: 10 }, { ...angle, status: 'failed' }])
    expect(result.drawings).toEqual([])
    expect(result.unplaced.map(m => m.id)).toEqual(['bad'])
  })
})


describe('thread annotations', () => {
  const fixture = threadFixture as unknown as { features: FeatureRecord[]; measurements: MeasurementRecord[] }
  it('expands all extrema and keeps one summary with shared visibility ownership', () => {
    const expanded = expandCompoundAnnotations(fixture.measurements)
    expect(expanded).toHaveLength(7)
    expect(expanded.filter(m => m.operator === 'thread')).toHaveLength(1)
    for (const m of expanded.slice(1)) {
      expect(m.annotation_parent_id).toBe(fixture.measurements[0].id)
      expect(m.witness).toHaveLength(2)
      expect(m.annotation_label).toMatch(/min|max/)
    }
  })
  it('places the summary with a leader and extrema through the shared layout', () => {
    const measurements = expandCompoundAnnotations(fixture.measurements)
    const result = run(measurements, { features: fixture.features,
      sizes: new Map([[fixture.measurements[0].id, { width: 400, height: 195 }]]) })
    const summary = result.drawings.find(d => d.measurement.operator === 'thread')
    expect(summary?.kind).toBe('callout')
    expect(summary?.segments.length).toBeGreaterThan(0)
    expect(result.drawings.length + result.unplaced.length).toBe(7)
    clear(result.drawings)
  })
  it('preserves distinct minimum and maximum witnesses', () => {
    const source = fixture.measurements[0]
    const metric = { ...source.thread_metrics!.pitch, maximum: 64, maximum_witness: [[350, 240], [414, 240]] as Point[] }
    const dimensions = expandCompoundAnnotations([{ ...source, thread_metrics: { pitch: metric } }]).slice(1)
    expect(dimensions).toHaveLength(2)
    expect(dimensions.every(d => d.operator === 'compound_dimension')).toBe(true)
    expect(dimensions[0].witness).toEqual(metric.minimum_witness)
    expect(dimensions[1].witness).toEqual(metric.maximum_witness)
    expect(dimensions[1].value).toBe(64)
  })
  it('places every image 25 extremum in horizontal gutters without overlapping labels or leaders', () => {
    const fixture25 = thread25Fixture as unknown as { features: FeatureRecord[]; measurements: MeasurementRecord[] }
    const measurements = expandCompoundAnnotations(fixture25.measurements)
    const input: LayoutInput = { measurements, features: fixture25.features,
      stage: { x: 12, y: 12, width: 1256, height: 650 }, toStage: ([x, y]) => [x + 200, y],
      sizes: new Map([[measurements[0].id, { width: 375, height: 190 }]]) }
    const result = layoutAnnotations(input)
    expect(result.unplaced).toEqual([])
    expect(result.drawings).toHaveLength(measurements.length)
    for (const [i, a] of result.drawings.entries()) {
      expect(a.rotation).toBe(0)
      expect(containsBox(input.stage, a.box)).toBe(true)
      for (const b of result.drawings.slice(i + 1)) {
        expect(boxesIntersect(expanded(a.box, 4), b.box)).toBe(false)
        expect(a.segments.some(s => segmentHitsBox(s, expanded(b.box, 2)))).toBe(false)
        expect(a.segments.some(s => b.segments.some(t => segmentsCross(s, t)))).toBe(false)
      }
      if (a.measurement.annotation_parent_id) expect(a.segments.length).toBeLessThanOrEqual(3)
      if (a.measurement.annotation_metric === 'number_of_threads') {
        expect(a.arrows).toEqual([])
        expect(a.witnesses).toHaveLength(2)
        expect(a.segments).toHaveLength(2)
      }
    }
    expect(layoutAnnotations({ ...input, measurements: [...measurements].reverse() })).toEqual(result)
  })
  it('does not invent extrema for unresolved radii', () => {
    const source = fixture.measurements[0]
    const metrics = { ...source.thread_metrics!, root_radius: { status: 'unavailable', minimum: null, maximum: null, average: null, units: 'px', count: 0 } }
    expect(expandCompoundAnnotations([{ ...source, thread_metrics: metrics }])).toHaveLength(6)
  })
})

describe('size and compound measurements', () => {
  const metric = (value: number, units: string, witness: Point[]) => ({ status: 'measured', minimum: value, maximum: value, average: value, units, count: 1,
    minimum_witness: witness, maximum_witness: witness })
  it('draws diameter, radius and length as dimensions between their witnesses', () => {
    const diameter: MeasurementRecord = { id: 'd', operator: 'least_squares_diameter', feature_refs: ['outline'], status: 'measured', value: 120, units: 'px', witness: [[300, 300], [420, 300]] }
    const length: MeasurementRecord = { id: 'l', operator: 'length', feature_refs: ['outline'], status: 'measured', value: 460, units: 'px', witness: [[180, 160], [640, 160]] }
    expect(formattedValue(diameter)).toBe('Ø120.0 px')
    expect(formattedValue({ ...diameter, operator: 'radius', value: 8 })).toBe('R8.000 px')
    const result = run([diameter, length])
    expect(result.unplaced).toEqual([])
    expect(result.drawings.map(d => d.kind)).toEqual(['distance', 'distance'])
    clear(result.drawings)
  })
  it('expands tooth profile extrema but keeps angular pitch in the table only', () => {
    const teeth: MeasurementRecord = { id: 'teeth', operator: 'tooth_profile', feature_refs: ['outline'], status: 'measured', value: 30, units: 'deg',
      compound_metrics: { pitch: metric(30, 'deg', [[300, 200], [340, 210]]), tooth_height: metric(10, 'px', [[400, 160], [400, 170]]),
        number_of_teeth: metric(12, 'count', [[200, 160], [600, 160]]) } }
    const expanded = expandCompoundAnnotations([teeth])
    expect(expanded.map(m => m.annotation_metric)).toEqual([undefined, 'tooth_height', 'number_of_teeth'])
    expect(expanded[1]).toMatchObject({ operator: 'compound_dimension', annotation_label: 'Height min/max', annotation_parent_id: 'teeth' })
    expect(formattedValue(expanded[2])).toBe('Teeth min/max: 12')
    const result = run(expanded, { sizes: new Map([['teeth', { width: 300, height: 100 }]]) })
    expect(result.unplaced).toEqual([])
    const count = result.drawings.find(d => d.measurement.annotation_metric === 'number_of_teeth')
    expect(count?.arrows).toEqual([])
    clear(result.drawings)
  })
  it('expands slot metrics with the slot table as their parent', () => {
    const slot: MeasurementRecord = { id: 'slot', operator: 'slot', feature_refs: ['outline'], status: 'measured', value: 10, units: 'px',
      compound_metrics: { width: metric(10, 'px', [[300, 300], [300, 310]]), overall_length: metric(30, 'px', [[285, 305], [315, 305]]) } }
    expect(expandCompoundAnnotations([slot]).slice(1).map(m => m.annotation_label)).toEqual(['Width min/max', 'Slot length min/max'])
  })
})

describe('section evidence annotations', () => {
  const features = [line('a', [200, 200], [500, 200]), line('b', [300, 300], [600, 300])]
  it('preserves a normal section in distance placement', () => {
    const m: MeasurementRecord = { ...distance('gap', [300, 200], [300, 300]), operator: 'distance_construction',
      definition: { construction: 'normal_gap', basis: 'observed', extend: false, statistic: 'maximum' } }
    const result = run([m], { features })
    expect(result.unplaced).toEqual([])
    const [a, b] = result.drawings[0].witnesses
    expect(a[0]).toBe(b[0])
  })
  it('attaches a mean callout to the contributing band without inventing a dimension', () => {
    const m: MeasurementRecord = { id: 'mean', operator: 'distance_construction', feature_refs: ['a', 'b'], status: 'measured', value: 100,
      selected_statistic: 'average', field_bands: [[[[300, 200], [300, 300]], [[500, 200], [500, 300]]]],
      definition: { construction: 'normal_gap', basis: 'observed', extend: false, statistic: 'average' } }
    const result = run([m], { features })
    expect(result.unplaced).toEqual([])
    expect(result.drawings[0].kind).toBe('callout')
    expect(result.drawings[0].witnesses).toHaveLength(1)
    expect(result.drawings[0].segments.every(s => s.role === 'leader')).toBe(true)
  })
})

describe('axis-projected distances', () => {
  it('draws a horizontal dimension with extension lines from diagonal witness points', () => {
    const m: MeasurementRecord = { ...distance('dx', [260, 250], [520, 380]), value: 260, dimension_direction: [1, 0],
      definition: { construction: 'center_distance', basis: 'fitted', extend: false, statistic: 'minimum', direction_mode: 'x', anchors: ['center', 'center'] } }
    const [drawing] = run([m]).drawings
    expect(drawing.kind).toBe('distance')
    const dimension = drawing.segments.find(s => !s.role)!
    expect(dimension.start[1]).toBeCloseTo(dimension.end[1])
    expect(Math.abs(dimension.end[0] - dimension.start[0])).toBeGreaterThanOrEqual(260)
    for (const e of drawing.segments.filter(s => s.role === 'extension')) expect(e.start[0]).toBeCloseTo(e.end[0])
  })
})
