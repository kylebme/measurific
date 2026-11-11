import { describe, expect, it } from 'vitest'
import { availableConstructions, describeDistance, distanceShortLabel } from './distanceText'
import type { DistanceDefinition, FeatureRecord, MeasurementRecord } from './types'

const line: FeatureRecord = { id: 'l', type: 'line', status: 'fitted', support: [[[0, 0], [10, 0]]], geometry: { center: [5, 0], direction: [1, 0] } }
const circle: FeatureRecord = { id: 'c', type: 'circle', status: 'fitted', support: [[[1, 0], [0, 1], [-1, 0]]], geometry: { center: [0, 0], radius: 1 } }
const corner: FeatureRecord = { id: 'v', type: 'point', status: 'fitted', semantic: 'vertex', geometry: { center: [3, 3] } }
const def = (d: Partial<DistanceDefinition>): DistanceDefinition => ({ construction: 'center_distance', basis: 'fitted', extend: false, statistic: 'minimum', ...d })

describe('distance constructions offered for a selection', () => {
  it('offers only constructions that can succeed, most specific first', () => {
    expect(availableConstructions([line, circle])).toEqual(['center_to_line', 'normal_gap', 'section', 'closest_clearance'])
    expect(availableConstructions([corner, line])).toEqual(['center_to_line', 'closest_clearance'])
    expect(availableConstructions([circle, circle])).toEqual(['center_distance', 'closest_clearance', 'normal_gap', 'section'])
    expect(availableConstructions([corner, corner])).toEqual(['center_distance', 'closest_clearance'])
    expect(availableConstructions([line, line])).toEqual(['normal_gap', 'section', 'closest_clearance'])
  })
})

describe('distance descriptions', () => {
  it('names the points, sides and direction in plain language', () => {
    expect(describeDistance(def({ construction: 'center_to_line', anchors: ['far', 'center'] }), [circle, line]))
      .toBe('Perpendicular distance from line B to the far side of circle A.')
    expect(describeDistance(def({ direction_mode: 'x', anchors: ['center', 'near'] }), [corner, circle]))
      .toBe('Horizontal distance (ΔX) from corner A to the near side of circle B.')
    expect(describeDistance(def({ direction_mode: 'custom', datum_feature_ref: 'l', datum_axis: 'along', anchors: ['center', 'center'] }), [circle, circle], 'A'))
      .toBe('Distance measured parallel to datum A from the center of circle A to the center of circle B.')
    expect(describeDistance(def({ construction: 'tangent_clearance' }), [line, circle]))
      .toBe('Perpendicular distance from line A to the near side of circle B.')
  })
  it('builds compact on-image labels', () => {
    const m = (d: Partial<DistanceDefinition>): MeasurementRecord => ({ id: 'm', operator: 'distance_construction', feature_refs: [], status: 'measured', value: 1, definition: def(d) })
    expect(distanceShortLabel(m({ direction_mode: 'y', anchors: ['center', 'center'] }))).toBe('Centers ΔY')
    expect(distanceShortLabel(m({ anchors: ['near', 'near'] }))).toBe('Distance')
    expect(distanceShortLabel(m({ construction: 'center_to_line', anchors: ['center', 'far'] }))).toBe('Far side to line')
  })
})
