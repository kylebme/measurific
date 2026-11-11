import type { DistanceAnchor, DistanceDefinition, FeatureRecord, MeasurementRecord } from './types'

type Construction = DistanceDefinition['construction']

// Legacy `tangent_clearance` is still measured, but is edited as point-to-line with a near-side anchor.
export const constructionNames: Record<Construction, string> = {
  center_to_line: 'Point to line', center_distance: 'Point to point', normal_gap: 'Gap along edges',
  section: 'Gap at one position', closest_clearance: 'Shortest distance', tangent_clearance: 'Point to line',
}
export const constructionHints: Record<Construction, string> = {
  center_to_line: 'Perpendicular to the line, from a center, corner or circle side',
  center_distance: 'Between centers, corners or circle sides; straight or along an axis',
  normal_gap: 'Smallest, largest and average gap where both edges are visible',
  section: 'The gap at a single position you choose',
  closest_clearance: 'Closest points on the actual edges, in any direction',
  tangent_clearance: 'Perpendicular to the line, to the near side of the circle',
}
export const anchorNames: Record<DistanceAnchor, string> = { center: 'Center', near: 'Near side', far: 'Far side' }

/** Compact display id: base_00007 → #7, compound_0003 → C3. */
export const shortFeatureId = (id: string) => id.replace(/^(?:feature|base)_0*(?=\d)/, '#').replace(/^compound_0*(?=\d)/, 'C')
export const isRound = (f?: FeatureRecord) => f?.type === 'circle' || f?.type === 'arc'
export const isCorner = (f?: FeatureRecord) => f?.type === 'point'
const hasCenter = (f: FeatureRecord) => f.type !== 'line' && Boolean(f.geometry?.center)
const hasEdges = (f: FeatureRecord) => f.type !== 'point' && Boolean(f.support?.length)

export function featureNoun(f?: FeatureRecord) {
  if (!f) return 'feature'
  if (isCorner(f)) return 'corner'
  return (f.label_level === 'compound' && f.semantic ? f.semantic : f.type).replaceAll('_', ' ')
}

/** Constructions that can succeed for this pair, most specific first. */
export function availableConstructions(features: FeatureRecord[]): Construction[] {
  const lines = features.filter(f => f.type === 'line').length
  const out: Construction[] = []
  if (lines === 1 && features.some(hasCenter)) out.push('center_to_line')
  if (features.every(hasCenter)) out.push('center_distance')
  if (features.every(hasEdges)) out.push(...(lines ? ['normal_gap', 'section'] as const : []))
  out.push('closest_clearance')
  if (features.every(hasEdges) && !lines) out.push('normal_gap', 'section')
  return out
}

function directionPhrase(d: DistanceDefinition, datumLabel?: string) {
  if (d.datum_feature_ref) return `${d.datum_axis === 'along' ? 'parallel' : 'perpendicular'} to datum ${datumLabel || '?'}`
  switch (d.direction_mode) {
    case 'x': return 'horizontally'
    case 'y': return 'vertically'
    case 'custom': return `at ${Math.round(Math.atan2(d.direction?.[1] ?? 1, d.direction?.[0] ?? 0)*1800/Math.PI)/10}° from horizontal`
    case 'normal_b': return 'perpendicular to B'
    default: return 'perpendicular to A'
  }
}

/** One plain-English sentence describing exactly what is measured. */
export function describeDistance(d: DistanceDefinition, features: (FeatureRecord | undefined)[], datumLabel?: string) {
  const name = (i: number) => `${featureNoun(features[i])} ${'AB'[i]}`
  const point = (i: number) => {
    const anchor = d.construction === 'tangent_clearance' && features[i]?.type !== 'line' ? 'near' : d.anchors?.[i] || 'center'
    if (anchor !== 'center') return `the ${anchor} side of ${name(i)}`
    return isCorner(features[i]) ? name(i) : `the center of ${name(i)}`
  }
  switch (d.construction) {
    case 'center_to_line':
    case 'tangent_clearance': {
      const line = features[1]?.type === 'line' ? 1 : 0
      return `Perpendicular distance from line ${'AB'[line]} to ${point(1-line)}${d.extend ? ', extending the line where needed' : ''}.`
    }
    case 'center_distance': {
      const how = !d.direction_mode || d.direction_mode === 'direct' ? 'Straight-line distance'
        : d.direction_mode === 'x' && !d.datum_feature_ref ? 'Horizontal distance (ΔX)'
          : d.direction_mode === 'y' && !d.datum_feature_ref ? 'Vertical distance (ΔY)' : `Distance measured ${directionPhrase(d, datumLabel)}`
      return `${how} from ${point(0)} to ${point(1)}.`
    }
    case 'closest_clearance': return `Shortest distance between the actual edges of ${name(0)} and ${name(1)}, in any direction.`
    case 'section': return `Gap from ${name(0)} to ${name(1)} at one position, measured ${directionPhrase(d, datumLabel)}.`
    default: return `Gap from ${name(0)} to ${name(1)}, measured ${directionPhrase(d, datumLabel)} wherever both are visible.`
  }
}

/** Compact prefix for on-image dimension labels. */
export function distanceShortLabel(m: MeasurementRecord) {
  const d = m.definition
  const stat = m.selected_statistic ? ' ' + ({ minimum: 'min', maximum: 'max', average: 'average' }[m.selected_statistic]) : ''
  switch (d?.construction) {
    case 'center_to_line': {
      const anchor = d.anchors?.find(a => a !== 'center')
      return anchor ? `${anchorNames[anchor]} to line` : 'Point to line'
    }
    case 'tangent_clearance': return 'Near side to line'
    case 'center_distance': {
      const base = d.anchors?.some(a => a !== 'center') ? 'Distance' : 'Centers'
      const axis = d.datum_feature_ref ? (d.datum_axis === 'along' ? ' ∥ datum' : ' ⊥ datum')
        : ({ x: ' ΔX', y: ' ΔY', custom: ' ∠' } as Record<string, string>)[d.direction_mode || ''] || ''
      return base + axis
    }
    case 'section': return 'Section gap'
    case 'closest_clearance': return 'Clearance'
    default: return 'Gap' + stat
  }
}
