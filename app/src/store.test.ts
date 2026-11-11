import { beforeEach, describe, expect, it } from 'vitest'
import { allRelationships, remeasureRequest, useInspectionStore } from './store'
import type { Analysis, MeasurementRecord } from './types'

beforeEach(() => {
  useInspectionStore.setState({
    analysis: null,
    hovered: null,
    pinned: [],
    draft: [],
    overrideDraft: [],
    relationships: [],
    suspendedRelationships: [],
    hiddenRelationshipIds: [],
    tool: 'select',
  })
})

describe('inspection interaction state', () => {
  it('pins and unpins the same semantic feature', () => {
    useInspectionStore.getState().togglePinned('feature_0001')
    expect(useInspectionStore.getState().pinned).toEqual(['feature_0001'])
    useInspectionStore.getState().togglePinned('feature_0001')
    expect(useInspectionStore.getState().pinned).toEqual([])
  })

  it('clears pinned, hover, and relationship draft selections together', () => {
    useInspectionStore.setState({ hovered: 'feature_0001', pinned: ['feature_0001'], draft: ['feature_0002'], overrideDraft: [[1, 2]] })
    useInspectionStore.getState().clearSelections()
    expect(useInspectionStore.getState()).toMatchObject({ hovered: null, pinned: [], draft: [], overrideDraft: [] })
  })

  it('keeps a relationship draft pair unique and bounded', () => {
    const state = useInspectionStore.getState()
    state.addDraft('a'); state.addDraft('a'); state.addDraft('b'); state.addDraft('c')
    expect(useInspectionStore.getState().draft).toEqual(['a', 'b'])
  })

  it('toggles a measurement annotation independently of its result', () => {
    const result = { id: 'result_1', operator: 'straightness', feature_refs: ['a'], status: 'measured', value: 0.02 }
    const state = useInspectionStore.getState()
    state.addRelationship(result)
    state.toggleRelationshipVisibility(result.id)
    expect(useInspectionStore.getState()).toMatchObject({ relationships: [result], hiddenRelationshipIds: [result.id] })
    useInspectionStore.getState().toggleRelationshipVisibility(result.id)
    expect(useInspectionStore.getState().hiddenRelationshipIds).toEqual([])
  })

  it('cleans hidden annotation state when a result is removed', () => {
    const result = { id: 'result_1', operator: 'circularity_ls', feature_refs: ['a'], status: 'measured', value: 0.02 }
    const state = useInspectionStore.getState()
    state.addRelationship(result)
    state.toggleRelationshipVisibility(result.id)
    state.removeRelationship(result.id)
    expect(useInspectionStore.getState()).toMatchObject({ relationships: [], hiddenRelationshipIds: [] })
  })

  it('removes controls that depend on a deleted datum', () => {
    const state = useInspectionStore.getState()
    state.addRelationship({ id: 'datum-a', operator: 'datum', feature_refs: ['a'], status: 'measured', value: 0, datum_label: 'A', datum_feature_ref: 'a' })
    state.addRelationship({ id: 'parallel-a', operator: 'parallelism', feature_refs: ['a', 'b'], status: 'measured', value: .02, datum_label: 'A', datum_feature_ref: 'a' })
    state.addRelationship({ id: 'unrelated', operator: 'straightness', feature_refs: ['c'], status: 'measured', value: .01 })
    state.toggleRelationshipVisibility('parallel-a')
    state.removeRelationship('datum-a')
    expect(useInspectionStore.getState().relationships.map((result) => result.id)).toEqual(['unrelated'])
    expect(useInspectionStore.getState().hiddenRelationshipIds).toEqual([])
  })

  it('keeps datum-based controls ordered and rejects an exact duplicate', () => {
    const state = useInspectionStore.getState()
    state.addRelationship({ id: 'first', operator: 'parallelism', feature_refs: ['a', 'b'], status: 'measured', value: 0.02 })
    state.addRelationship({ id: 'reversed-datum', operator: 'parallelism', feature_refs: ['b', 'a'], status: 'measured', value: 0.02 })
    state.addRelationship({ id: 'duplicate', operator: 'parallelism', feature_refs: ['a', 'b'], status: 'measured', value: 0.02 })
    state.addRelationship({ id: 'different', operator: 'perpendicularity', feature_refs: ['a', 'b'], status: 'measured', value: 89.98 })
    expect(useInspectionStore.getState().relationships.map((result) => result.id)).toEqual(['different', 'reversed-datum', 'first'])
  })

  it('keeps unrelated measurements when a geometry override invalidates a feature', () => {
    const prediction = {
      schema_version: 1, label_version: 'test', features: [], compound_features: [], relations: [], threshold: 0.5,
    }
    const report: Analysis['report'] = {
      features: [
        { id: 'untouched', type: 'line', status: 'fitted' },
        { id: 'edited', type: 'line', status: 'fitted' },
        { id: 'user_line_0001', type: 'line', status: 'fitted' },
      ],
      measurements: [],
      summary: { measured: 0, failed_or_unavailable: 0, tolerance_failures: 0 },
    }
    const relationships: MeasurementRecord[] = [
      { id: 'keep', operator: 'straightness', feature_refs: ['untouched'], status: 'measured', value: 0.01 },
      { id: 'drop-edited', operator: 'straightness', feature_refs: ['edited'], status: 'measured', value: 0.02 },
      { id: 'drop-missing', operator: 'circularity_ls', feature_refs: ['removed'], status: 'measured', value: 0.03 },
    ]
    useInspectionStore.setState({
      analysis: { path: '', name: '', imageUrl: '', width: 10, height: 10, prediction, report, timingsMs: { edges: 0, recognition: 0, measurement: 0 } },
      relationships,
      hiddenRelationshipIds: ['keep', 'drop-edited'],
    })

    useInspectionStore.getState().applyGeometryOverride(prediction, report, ['edited'])

    expect(useInspectionStore.getState().relationships.map((result) => result.id)).toEqual(['keep'])
    expect(useInspectionStore.getState().hiddenRelationshipIds).toEqual(['keep'])
  })
})

describe('editable distance constructions', () => {
  const gap: MeasurementRecord = { id: 'gap', operator: 'distance_construction', feature_refs: ['a', 'b'], status: 'measured', value: 2,
    definition: { construction: 'normal_gap', basis: 'observed', extend: false, statistic: 'minimum', direction_mode: 'normal_a', branches: ['auto', 'near'] } }
  it('updates the same measurement when only the displayed statistic changes', () => {
    const state = useInspectionStore.getState()
    state.saveDistance(gap)
    state.toggleRelationshipVisibility('gap')
    state.saveDistance({ ...gap, id: 'new-id', value: 4, definition: { ...gap.definition!, statistic: 'maximum' } })
    expect(useInspectionStore.getState().relationships).toHaveLength(1)
    expect(useInspectionStore.getState().relationships[0]).toMatchObject({ id: 'gap', value: 4 })
    expect(useInspectionStore.getState().hiddenRelationshipIds).toEqual(['gap'])
  })
  it('preserves reference order, basis and branch in identity', () => {
    const state = useInspectionStore.getState()
    state.saveDistance(gap)
    state.saveDistance({ ...gap, id: 'reverse', feature_refs: ['b', 'a'] })
    state.saveDistance({ ...gap, id: 'fitted', definition: { ...gap.definition!, basis: 'fitted' } })
    state.saveDistance({ ...gap, id: 'far', definition: { ...gap.definition!, branches: ['auto', 'far'] } })
    expect(useInspectionStore.getState().relationships).toHaveLength(4)
  })
  it('edits in place and discards draft previews on cancellation', () => {
    const state = useInspectionStore.getState()
    state.saveDistance(gap)
    state.editMeasurement(gap.id)
    state.setDistancePreview({ ...gap, value: 9 })
    expect(useInspectionStore.getState().relationships[0].value).toBe(2)
    state.clearSelections()
    expect(useInspectionStore.getState()).toMatchObject({ distancePreview: null, editingMeasurementId: null })
    state.editMeasurement(gap.id)
    state.saveDistance({ ...gap, value: 3, definition: { ...gap.definition!, basis: 'fitted' } })
    expect(useInspectionStore.getState().relationships).toHaveLength(1)
    expect(useInspectionStore.getState().relationships[0]).toMatchObject({ id: 'gap', value: 3 })
  })
})

describe('corner exclusion changes', () => {
  const prediction = { schema_version: 1, label_version: 'test', features: [], compound_features: [], relations: [] }
  const report = (features: Array<{ id: string; status: string }>) => ({
    features: features.map((feature) => ({ ...feature, type: 'line' })), measurements: [],
    summary: { measured: 0, failed_or_unavailable: 0, tolerance_failures: 0 },
  })
  const records: MeasurementRecord[] = [
    { id: 'datum_a', operator: 'datum', feature_refs: ['line'], status: 'measured', value: null, datum_label: 'A', datum_feature_ref: 'line', datum: { direction: [1, 0] } },
    { id: 'short', operator: 'straightness', feature_refs: ['short_line'], status: 'measured', value: 0.1 },
    { id: 'parallel', operator: 'parallelism', feature_refs: ['line', 'other'], status: 'measured', value: 0.2, datum_label: 'A', datum_feature_ref: 'line', datum: { direction: [1, 0] } },
  ]

  it('keeps measurements, suspends ones whose features vanish, and restores them in order', () => {
    useInspectionStore.setState({
      analysis: { path: 'p', name: '', imageUrl: '', width: 10, height: 10, prediction, report: report([]), timingsMs: { edges: 0, recognition: 0, measurement: 0 } },
      relationships: records, hiddenRelationshipIds: ['parallel'],
    })
    const measured = (record: MeasurementRecord, value: number, extra: Partial<MeasurementRecord> = {}) =>
      ({ id: record.id, status: 'measured' as const, result: { ...record, datum_label: undefined, datum: undefined, value, ...extra }, feature_map: {} })

    useInspectionStore.getState().applyCornerExclusion(prediction, report([{ id: 'line', status: 'fitted' }, { id: 'other', status: 'fitted' }]), [
      measured(records[0], 0, { datum: { direction: [0, 1] } }),
      { id: 'short', status: 'unavailable', reason: 'Selected feature is unavailable for measurement' },
      measured(records[2], 0.3),
    ])
    let state = useInspectionStore.getState()
    expect(state.relationships.map((item) => item.id)).toEqual(['datum_a', 'parallel'])
    expect(state.relationships[0]).toMatchObject({ datum_label: 'A', datum: { direction: [0, 1] } })
    expect(state.relationships[1]).toMatchObject({ value: 0.3, datum_label: 'A', datum: { direction: [0, 1] } })
    expect(state.suspendedRelationships.map((item) => item.record.id)).toEqual(['short'])
    expect(state.hiddenRelationshipIds).toEqual(['parallel'])
    expect(allRelationships(state).map(remeasureRequest).map((request) => request.id)).toEqual(['datum_a', 'short', 'parallel'])

    useInspectionStore.getState().applyCornerExclusion(prediction, report([]), allRelationships(state).map((record) => measured(record, 1)))
    state = useInspectionStore.getState()
    expect(state.relationships.map((item) => item.id)).toEqual(['datum_a', 'short', 'parallel'])
    expect(state.suspendedRelationships).toEqual([])
  })

  it('re-sends distance definitions and vertex anchors', () => {
    const gap: MeasurementRecord = { id: 'gap', operator: 'distance_construction', feature_refs: ['vertex_0001', 'b'], status: 'measured', value: 2,
      definition: { construction: 'center_distance', basis: 'fitted', extend: false, statistic: 'minimum' },
      feature_anchors: { vertex_0001: { adjacent_feature_ids: ['a', 'c'], point: [1, 2] } } }
    expect(remeasureRequest(gap)).toEqual({ id: 'gap', operator: 'distance_construction', featureRefs: ['vertex_0001', 'b'],
      params: gap.definition, featureAnchors: gap.feature_anchors })
  })
})
