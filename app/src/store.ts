import { create } from 'zustand'
import type { Analysis, ExtractionFailure, MeasurementRecord, Point, RemeasureOutcome, RemeasureRequest, Tool } from './types'

/** A measurement whose features are unavailable under the current settings. */
export interface SuspendedMeasurement {
  index: number
  record: MeasurementRecord
  reason: string
}

interface AppState {
  analysis: Analysis | null
  extractionFailure: ExtractionFailure | null
  loading: boolean
  error: string | null
  tool: Tool
  hovered: string | null
  pinned: string[]
  draft: string[]
  overrideDraft: Point[]
  relationships: MeasurementRecord[]
  suspendedRelationships: SuspendedMeasurement[]
  distancePreview: MeasurementRecord | null
  editingMeasurementId: string | null
  setDistancePreview: (result: MeasurementRecord | null) => void
  editMeasurement: (id: string | null) => void
  saveDistance: (result: MeasurementRecord) => void
  hiddenRelationshipIds: string[]
  deformationFactor: number
  setAnalysis: (analysis: Analysis | null) => void
  setExtractionFailure: (failure: ExtractionFailure | null) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
  setTool: (tool: Tool) => void
  setHovered: (id: string | null) => void
  togglePinned: (id: string) => void
  addDraft: (id: string, limit?: number) => void
  toggleDraft: (id: string) => void
  clearSelections: () => void
  clearDraft: () => void
  setOverrideDraft: (points: Point[]) => void
  applyGeometryOverride: (prediction: Analysis['prediction'], report: Analysis['report'], invalidatedFeatureIds?: string[]) => void
  applyCornerExclusion: (prediction: Analysis['prediction'], report: Analysis['report'], outcomes: RemeasureOutcome[]) => void
  addRelationship: (result: MeasurementRecord) => void
  removeRelationship: (id: string) => void
  toggleRelationshipVisibility: (id: string) => void
  setDeformationFactor: (factor: number) => void
}

export function measurementIdentity(operator: string, featureRefs: string[], variant: {
  definition?: unknown
} = {}) {
  if (operator === 'distance_construction') {
    const definition = (variant.definition || variant) as Record<string, unknown>
    const semantic = Object.fromEntries(Object.entries(definition).filter(([key]) => key !== 'statistic').sort(([a], [b]) => a.localeCompare(b)))
    return `${operator}:${featureRefs.join('|')}:${JSON.stringify(semantic)}`
  }
  // For datum-based controls the first feature is the reference and the second
  // is the controlled feature, so reversing them creates a different control.
  if (['concentricity', 'parallelism', 'perpendicularity'].includes(operator)) {
    return `${operator}:${featureRefs.join('|')}`
  }
  return `${operator}:${[...featureRefs].sort().join('|')}`
}

/** Every saved measurement, active and suspended, in list order. */
export function allRelationships(state: Pick<AppState, 'relationships' | 'suspendedRelationships'>) {
  const ordered = [...state.relationships]
  for (const item of [...state.suspendedRelationships].sort((a, b) => a.index - b.index)) {
    ordered.splice(Math.min(item.index, ordered.length), 0, item.record)
  }
  return ordered
}

export function remeasureRequest(record: MeasurementRecord): RemeasureRequest {
  return {
    id: record.id, operator: record.operator, featureRefs: record.feature_refs,
    params: record.operator === 'distance_construction' && record.definition ? { ...record.definition } : {},
    featureAnchors: record.feature_anchors,
  }
}

export const useInspectionStore = create<AppState>((set) => ({
  distancePreview: null,
  editingMeasurementId: null,
  setDistancePreview: (distancePreview) => set({ distancePreview }),
  editMeasurement: (editingMeasurementId) => set({ editingMeasurementId, distancePreview: null, tool: 'distance_construction', draft: [] }),
  saveDistance: (result) => set((state) => {
    const existing = state.relationships.find(item => item.id === state.editingMeasurementId) || state.relationships.find(item =>
      measurementIdentity(item.operator, item.feature_refs, item) === measurementIdentity(result.operator, result.feature_refs, result))
    const saved = { ...result, id: existing?.id || result.id }
    return { relationships: existing ? state.relationships.map(item => item.id === existing.id ? saved : item) : [saved, ...state.relationships],
      editingMeasurementId: null, distancePreview: null, draft: [] }
  }),
  analysis: null,
  extractionFailure: null,
  loading: false,
  error: null,
  tool: 'select',
  hovered: null,
  pinned: [],
  draft: [],
  overrideDraft: [],
  relationships: [],
  suspendedRelationships: [],
  hiddenRelationshipIds: [],
  deformationFactor: 1,
  setAnalysis: (analysis) => set({ analysis, extractionFailure: null, distancePreview: null, editingMeasurementId: null, relationships: [], suspendedRelationships: [], hiddenRelationshipIds: [], pinned: [], draft: [], overrideDraft: [], hovered: null, deformationFactor: 1 }),
  setExtractionFailure: (extractionFailure) => set({ extractionFailure, analysis: null, distancePreview: null, editingMeasurementId: null, relationships: [], suspendedRelationships: [], hiddenRelationshipIds: [], pinned: [], draft: [], overrideDraft: [], hovered: null, deformationFactor: 1 }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
  setTool: (tool) => set({ tool, distancePreview: null, editingMeasurementId: null, draft: [], overrideDraft: [] }),
  setHovered: (hovered) => set({ hovered }),
  togglePinned: (id) => set((state) => ({
    pinned: state.pinned.includes(id) ? state.pinned.filter((value) => value !== id) : [...state.pinned, id],
  })),
  addDraft: (id, limit = 2) => set((state) => ({
    draft: state.draft.includes(id) || state.draft.length >= limit ? state.draft : [...state.draft, id],
  })),
  toggleDraft: (id) => set((state) => ({
    draft: state.draft.includes(id) ? state.draft.filter((value) => value !== id) : [...state.draft, id],
  })),
  clearSelections: () => set({ distancePreview: null, editingMeasurementId: null, pinned: [], draft: [], overrideDraft: [], hovered: null }),
  clearDraft: () => set({ draft: [] }),
  setOverrideDraft: (overrideDraft) => set({ overrideDraft }),
  applyGeometryOverride: (prediction, report, invalidatedFeatureIds = []) => set((state) => {
    const invalidated = new Set(invalidatedFeatureIds)
    const available = new Set(report.features.map((feature) => feature.id))
    const retained = (measurement: MeasurementRecord) => [...measurement.feature_refs, ...(measurement.definition?.datum_feature_ref ? [measurement.definition.datum_feature_ref] : [])].every(
      (featureId) => available.has(featureId) && !invalidated.has(featureId),
    )
    const relationships = state.relationships.filter(retained)
    const suspendedRelationships = state.suspendedRelationships.filter((item) => retained(item.record))
    const retainedMeasurementIds = new Set([...relationships, ...suspendedRelationships.map((item) => item.record)].map((measurement) => measurement.id))
    return {
      analysis: state.analysis ? { ...state.analysis, prediction, report } : null,
      relationships,
      suspendedRelationships,
      hiddenRelationshipIds: state.hiddenRelationshipIds.filter((id) => retainedMeasurementIds.has(id)),
      distancePreview: null, editingMeasurementId: null, pinned: [], draft: [], overrideDraft: [], hovered: null,
      deformationFactor: 1,
    }
  }),
  applyCornerExclusion: (prediction, report, outcomes) => set((state) => {
    const byId = new Map(outcomes.map((outcome) => [outcome.id, outcome]))
    const previouslySuspended = new Map(state.suspendedRelationships.map((item) => [item.record.id, item]))
    const active: MeasurementRecord[] = []
    const suspendedRelationships: SuspendedMeasurement[] = []
    allRelationships(state).forEach((record, index) => {
      const outcome = byId.get(record.id)
      if (!outcome) {
        // Added while this update was in flight; keep its current status.
        const suspended = previouslySuspended.get(record.id)
        if (suspended) suspendedRelationships.push({ ...suspended, index })
        else active.push(record)
      } else if (outcome.status === 'unavailable') {
        suspendedRelationships.push({ index, record, reason: outcome.reason })
      } else {
        const datumRef = record.datum_feature_ref && (outcome.feature_map[record.datum_feature_ref] || record.datum_feature_ref)
        active.push({
          ...outcome.result, id: record.id,
          ...(record.datum_label ? { datum_label: record.datum_label } : {}),
          ...(datumRef ? { datum_feature_ref: datumRef } : {}),
          ...(outcome.result.datum || record.datum ? { datum: outcome.result.datum || record.datum } : {}),
        })
      }
    })
    // Datum-referenced controls display the datum's current fitted frame.
    const datums = new Map(active.filter((item) => item.operator === 'datum' && item.datum_label).map((item) => [item.datum_label, item.datum]))
    const relationships = active.map((item) => item.operator !== 'datum' && item.datum_label && datums.has(item.datum_label)
      ? { ...item, datum: datums.get(item.datum_label) } : item)
    const available = new Set(report.features.map((feature) => feature.id))
    const fitted = new Set(report.features.filter((feature) => feature.status === 'fitted').map((feature) => feature.id))
    return {
      analysis: state.analysis ? { ...state.analysis, prediction, report } : null,
      relationships, suspendedRelationships,
      distancePreview: null, editingMeasurementId: null,
      pinned: state.pinned.filter((id) => available.has(id)),
      draft: state.draft.every((id) => fitted.has(id)) ? state.draft : [],
      hovered: null,
    }
  }),
  addRelationship: (result) => set((state) => ({
    relationships: state.relationships.some((item) => measurementIdentity(item.operator, item.feature_refs, item) === measurementIdentity(result.operator, result.feature_refs, result))
      ? state.relationships
      : [result, ...state.relationships],
  })),
  removeRelationship: (id) => set((state) => {
    const removed = state.relationships.find((item) => item.id === id)
    const removedIds = new Set([id])
    if (removed?.operator === 'datum') {
      for (const item of state.relationships) {
        if ((item.datum_feature_ref || item.definition?.datum_feature_ref) === removed.feature_refs[0]) removedIds.add(item.id)
      }
    }
    return {
      editingMeasurementId: removedIds.has(state.editingMeasurementId || "") ? null : state.editingMeasurementId,
      distancePreview: null,
      relationships: state.relationships.filter((item) => !removedIds.has(item.id)),
      suspendedRelationships: state.suspendedRelationships.filter((item) => !removedIds.has(item.record.id)),
      hiddenRelationshipIds: state.hiddenRelationshipIds.filter((value) => !removedIds.has(value)),
    }
  }),
  toggleRelationshipVisibility: (id) => set((state) => ({
    hiddenRelationshipIds: state.hiddenRelationshipIds.includes(id)
      ? state.hiddenRelationshipIds.filter((value) => value !== id)
      : [...state.hiddenRelationshipIds, id],
  })),
  setDeformationFactor: (deformationFactor) => set({ deformationFactor }),
}))
