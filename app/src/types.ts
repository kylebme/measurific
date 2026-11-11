export type Point = [number, number]

export interface FeatureRecord {
  id: string
  type: string
  source?: 'model' | 'geometry' | 'adapter' | 'user'
  semantic?: string
  confidence?: number
  status: string
  support?: Point[][]
  closed?: boolean[]
  geometry?: Record<string, unknown>
  parent?: string
  label_level?: 'base' | 'compound'
  base_instance_ids?: string[]
  confidence_calibrated?: boolean
  model_status?: string
  reason?: string
  measurement_note?: string
  display_center?: Point
  ideal_support?: Point[][]
  deformation_max?: number
  adjacent_feature_ids?: string[]
}

export interface FeatureRelation {
  type: 'contour_adjacency' | 'compound_membership'
  feature_a: string
  feature_b: string
}

export interface DistanceDefinition {
  construction: 'normal_gap' | 'section' | 'closest_clearance' | 'center_distance' | 'center_to_line' | 'tangent_clearance'
  basis: 'observed' | 'fitted'
  extend: boolean
  statistic: 'minimum' | 'maximum' | 'average'
  // `direct` (the default) applies to point-to-point only; normal modes need a line.
  direction_mode?: 'direct' | 'normal_a' | 'normal_b' | 'x' | 'y' | 'custom'
  direction?: Point
  branches?: string[]
  station_range?: Point
  station?: number
  datum_feature_ref?: string
  datum_axis?: 'normal' | 'along'
  /** Point constructions: which point of each feature (in feature_refs order). */
  anchors?: DistanceAnchor[]
}
export type DistanceAnchor = 'center' | 'near' | 'far'

export interface MeasurementRecord {
  id: string
  operator: string
  feature_refs: string[]
  status: string
  value: number | number[] | null
  units?: string
  method?: string
  minimum?: number
  maximum?: number
  average?: number
  witness?: Point[]
  minimum_witness?: Point[]
  maximum_witness?: Point[]
  selected_statistic?: 'minimum' | 'maximum' | 'average'
  definition?: DistanceDefinition
  section_direction?: Point
  station_direction?: Point
  requested_span?: Point
  valid_intervals?: Point[]
  excluded_intervals?: Point[]
  coverage?: number
  crosses?: boolean
  signed?: boolean
  field_bands?: Point[][][]
  /** Axis-projected distances: dimension lines run along this direction. */
  dimension_direction?: Point
  /** Fitted geometry extended beyond the visible support to reach a witness. */
  virtual_segments?: Point[][]
  envelope_runs?: Point[][]
  axis_witness?: Point[]
  annotation_label?: string
  annotation_parent_id?: string
  annotation_metric?: string
  thread_metrics?: Record<string, MetricStatistics>
  /** Slot and tooth profile metrics, in the same form as thread metrics. */
  compound_metrics?: Record<string, MetricStatistics>
  datum_label?: string
  datum_feature_ref?: string
  datum?: { origin?: Point; direction?: Point; normal?: Point; type?: 'line' | 'center' }
  feature_anchors?: Record<string, FeatureAnchor>
  reason?: string
}

export interface MetricStatistics {
  status: string; minimum: number | null; maximum: number | null; average: number | null
  units: string; count: number; reason?: string; method?: string
  minimum_witness?: Point[]; maximum_witness?: Point[]
}

/** Locates a regenerated junction vertex whose index-based ID may change. */
export interface FeatureAnchor {
  adjacent_feature_ids: string[]
  point?: Point
}

export interface RemeasureRequest {
  id: string
  operator: string
  featureRefs: string[]
  params: Record<string, unknown>
  featureAnchors?: Record<string, FeatureAnchor>
}

export type RemeasureOutcome =
  | { id: string; status: 'measured'; result: MeasurementRecord; feature_map: Record<string, string> }
  | { id: string; status: 'unavailable'; reason: string }

export interface CornerExclusionResult extends Pick<Analysis, 'prediction' | 'report'> {
  measurements: RemeasureOutcome[]
}

export interface MeasurementSpec {
  label: string
  count: number
  accepts: string[]
}

export interface Analysis {
  path: string
  name: string
  imageUrl: string
  width: number
  height: number
  prediction: {
    schema_version: number
    label_version: string
    features: Array<Record<string, unknown>>
    compound_features: Array<Record<string, unknown>>
    relations: FeatureRelation[]
    threshold?: number
    recognition_backend?: string
    configuration?: Record<string, unknown>
    corner_exclusion_px?: number
  }
  report: {
    corner_exclusion?: {
      method: string
      radius_px: number
      corners: Array<{ path: number; nearest_vertex_index: number; arclength_px: number; observed_xy: Point }>
      excluded_vertex_indices: number[]
      excluded_vertex_count: number
    }
    features: FeatureRecord[]
    measurements: MeasurementRecord[]
    summary: { measured: number; failed_or_unavailable: number; tolerance_failures: number }
    extraction?: Record<string, unknown>
    relations?: FeatureRelation[]
    interaction?: {
      measurement_specs: Record<string, MeasurementSpec>
      deformation: { max_factor: number; maximum_observed_deviation: number }
    }
  }
  timingsMs: { edges: number; recognition: number; measurement: number }
}

export interface ExtractionFailure {
  path: string
  name: string
  imageUrl: string
  width: number
  height: number
  error: string
  diagnostic: {
    edges: Point[]
    guides: Point[][]
    issue: { label: string; bounds: [number, number, number, number] } | null
  }
}

export type GeometryOverrideTool = 'redefine_line' | 'redefine_arc' | 'redefine_circle'
export type CompoundOverrideTool = 'redefine_full_round_slot' | 'redefine_triangular_thread' |
  'redefine_trapezoidal_thread' | 'redefine_tooth_profile'

export interface GeometryOverrideResult {
  complete: boolean
  snapped_points: Point[]
  ring_index?: number
  feature_id?: string
  invalidated_feature_ids?: string[]
  report?: Analysis['report']
  prediction?: Analysis['prediction']
}

export interface CameraDevice {
  id: string
  vendor: string
  model: string
  serial: string
  protocol: string
}

export interface CameraNumericControl {
  available: boolean
  autoAvailable: boolean
  auto: boolean
  value: number
  min: number
  max: number
}

export interface CameraControls {
  exposure: CameraNumericControl
  gain: CameraNumericControl
}

export interface CameraFrameMeta {
  width: number
  height: number
  storageBits: 8 | 16
  sensorBits: number
  pixelFormat: string
  sequence: number
  timestampNs: number
  fullWidth?: number
  fullHeight?: number
  /** Sensor pixel of image pixel (x, y): (sampleStep·x + sampleOffsetX, sampleStep·y + sampleOffsetY). */
  sensorWidth?: number
  sensorHeight?: number
  channel?: CameraChannel
  sampleStep?: 1 | 2
  sampleOffsetX?: number
  sampleOffsetY?: number
  roi?: CameraRoi
  roiOffsetX?: number
  roiOffsetY?: number
}

export type CameraRoi =
  | { shape: 'rectangle'; x0: number; y0: number; x1: number; y1: number }
  | { shape: 'circle'; cx: number; cy: number; r: number }

export interface CameraAnalysisMask {
  shape: 'circle'
  cx: number
  cy: number
  r: number
}

export interface CameraFocus {
  score: number | null
  peak: number | null
  status: 'ok' | 'low-detail' | 'too-small' | 'unavailable'
  updatedAt: number
}

export interface CameraState {
  phase: 'idle' | 'starting' | 'streaming' | 'pausing' | 'paused' | 'error' | 'closed'
  devices: CameraDevice[]
  camera: (CameraDevice & {
    pixelFormat: string; sensorBits: number; width: number; height: number
    sensorWidth: number; sensorHeight: number; channel: CameraChannel; channels: CameraChannel[]
  }) | null
  controls: CameraControls | null
  error: string | null
  bridgePath?: string
}

export type ComputeDevice = 'cpu' | 'gpu'
/** PAE seeded by the coarse edge model, or the full coarse + fine deep learning edge model. */
export type EdgeMethod = 'pae' | 'deep-learning'
/** One Bayer color site at half resolution, or the camera's own grayscale output. */
export type CameraChannel = 'red' | 'green' | 'blue' | 'camera'

/** Recognition always uses the learned model bundled with the app. */
export interface Settings {
  schemaVersion: number
  computeDevice: ComputeDevice
  edgeMethod: EdgeMethod
  cameraChannel: CameraChannel
}

export interface Preferences {
  schemaVersion: number
  lastImageDirectory: string | null
  inspectorWidth: number
}

export interface PersistentStateView {
  settings: { values: Settings; effective: Settings; overridden: (keyof Settings)[]; path: string }
  preferences: { values: Preferences; path: string }
  calibration: {
    /** where new calibrations are saved: machine-wide when writable, else this user */
    scope: 'machine' | 'user'
    machineDirectory: string
    userDirectory: string
    directory: string
    errors: Partial<Record<'machine' | 'user', string>>
  }
  userDirectory: string
  warnings: string[]
}

/** Stored as `calibration.<id>`; the id is the save time as a dot-free ISO timestamp. */
export interface CalibrationRecord {
  id: string
  scope: 'machine' | 'user'
  writable: boolean
  savedAt: string
  savedBy: string
  appVersion: string
  [value: string]: unknown
}

export interface WorkerHealth {
  python: string
  platform: string
  executable: string
  device: ComputeDevice
  availableProviders: string[]
  model: { format: string; providers?: string[]; directory?: string; path?: string; source_checkpoint_sha256?: string } | null
  edgeModel?: { format: string; providers?: string[]; directory?: string; source_checkpoint_sha256?: string; pae_smoothing?: number }
  packages: Record<string, string>
  pendingRestart: boolean
  appVersion: string
}

export type MeasurementTool = 'distance_construction' | 'angle' | 'datum' | 'concentricity' |
  'perpendicularity' | 'parallelism' | 'circularity_ls' | 'straightness' | 'symmetry' | 'thread' |
  'least_squares_diameter' | 'radius' | 'length' | 'angular_span' | 'slot' | 'tooth_profile'
export type Tool = 'select' | MeasurementTool | GeometryOverrideTool | CompoundOverrideTool | 'remove_compound'

declare global {
  interface Window {
    inspection: {
      chooseImage(): Promise<string | null>
      analyze(path: string, validMask?: CameraAnalysisMask | null): Promise<Analysis | ExtractionFailure>
      measureRelationship(request: Record<string, unknown>): Promise<MeasurementRecord>
      setCornerExclusion(radius: number, measurements: RemeasureRequest[]): Promise<CornerExclusionResult>
      redefineGeometry(request: { type: 'line' | 'arc' | 'circle'; points: Point[]; cursorPath?: Point[] }): Promise<GeometryOverrideResult>
      redefineCompound(request: { semantic: string; featureIds?: string[]; points?: Point[]; cursorPaths?: Point[][] }): Promise<GeometryOverrideResult>
      removeCompound(request: { featureId: string }): Promise<GeometryOverrideResult>
      defaultSample(): Promise<string>
      cameraState(): Promise<CameraState>
      discoverCameras(): Promise<CameraDevice[]>
      startCamera(deviceId?: string): Promise<CameraState>
      setCameraRoi(roi: CameraRoi | null): Promise<void>
      resetCameraFocus(): Promise<void>
      pauseCamera(roi?: CameraRoi | null): Promise<{ snapshotPath: string; imageUrl: string; frame: CameraFrameMeta; analysisMask: CameraAnalysisMask | null }>
      setCameraControl(name: 'exposureAuto' | 'exposure' | 'gainAuto' | 'gain', value: boolean | number): Promise<CameraState>
      getState(): Promise<PersistentStateView>
      updateSettings(patch: Partial<Omit<Settings, 'schemaVersion'>>): Promise<PersistentStateView>
      updatePreferences(patch: Partial<Omit<Preferences, 'schemaVersion'>>): Promise<PersistentStateView>
      listCalibrations(): Promise<CalibrationRecord[]>
      getCalibration(id: string): Promise<CalibrationRecord | null>
      saveCalibration(calibration: Record<string, unknown>): Promise<CalibrationRecord>
      deleteCalibration(id: string): Promise<boolean>
      workerHealth(): Promise<WorkerHealth>
      openDataFolder(which: 'user' | 'calibration'): Promise<void>
      exportState(): Promise<string | null>
      importState(): Promise<{ added: string[]; conflicts: string[] } | null>
      onState(callback: (state: PersistentStateView) => void): () => void
      onCameraState(callback: (state: CameraState) => void): () => void
      onCameraFocus(callback: (focus: CameraFocus | null) => void): () => void
      onCameraFrame(callback: (meta: CameraFrameMeta, pixels: Uint8Array) => void): () => void
    }
  }
}
