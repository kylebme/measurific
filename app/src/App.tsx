import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import * as Tooltip from '@radix-ui/react-tooltip'
import {
  ChevronDown, Circle, CircleDotDashed, Cog, Crosshair, FolderOpen, Minus, TriangleRight,
  Eye, EyeOff, Pencil, LoaderCircle, MousePointer2, Pause, Ruler, Settings, Square, Trash2, Ungroup, X,
} from 'lucide-react'
import { CameraPanel } from './CameraPanel'
import { SettingsPanel } from './SettingsPanel'
import { DistanceInspector, DistanceResult } from './DistanceInspector'
import { constructionNames, shortFeatureId } from './distanceText'
import { annotationSymbols, isMetricTable } from './annotationLayout'
import { MeasurementEvidence } from './MeasurementEvidence'
import { MetricTable, isAnnotatedMeasurement, MeasurementAnnotations } from './MeasurementAnnotations'
import { allRelationships, measurementIdentity, remeasureRequest, useInspectionStore } from './store'
import type { CameraAnalysisMask, CameraRoi, CameraState, CompoundOverrideTool, FeatureRecord, MeasurementSpec, MeasurementTool, Point, Tool } from './types'

type RoiTool = 'rectangle' | 'circle' | null

const colors: Record<string, string> = {
  line: '#4ea1ff', point: '#ffb454', vertex: '#ffb454', arc: '#f37c9b', circle: '#ab8dff', arbitrary_profile: '#70e080', open_profile: '#70e080',
  full_round_slot: '#ff8d63', triangular_thread: '#f3d34a', trapezoidal_thread: '#45c6ef',
  flat: '#39d6b4', chamfer: '#f7bb50', fillet: '#f37c9b', shoulder: '#6eb5ff',
  hole: '#ab8dff', slot: '#ff8d63', groove: '#f56868', taper: '#45c6ef',
  thread: '#f3d34a', tooth_profile: '#e786ff', rounded_profile: '#70e080',
  external_profile: '#8a97a8', polygon: '#81d4a8', notch: '#ff836f', tab: '#56b8ff',
}

const geometryOverrideKinds = {
  redefine_line: 'line', redefine_arc: 'arc', redefine_circle: 'circle',
} as const

function isGeometryOverride(tool: Tool): tool is keyof typeof geometryOverrideKinds {
  return tool in geometryOverrideKinds
}

const compoundOverrideKinds: Record<CompoundOverrideTool, { semantic: string; label: string; hint: string }> = {
  redefine_full_round_slot: { semantic: 'full_round_slot', label: 'Full round slot', hint: 'Select every segment of one closed slot opening: both sides and both round ends.' },
  redefine_triangular_thread: { semantic: 'triangular_thread', label: 'Triangular thread', hint: 'Click where the thread starts and ends along one side, then do the same on the opposite side. Segments between the clicks become the thread.' },
  redefine_trapezoidal_thread: { semantic: 'trapezoidal_thread', label: 'Trapezoidal thread', hint: 'Click where the thread starts and ends along one side, then do the same on the opposite side. Segments between the clicks become the thread.' },
  redefine_tooth_profile: { semantic: 'tooth_profile', label: 'Tooth profile', hint: 'Select the line, arc, and profile segments of the teeth.' },
}

function isCompoundOverride(tool: Tool): tool is CompoundOverrideTool {
  return tool in compoundOverrideKinds
}

// Threads are picked like lines: start and end along one side, then the other side.
const sideCompoundTools = new Set<Tool>(['redefine_triangular_thread', 'redefine_trapezoidal_thread'])

function isSideCompound(tool: Tool): tool is CompoundOverrideTool {
  return sideCompoundTools.has(tool)
}

const sideSteps = ['Click the start of side A', 'Click the end of side A', 'Click the start of side B', 'Click the end of side B']

/** Tools whose clicks snap to the contour rather than select features. */
function isContourPicking(tool: Tool) {
  return isGeometryOverride(tool) || isSideCompound(tool)
}

/** Compact glyphs for compound classes, drawn on lucide's 24 px stroke grid. */
function CompoundGlyph({ kind }: { kind: 'slot' | 'triangular' | 'trapezoidal' }) {
  return <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {kind === 'slot' && <rect x="2.5" y="7.5" width="19" height="9" rx="4.5"/>}
    {kind === 'triangular' && <polyline points="2,16 6,8 10,16 14,8 18,16 22,8"/>}
    {kind === 'trapezoidal' && <polyline points="2,16 4,16 6.5,8 9.5,8 12,16 14,16 16.5,8 19.5,8 22,16"/>}
  </svg>
}

const fallbackSpecs: Record<MeasurementTool, MeasurementSpec> = {
  thread: { label: 'Thread measurements', count: 1, accepts: ['thread'] },
  distance_construction: { label: 'Distance', count: 2, accepts: ['any', 'any'] },
  angle: { label: 'Included angle', count: 2, accepts: ['line', 'line'] },
  datum: { label: 'Define datum', count: 1, accepts: ['datum'] },
  concentricity: { label: 'Concentricity', count: 2, accepts: ['center', 'center'] },
  perpendicularity: { label: 'Perpendicularity', count: 2, accepts: ['line', 'line'] },
  parallelism: { label: 'Parallelism', count: 2, accepts: ['line', 'line'] },
  circularity_ls: { label: 'Circularity', count: 1, accepts: ['circle'] },
  straightness: { label: 'Straightness', count: 1, accepts: ['line'] },
  symmetry: { label: 'Symmetry', count: 3, accepts: ['center', 'center', 'line'] },
  least_squares_diameter: { label: 'Diameter', count: 1, accepts: ['circle'] },
  radius: { label: 'Radius', count: 1, accepts: ['round'] },
  length: { label: 'Length', count: 1, accepts: ['line'] },
  angular_span: { label: 'Arc angle', count: 1, accepts: ['arc'] },
  slot: { label: 'Slot measurements', count: 1, accepts: ['slot'] },
  tooth_profile: { label: 'Tooth profile measurements', count: 1, accepts: ['tooth_profile'] },
}

const datumOperators = new Set<MeasurementTool>(['concentricity', 'perpendicularity', 'parallelism'])

function datumLabel(index: number) {
  let value = index + 1, label = ''
  while (value > 0) {
    value -= 1
    label = String.fromCharCode(65 + value % 26) + label
    value = Math.floor(value / 26)
  }
  return label
}

function nextDatumLabel(measurements: { operator: string; datum_label?: string }[]) {
  const used = new Set(measurements.filter((m) => m.operator === 'datum').map((m) => m.datum_label).filter(Boolean))
  let index = 0
  while (used.has(datumLabel(index))) index += 1
  return datumLabel(index)
}

function formatValue(value: unknown, units = '') {
  if (typeof value === 'number') return `${value.toFixed(value >= 100 ? 1 : 3)}${units ? ` ${units}` : ''}`
  if (Array.isArray(value)) return `[${value.map((v) => typeof v === 'number' ? v.toFixed(2) : v).join(', ')}] ${units}`
  return '—'
}

function ResultValue({ value, units = '' }: { value: unknown; units?: string }) {
  if (typeof value !== 'number') return <span className="result-value">{formatValue(value, units)}</span>
  return <span className="result-value">{value.toFixed(value >= 100 ? 1 : 3)}{units && <small>{units}</small>}</span>
}

const resultSymbols: Record<string, string> = {
  thread: '≋', angle: '∠', symmetry: '⌯', least_squares_diameter: 'Ø', radius: 'R', length: '↔', angular_span: '⌒',
}

/** Compound kinds a compound measurement tool accepts, with the message for any other click. */
const compoundMeasurementTargets: Partial<Record<MeasurementTool, { accepts: (feature: FeatureRecord) => boolean; message: string }>> = {
  thread: { accepts: (f) => f.type === 'repeated_profile' && ['thread', 'triangular_thread', 'trapezoidal_thread', 'square_thread'].includes(f.semantic || ''),
    message: 'Select a triangular or trapezoidal/square thread compound.' },
  slot: { accepts: (f) => f.type === 'slot', message: 'Select a full round slot compound.' },
  tooth_profile: { accepts: (f) => f.type === 'repeated_profile' && f.semantic === 'tooth_profile', message: 'Select a tooth profile compound.' },
}

function ResultSymbol({ result }: { result: { operator: string; datum_label?: string } }) {
  if (result.operator === 'datum') return <span className="datum-tool-glyph">{result.datum_label || '?'}</span>
  if (result.operator === 'distance_construction') return <Ruler size={15}/>
  if (result.operator === 'slot') return <CompoundGlyph kind="slot"/>
  if (result.operator === 'tooth_profile') return <Cog size={15}/>
  return <>{resultSymbols[result.operator] || annotationSymbols[result.operator] || ''}</>
}

function featureName(feature: FeatureRecord) {
  return (feature.semantic || feature.type || 'feature').replaceAll('_', ' ')
}

function featureCenter(feature: FeatureRecord): Point {
  if (feature.display_center) return feature.display_center
  const center = feature.geometry?.center
  if (Array.isArray(center) && center.length === 2) return center as Point
  return [0, 0]
}

function displayedSupport(feature: FeatureRecord, factor: number) {
  if (factor === 1 || !feature.ideal_support) return feature.support || []
  return (feature.support || []).map((run, runIndex) => run.map((point, pointIndex) => {
    const ideal = feature.ideal_support?.[runIndex]?.[pointIndex]
    return ideal ? [ideal[0] + factor * (point[0] - ideal[0]), ideal[1] + factor * (point[1] - ideal[1])] as Point : point
  }))
}

function featurePath(feature: FeatureRecord, factor = 1) {
  return displayedSupport(feature, factor).map((run, index) => {
    if (!run.length) return ''
    const closed = feature.closed?.[index]
    return `M ${run.map((p) => `${p[0].toFixed(2)} ${p[1].toFixed(2)}`).join(' L ')}${closed ? ' Z' : ''}`
  }).join(' ')
}

function featureColor(feature: FeatureRecord) {
  const kind = feature.label_level === 'compound' ? (feature.semantic || feature.type) : feature.type
  return colors[kind] || colors[feature.semantic || ''] || '#57c7ff'
}

function FeatureTooltip({ feature }: { feature: FeatureRecord }) {
  return <div className="feature-tooltip">
    <div className="tooltip-title"><span>{featureName(feature)}</span><span className="mono">{shortFeatureId(feature.id)}</span></div>
    <div className="tooltip-meta">{feature.label_level || feature.type} · {feature.status} · {feature.confidence == null ? (feature.source || 'user') : `${Math.round(feature.confidence * 100)}% score (uncalibrated)`}</div>
    {feature.reason && <div className="tooltip-empty">{feature.reason}</div>}
    {feature.measurement_note && <div className="tooltip-empty">{feature.measurement_note}</div>}
    {feature.base_instance_ids && <div className="tooltip-meta">{feature.base_instance_ids.length} constituent primitives</div>}
  </div>
}

function CameraView({ roiTool, roi, onRoiChange }: { roiTool: RoiTool; roi: CameraRoi | null; onRoiChange: (roi: CameraRoi) => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [frameSize, setFrameSize] = useState({ width: 1, height: 1 })
  const [draft, setDraft] = useState<{ start: Point; current: Point } | null>(null)
  useEffect(() => {
    const canvas = canvasRef.current
    const gl = canvas?.getContext('webgl2', { alpha: false, antialias: false })
    if (!canvas || !gl) return
    const compile = (type: number, source: string) => {
      const shader = gl.createShader(type)!
      gl.shaderSource(shader, source); gl.compileShader(shader)
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader) || 'Camera shader failed')
      return shader
    }
    const program = gl.createProgram()!
    gl.attachShader(program, compile(gl.VERTEX_SHADER, `#version 300 es
      in vec2 position; out vec2 uv;
      void main() { gl_Position=vec4(position,0.,1.); uv=vec2((position.x+1.)*.5,(1.-position.y)*.5); }`))
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, `#version 300 es
      precision highp float; uniform sampler2D frame; in vec2 uv; out vec4 color;
      void main() { float y=texture(frame,uv).r; color=vec4(y,y,y,1.); }`))
    gl.linkProgram(program); gl.useProgram(program)
    const vertices = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, vertices)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW)
    const location = gl.getAttribLocation(program, 'position')
    gl.enableVertexAttribArray(location); gl.vertexAttribPointer(location, 2, gl.FLOAT, false, 0, 0)
    const texture = gl.createTexture()
    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1)
    const stop = window.inspection.onCameraFrame((meta, pixels) => {
      setFrameSize((current) => current.width === meta.width && current.height === meta.height ? current : { width: meta.width, height: meta.height })
      const scale = Math.min(window.devicePixelRatio || 1, 2)
      const targetWidth = Math.max(1, Math.round(canvas.clientWidth * scale))
      const targetHeight = Math.max(1, Math.round(canvas.clientHeight * scale))
      if (canvas.width !== targetWidth || canvas.height !== targetHeight) { canvas.width = targetWidth; canvas.height = targetHeight }
      gl.viewport(0, 0, canvas.width, canvas.height); gl.clearColor(.03, .04, .05, 1); gl.clear(gl.COLOR_BUFFER_BIT)
      const ratio = Math.min(canvas.width / meta.width, canvas.height / meta.height)
      const width = Math.round(meta.width * ratio), height = Math.round(meta.height * ratio)
      gl.viewport(Math.floor((canvas.width - width) / 2), Math.floor((canvas.height - height) / 2), width, height)
      gl.bindTexture(gl.TEXTURE_2D, texture)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, meta.width, meta.height, 0, gl.RED, gl.UNSIGNED_BYTE, pixels)
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
    })
    return () => { stop(); gl.deleteTexture(texture); gl.deleteBuffer(vertices); gl.deleteProgram(program) }
  }, [])
  const imagePoint = (event: React.PointerEvent<SVGSVGElement>): Point | null => {
    const bounds = event.currentTarget.getBoundingClientRect()
    const scale = Math.min(bounds.width / frameSize.width, bounds.height / frameSize.height)
    const left = bounds.left + (bounds.width - frameSize.width * scale) / 2
    const top = bounds.top + (bounds.height - frameSize.height * scale) / 2
    const x = (event.clientX - left) / scale, y = (event.clientY - top) / scale
    if (x < 0 || y < 0 || x > frameSize.width || y > frameSize.height) return null
    return [x, y]
  }
  const roiFromPoints = (start: Point, current: Point): CameraRoi | null => {
    if (roiTool === 'rectangle') {
      if (Math.abs(current[0] - start[0]) < 8 || Math.abs(current[1] - start[1]) < 8) return null
      return {
        shape: 'rectangle',
        x0: Math.min(start[0], current[0]) / frameSize.width,
        y0: Math.min(start[1], current[1]) / frameSize.height,
        x1: Math.max(start[0], current[0]) / frameSize.width,
        y1: Math.max(start[1], current[1]) / frameSize.height,
      }
    }
    if (roiTool === 'circle') {
      const radius = Math.hypot(current[0] - start[0], current[1] - start[1])
      const contained = Math.min(radius, start[0], frameSize.width - start[0], start[1], frameSize.height - start[1])
      if (contained < 8) return null
      return { shape: 'circle', cx: start[0] / frameSize.width, cy: start[1] / frameSize.height, r: contained / Math.min(frameSize.width, frameSize.height) }
    }
    return null
  }
  const activeRoi = draft ? roiFromPoints(draft.start, draft.current) : roi
  const roiGeometry = (value: CameraRoi) => value.shape === 'rectangle'
    ? { kind: 'rect' as const, x: value.x0 * frameSize.width, y: value.y0 * frameSize.height,
        width: (value.x1 - value.x0) * frameSize.width, height: (value.y1 - value.y0) * frameSize.height }
    : { kind: 'circle' as const, cx: value.cx * frameSize.width, cy: value.cy * frameSize.height,
        r: value.r * Math.min(frameSize.width, frameSize.height) }
  const geometry = activeRoi ? roiGeometry(activeRoi) : null
  return <div className={`camera-canvas ${roiTool ? 'drawing-roi' : ''}`}>
    <canvas className="camera-view" ref={canvasRef}/>
    <svg className="roi-overlay" viewBox={`0 0 ${frameSize.width} ${frameSize.height}`} preserveAspectRatio="xMidYMid meet"
      onPointerDown={(event) => {
        if (!roiTool) return
        const point = imagePoint(event)
        if (!point) return
        event.currentTarget.setPointerCapture(event.pointerId)
        setDraft({ start: point, current: point })
      }}
      onPointerMove={(event) => {
        if (!draft) return
        const point = imagePoint(event)
        if (point) setDraft({ ...draft, current: point })
      }}
      onPointerUp={(event) => {
        if (!draft) return
        const point = imagePoint(event) || draft.current
        const completed = roiFromPoints(draft.start, point)
        setDraft(null)
        if (completed) onRoiChange(completed)
      }}>
      {geometry && <>
        <defs><mask id="roi-cutout"><rect width={frameSize.width} height={frameSize.height} fill="white"/>
          {geometry.kind === 'rect'
            ? <rect x={geometry.x} y={geometry.y} width={geometry.width} height={geometry.height} fill="black"/>
            : <circle cx={geometry.cx} cy={geometry.cy} r={geometry.r} fill="black"/>}
        </mask></defs>
        <rect className="roi-shade" width={frameSize.width} height={frameSize.height} mask="url(#roi-cutout)"/>
        {geometry.kind === 'rect'
          ? <rect className="roi-outline" x={geometry.x} y={geometry.y} width={geometry.width} height={geometry.height}/>
          : <circle className="roi-outline" cx={geometry.cx} cy={geometry.cy} r={geometry.r}/>} 
      </>}
    </svg>
    {roiTool && <div className="roi-instruction">Drag to set a {roiTool === 'rectangle' ? 'rectangular' : 'circular'} ROI</div>}
  </div>
}

function ImageStage({ onFeatureClick, onGeometryClick, controlDown, cameraLive, pausedImageUrl, roiTool, cameraRoi, onRoiChange }: { onFeatureClick: (feature: FeatureRecord) => void; onGeometryClick: (point: Point, cursorPath: Point[]) => void; controlDown: boolean; cameraLive: boolean; pausedImageUrl: string | null; roiTool: RoiTool; cameraRoi: CameraRoi | null; onRoiChange: (roi: CameraRoi) => void }) {
  const { analysis, extractionFailure, hovered, pinned, draft, overrideDraft, relationships, distancePreview, editingMeasurementId, hiddenRelationshipIds, toggleRelationshipVisibility, setHovered, deformationFactor, tool } = useInspectionStore()
  const displayedMeasurements = useMemo(() => [...relationships.filter(m => !distancePreview || m.id !== editingMeasurementId), ...(distancePreview ? [distancePreview] : [])], [relationships, distancePreview, editingMeasurementId])
  const stageRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const [pointer, setPointer] = useState({ x: 0, y: 0 })
  const [imagePointer, setImagePointer] = useState<Point | null>(null)
  const [view, setView] = useState({ scale: 1, x: 0, y: 0 })
  // The overlay renders crisply at overlayScale; while zooming, the compositor
  // only stretches it by view.scale / overlayScale, so gestures never re-raster
  // vectors. It re-renders once zooming pauses, or early when zooming well out
  // so the stretched layer never needs more than a few screens of tiles.
  const [overlayScale, setOverlayScale] = useState(1)
  if (view.scale < overlayScale / 2) setOverlayScale(view.scale)
  useEffect(() => {
    if (view.scale === overlayScale) return
    const timer = setTimeout(() => setOverlayScale(view.scale), 150)
    return () => clearTimeout(timer)
  }, [view.scale, overlayScale])
  const panRef = useRef<{ x: number; y: number; viewX: number; viewY: number } | null>(null)
  const cursorPathRef = useRef<Point[]>([])
  const tracingGeometryRef = useRef(false)
  useEffect(() => setView({ scale: 1, x: 0, y: 0 }), [analysis?.path])
  useEffect(() => {
    cursorPathRef.current = []
    tracingGeometryRef.current = false
  }, [tool])
  useEffect(() => {
    if (!overrideDraft.length) {
      cursorPathRef.current = []
      tracingGeometryRef.current = false
    }
  }, [overrideDraft.length])
  useEffect(() => {
    const cancelTrace = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        cursorPathRef.current = []
        tracingGeometryRef.current = false
      }
    }
    window.addEventListener('keydown', cancelTrace)
    return () => window.removeEventListener('keydown', cancelTrace)
  }, [])
  if (cameraLive) return <div className="image-stage live-stage"><CameraView roiTool={roiTool} roi={cameraRoi} onRoiChange={onRoiChange}/><div className="live-indicator"><span/>LIVE</div></div>
  if (extractionFailure) {
    const { diagnostic } = extractionFailure
    const edgePath = diagnostic.edges.map(([x, y]) => `M${x.toFixed(1)} ${y.toFixed(1)}h.01`).join('')
    return <div className="image-stage diagnostic-stage">
      <img src={extractionFailure.imageUrl} alt={extractionFailure.name} draggable={false}/>
      <svg className="diagnostic-overlay" viewBox={`-0.5 -0.5 ${extractionFailure.width} ${extractionFailure.height}`} preserveAspectRatio="xMidYMid meet">
        {diagnostic.guides.map((guide, index) => <polyline key={index} className="diagnostic-guide" points={guide.map(point => point.join(',')).join(' ')}/>) }
        <path className="diagnostic-edges" d={edgePath}/>
        {diagnostic.issue && <rect className="diagnostic-issue" x={diagnostic.issue.bounds[0]} y={diagnostic.issue.bounds[1]}
          width={diagnostic.issue.bounds[2]-diagnostic.issue.bounds[0]} height={diagnostic.issue.bounds[3]-diagnostic.issue.bounds[1]}/>}
      </svg>
      <div className="diagnostic-legend"><strong>Extraction diagnostic</strong><span><i className="edge-swatch"/>Edge-model contour points</span><span><i className="guide-swatch"/>Contour guide</span>{diagnostic.issue && <span><i className="issue-swatch"/>{diagnostic.issue.label}</span>}</div>
    </div>
  }
  if (!analysis && pausedImageUrl) return <div className="image-stage paused-camera-stage"><img src={pausedImageUrl} alt="Paused camera frame"/></div>
  if (!analysis) return <div className="empty-stage"><Crosshair size={42}/><h2>No image loaded</h2><p>Open an image to run edge detection and primitive and compound recognition.</p></div>

  const supportedFeatures = analysis.report.features.filter((feature) => feature.support?.length)
  const compoundFeatures = supportedFeatures.filter((feature) => feature.label_level === 'compound')
  const compoundMemberIds = new Set(compoundFeatures.flatMap((feature) => feature.base_instance_ids || []))
  const baseFeatures = supportedFeatures.filter((feature) => (feature.label_level || 'base') === 'base')
  // Canvas rendering is compound-priority regardless of which inspector list is
  // open. Ctrl is the sole temporary escape hatch to constituent geometry.
  const features = isCompoundOverride(tool) && !isSideCompound(tool)
    ? baseFeatures.filter((feature) => feature.type !== 'point')
    : controlDown
      ? baseFeatures
      : [...baseFeatures.filter((feature) => !compoundMemberIds.has(feature.id)), ...compoundFeatures]
  const positionFor = (feature: FeatureRecord) => {
    const bounds = stageRef.current?.getBoundingClientRect()
    if (!bounds) return { left: 0, top: 0 }
    const ratio = analysis.width / analysis.height
    let width = bounds.width, height = width / ratio, ox = 0, oy = (bounds.height - height) / 2
    if (height > bounds.height) { height = bounds.height; width = height * ratio; oy = 0; ox = (bounds.width - width) / 2 }
    const [x, y] = featureCenter(feature)
    const left = (ox + (x + .5) / analysis.width * width) * view.scale + view.x
    const top = (oy + (y + .5) / analysis.height * height) * view.scale + view.y
    return {
      left: Math.max(126, Math.min(bounds.width - 126, left)),
      top: Math.max(20, Math.min(bounds.height - 20, top)),
    }
  }
  const hoveredFeature = features.find((f) => f.id === hovered)
  const viewIsOffset = view.scale > 1.001 || Math.abs(view.x) > .5 || Math.abs(view.y) > .5

  const zoomAt = (clientX: number, clientY: number, multiplier: number) => {
    const bounds = stageRef.current?.getBoundingClientRect()
    if (!bounds) return
    const px = clientX - bounds.left, py = clientY - bounds.top
    setView((old) => {
      const scale = Math.max(1, Math.min(20, old.scale * multiplier))
      const ratio = scale / old.scale
      return { scale, x: px - (px - old.x) * ratio, y: py - (py - old.y) * ratio }
    })
  }
  const imagePointAt = (clientX: number, clientY: number): Point | null => {
    const svg = svgRef.current
    // The pixel-centered viewBox makes this inverse return detector coordinates.
    const matrix = svg?.getScreenCTM()?.inverse()
    if (!svg || !matrix) return null
    const point = new DOMPoint(clientX, clientY).matrixTransform(matrix)
    return [point.x, point.y]
  }

  return <div className={`image-stage ${panRef.current ? 'panning' : ''} ${isContourPicking(tool) ? 'redefining' : ''}`} ref={stageRef}
    onWheel={(event) => {
      event.preventDefault()
      const mouseWheel = event.deltaMode !== WheelEvent.DOM_DELTA_PIXEL || (!event.ctrlKey && event.deltaX === 0 && Math.abs(event.deltaY) >= 40)
      if (event.ctrlKey || mouseWheel) zoomAt(event.clientX, event.clientY, Math.exp(-event.deltaY * (event.ctrlKey ? .008 : .0025)))
      else setView((old) => ({ ...old, x: old.x - event.deltaX, y: old.y - event.deltaY }))
    }}
    onDoubleClick={() => setView({ scale: 1, x: 0, y: 0 })}
    onPointerDown={(event) => {
      if (event.button !== 1) return
      event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId)
      panRef.current = { x: event.clientX, y: event.clientY, viewX: view.x, viewY: view.y }
    }}
    onPointerUp={(event) => { if (panRef.current) event.currentTarget.releasePointerCapture(event.pointerId); panRef.current = null }}
    onPointerCancel={() => { panRef.current = null }}
    onPointerMove={(event) => {
    if (panRef.current) {
      const pan = panRef.current
      setView((old) => ({ ...old, x: pan.viewX + event.clientX - pan.x, y: pan.viewY + event.clientY - pan.y }))
      return
    }
    const bounds = stageRef.current?.getBoundingClientRect()
    if (bounds) setPointer({ x: event.clientX - bounds.left + 14, y: event.clientY - bounds.top + 14 })
    const events = event.nativeEvent.getCoalescedEvents?.() || [event.nativeEvent]
    const imagePoints = events.map((sample) => imagePointAt(sample.clientX, sample.clientY)).filter(Boolean) as Point[]
    const imagePoint = imagePoints.at(-1) || null
    setImagePointer(imagePoint)
    if (tracingGeometryRef.current && (tool === 'redefine_line' || tool === 'redefine_arc' || isSideCompound(tool))) {
      cursorPathRef.current.push(...imagePoints)
      if (cursorPathRef.current.length > 4096) cursorPathRef.current.splice(0, cursorPathRef.current.length-4096)
    }
  }}>
    {deformationFactor !== 1 && <div className="deformation-warning">DEFORMATION DISPLAY ×{deformationFactor.toFixed(deformationFactor < 10 ? 1 : 0)} — MEASUREMENTS UNCHANGED</div>}
    {controlDown && compoundFeatures.length > 0 && <div className="control-layer-note">Constituent primitives · release Ctrl to restore compounds</div>}
    {viewIsOffset && <button className="reset-view-button" onClick={() => setView({ scale: 1, x: 0, y: 0 })}>Reset View</button>}
    <div className="stage-content" style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})` }}>
    <img src={analysis.imageUrl} alt={analysis.name} draggable={false}/>
    </div>
    {/* The overlay stays out of the image layer so it can re-raster at the
        current zoom (see overlayScale) with screen-width non-scaling strokes. */}
    {/* Detector coordinates use integer pixel centers; image bounds start at -0.5. */}
    <svg ref={svgRef} className="feature-overlay" viewBox={`-0.5 -0.5 ${analysis.width} ${analysis.height}`} preserveAspectRatio="xMidYMid meet"
      style={{ width: `${overlayScale * 100}%`, height: `${overlayScale * 100}%`, transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale / overlayScale})` }}
      onClick={(event) => {
        if (!isContourPicking(tool)) return
        const point = imagePointAt(event.clientX, event.clientY)
        if (!point) return
        // A trace starts at each interval's first click: the line/arc start, or either side's start.
        const startsInterval = isSideCompound(tool) ? overrideDraft.length % 2 === 0 : (tool === 'redefine_line' || tool === 'redefine_arc') && !overrideDraft.length
        if (startsInterval) {
          cursorPathRef.current = [point]
          tracingGeometryRef.current = true
          onGeometryClick(point, [])
        } else {
          cursorPathRef.current.push(point)
          tracingGeometryRef.current = false
          onGeometryClick(point, [...cursorPathRef.current])
        }
      }}>
      {relationships.filter((measurement) => !hiddenRelationshipIds.includes(measurement.id)).flatMap((measurement) => {
        if (measurement.operator === 'distance_construction' || datumOperators.has(measurement.operator as MeasurementTool)) return []
        const witness = measurement.minimum_witness || measurement.witness
        if (!witness || witness.length !== 2) return []
        return <g key={measurement.id} className="witness">
          <line x1={witness[0][0]} y1={witness[0][1]} x2={witness[1][0]} y2={witness[1][1]}/>
          <circle cx={witness[0][0]} cy={witness[0][1]} r="4"/><circle cx={witness[1][0]} cy={witness[1][1]} r="4"/>
        </g>
      })}
      <MeasurementEvidence features={analysis.report.features} measurements={displayedMeasurements.filter(m => !hiddenRelationshipIds.includes(m.id) || m.id === distancePreview?.id)} previewId={distancePreview?.id}/>
      {features.map((feature) => {
        const active = hovered === feature.id || pinned.includes(feature.id) || (distancePreview?.feature_refs || draft).includes(feature.id)
        const color = featureColor(feature)
        // Removal acts only on a hovered compound; other contours stay inert.
        const interactive = tool !== 'remove_compound' || feature.label_level === 'compound'
        if (feature.type === 'point') {
          const [x, y] = featureCenter(feature)
          return <g key={feature.id} className={`feature vertex-feature ${active ? 'active' : ''} ${hovered === feature.id ? 'hovered' : ''}`}>
            {interactive && <circle className="vertex-hit" cx={x} cy={y} r={10 / overlayScale} onPointerEnter={() => setHovered(feature.id)} onPointerLeave={() => setHovered(null)} onClick={() => onFeatureClick(feature)}/>}
            <circle className="vertex-visible" cx={x} cy={y} r={4 / overlayScale} style={{ stroke: color }}/>
          </g>
        }
        return <g key={feature.id} className={`feature ${active ? 'active' : ''} ${feature.status !== 'fitted' ? 'unavailable' : ''}`}>
          {interactive && <path className="hit-path" d={featurePath(feature, deformationFactor)} onPointerEnter={() => setHovered(feature.id)} onPointerLeave={() => setHovered(null)} onClick={() => onFeatureClick(feature)}/>}
          <path className="visible-path" d={featurePath(feature, deformationFactor)} style={{ stroke: color, color }}/>
        </g>
      })}
      {overrideDraft[0] && <g className="geometry-draft">
        {tool === 'redefine_line' && imagePointer && <line x1={overrideDraft[0][0]} y1={overrideDraft[0][1]} x2={imagePointer[0]} y2={imagePointer[1]}/>}
        {(isSideCompound(tool) ? overrideDraft : overrideDraft.slice(0, 1)).map((point, index) =>
          <circle key={index} cx={point[0]} cy={point[1]} r={4 / overlayScale}/>)}
      </g>}
    </svg>
    <MeasurementAnnotations stageRef={stageRef} imageWidth={analysis.width} imageHeight={analysis.height}
      features={analysis.report.features} measurements={displayedMeasurements} hiddenIds={distancePreview ? hiddenRelationshipIds.filter(id => id !== distancePreview.id) : hiddenRelationshipIds}
      view={view} deformationFactor={deformationFactor} onHide={toggleRelationshipVisibility}/>
    {hoveredFeature && !pinned.includes(hoveredFeature.id) && <div className="floating-tooltip" style={{ left: pointer.x, top: pointer.y }}>
      <FeatureTooltip feature={hoveredFeature}/>
    </div>}
    {pinned.map((id) => {
      const feature = supportedFeatures.find((f) => f.id === id)
      if (!feature) return null
      return <div key={id} className="floating-tooltip pinned-tooltip" style={positionFor(feature)}>
        <FeatureTooltip feature={feature}/>
      </div>
    })}
  </div>
}

const inspectorWidthRange = { min: 260, max: 720, initial: 336 }
const clampInspectorWidth = (width: number) => Math.round(Math.max(inspectorWidthRange.min, Math.min(inspectorWidthRange.max, width)))

/** Drag handle on the inspector's left edge; double-click restores the default width. */
function PanelResizer({ width, onChange, onCommit }: { width: number; onChange: (width: number) => void; onCommit: (width: number) => void }) {
  const drag = useRef<{ x: number; width: number } | null>(null)
  const [dragging, setDragging] = useState(false)
  const widthAt = (clientX: number) => clampInspectorWidth(drag.current!.width - (clientX - drag.current!.x))
  const end = (event: React.PointerEvent) => {
    if (!drag.current) return
    const next = widthAt(event.clientX)
    drag.current = null; setDragging(false)
    onChange(next); onCommit(next)
  }
  const move = (event: React.PointerEvent) => { if (drag.current) onChange(widthAt(event.clientX)) }
  return <div className={`panel-resizer ${dragging ? 'dragging' : ''}`} role="separator" aria-orientation="vertical" aria-label="Resize side panel"
    aria-valuenow={width} aria-valuemin={inspectorWidthRange.min} aria-valuemax={inspectorWidthRange.max} tabIndex={0}
    onPointerDown={(event) => {
      if (event.button !== 0) return
      event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId)
      drag.current = { x: event.clientX, width }; setDragging(true)
    }}
    onPointerMove={move} onPointerUp={end} onPointerCancel={end}
    onDoubleClick={() => { onChange(inspectorWidthRange.initial); onCommit(inspectorWidthRange.initial) }}
    onKeyDown={(event) => {
      const step = event.shiftKey ? 64 : 16
      const delta = event.key === 'ArrowLeft' ? step : event.key === 'ArrowRight' ? -step : 0
      if (!delta) return
      event.preventDefault()
      const next = clampInspectorWidth(width + delta)
      onChange(next); onCommit(next)
    }}>
    {/* While dragging, a window-wide layer keeps the image from hovering; its pointer events bubble here if capture is lost. */}
    {dragging && <div className="panel-resize-overlay"/>}
  </div>
}

function ToolGroup({ label, children }: { label: string; children: React.ReactNode }) {
  return <div className="tool-group" role="group" aria-label={label}>
    <div className="tool-group-buttons">{children}</div>
  </div>
}

function ToolButton({ tool, active, icon, label, onClick, disabled = false }: { tool: string; active: boolean; icon: React.ReactNode; label: string; onClick: () => void; disabled?: boolean }) {
  return <Tooltip.Root><Tooltip.Trigger asChild>
    <button className={`tool-button ${active ? 'active' : ''}`} onClick={onClick} aria-label={label} data-tool={tool} disabled={disabled}>{icon}</button>
  </Tooltip.Trigger><Tooltip.Portal><Tooltip.Content className="radix-tooltip" side="bottom">{label}<Tooltip.Arrow className="tooltip-arrow"/></Tooltip.Content></Tooltip.Portal></Tooltip.Root>
}

function InspectorPanel({ runRelationship, updateCornerExclusion, applyCompound }: {
  runRelationship: (params?: Record<string, unknown>) => Promise<void>
  updateCornerExclusion: (radius: number) => Promise<void>
  applyCompound: () => Promise<void>
}) {
  const { analysis, tool, draft, overrideDraft, relationships, suspendedRelationships, hiddenRelationshipIds, toggleRelationshipVisibility, removeRelationship, error, deformationFactor, setDeformationFactor, setTool, editMeasurement } = useInspectionStore()
  const geometryOverride = isGeometryOverride(tool)
  const compoundOverride = isCompoundOverride(tool) ? compoundOverrideKinds[tool] : null
  const selectedFeatures = draft.map((id) => analysis?.report.features.find((f) => f.id === id)).filter(Boolean) as FeatureRecord[]
  const specs = (analysis?.report.interaction?.measurement_specs || fallbackSpecs) as Record<string, MeasurementSpec>
  const spec = tool === 'select' || tool === 'remove_compound' || isGeometryOverride(tool) || isCompoundOverride(tool) ? null : (specs[tool] || fallbackSpecs[tool])
  const deformationMax = analysis?.report.interaction?.deformation.max_factor || 1
  const appliedCornerExclusion = analysis?.report.corner_exclusion?.radius_px ?? 1
  const [cornerExclusion, setCornerExclusion] = useState(appliedCornerExclusion)
  const selectedDatumLabel = relationships.find((result) => result.operator === 'datum' && result.feature_refs[0] === draft[0])?.datum_label

  useEffect(() => setCornerExclusion(appliedCornerExclusion), [appliedCornerExclusion, analysis?.path])

  const duplicateSelection = Boolean(spec && draft.length === spec.count && relationships.some((result) =>
    measurementIdentity(result.operator, result.feature_refs, result) === measurementIdentity(tool, draft),
  ))
  useEffect(() => {
    if (!compoundOverride || isSideCompound(tool)) return
    const onEnter = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      if (event.key !== 'Enter' || event.repeat || !draft.length || target?.tagName === 'INPUT' || target?.tagName === 'SELECT') return
      void applyCompound()
    }
    window.addEventListener('keydown', onEnter)
    return () => window.removeEventListener('keydown', onEnter)
  }, [applyCompound, compoundOverride, draft.length, tool])
  useEffect(() => {
    const onEnter = (event: KeyboardEvent) => {
      if (tool === 'distance_construction' || event.key !== 'Enter' || event.repeat || tool === 'select' || geometryOverride || duplicateSelection || spec?.count === 1 || draft.length !== spec?.count) return
      const target = event.target as HTMLElement | null
      if (target?.tagName === 'INPUT' || target?.tagName === 'SELECT') return
      void runRelationship()
    }
    window.addEventListener('keydown', onEnter)
    return () => window.removeEventListener('keydown', onEnter)
  }, [draft.length, duplicateSelection, geometryOverride, runRelationship, spec?.count, tool])

  return <aside className="inspector-panel">
    <section className="panel-section relationship-builder">
      <label className="tool-heading">
      <select className="tool-select" aria-label="Measurement tool" value={tool} onChange={(event) => setTool(event.target.value as Tool)}>
        <option value="select">Choose a tool</option>
        <optgroup label="Geometry override">
          <option value="redefine_line">Redefine as line</option><option value="redefine_arc">Redefine as arc</option><option value="redefine_circle">Redefine as circle</option>
          {Object.entries(compoundOverrideKinds).map(([value, kind]) => <option key={value} value={value}>Redefine as {kind.label.toLowerCase()}</option>)}
          <option value="remove_compound">Remove compound</option>
        </optgroup>
        <optgroup label="General">
          <option value="distance_construction">Distance</option><option value="angle">Included angle</option>
        </optgroup>
        <optgroup label="Size">
          <option value="least_squares_diameter">Diameter</option><option value="radius">Radius</option>
          <option value="length">Length</option><option value="angular_span">Arc angle</option>
        </optgroup>
        <optgroup label="Compounds">
          <option value="thread">Thread measurements</option><option value="slot">Slot measurements</option><option value="tooth_profile">Tooth profile measurements</option>
        </optgroup>
        <optgroup label="Planar GD&amp;T">
          <option value="datum">Define datum</option><option value="concentricity">Concentricity</option>
          <option value="perpendicularity">Perpendicularity</option><option value="parallelism">Parallelism</option>
          <option value="circularity_ls">Circularity</option><option value="straightness">Straightness</option><option value="symmetry">Symmetry</option>
        </optgroup>
      </select><ChevronDown size={15}/>
      </label>
      {tool === 'distance_construction' ? <DistanceInspector/> : compoundOverride && isSideCompound(tool) ? <>
        <p className="hint">{compoundOverride.hint} Short segments between the clicks are included automatically. Press Esc to start over.</p>
        <div className="selection-slots">
          {sideSteps.map((step, index) => <div className={index < overrideDraft.length ? 'selection-slot filled' : 'selection-slot'} key={step}>
            <span>{index < 2 ? 'A' : 'B'}</span><div>{index < overrideDraft.length ? <b>{index % 2 ? 'End' : 'Start'} selected</b> : index === overrideDraft.length ? <em>{step}</em> : <small>{step}</small>}</div>
          </div>)}
        </div>
      </> : compoundOverride ? <>
        <p className="hint">{compoundOverride.hint} Individual segments are shown while this tool is active; click one again to deselect it. Segments already in another compound move to the new one.</p>
        <div className="selection-slots">
          {selectedFeatures.length ? selectedFeatures.map((feature, index) => <div className="selection-slot filled" key={feature.id}>
            <span>{index + 1}</span><div><b>{featureName(feature)}</b><small>{shortFeatureId(feature.id)}</small></div>
          </div>) : <div className="selection-slot"><span>1</span><div><em>Click a segment</em></div></div>}
        </div>
        <button className="primary-button" disabled={!draft.length} onClick={() => void applyCompound()}>Apply {compoundOverride.label.toLowerCase()} (Enter)</button>
      </> : tool === 'remove_compound' ? <p className="hint">Hover a compound (thread, slot, or tooth profile) until it is highlighted, then click it. It is ungrouped back into its lines and arcs, and measurements of it are removed.</p>
      : geometryOverride ? <p className="hint">{tool === 'redefine_circle' ? 'Click near a closed contour to redefine its entire ring. The change is applied immediately.' : 'Click the start and end points in contour order. Both points snap to the same ring, and the change is applied on the second click.'} Press Esc to cancel.</p> : tool === 'select' ? <p className="hint">Choose distance or angle from the toolbar, then select two outlined features.</p> : <>
        {datumOperators.has(tool as MeasurementTool) && <p className="hint">Select a datum feature, then the feature to inspect. If the first feature is not yet a datum, it will become one automatically.</p>}
        <div className="selection-slots">
          {Array.from({ length: spec?.count || 0 }, (_, index) => <div className={selectedFeatures[index] ? 'selection-slot filled' : 'selection-slot'} key={index}>
            <span>{datumOperators.has(tool as MeasurementTool) && index === 0 ? (selectedDatumLabel || 'D') : index + 1}</span><div>{selectedFeatures[index] ? <><b>{featureName(selectedFeatures[index])}</b><small>{shortFeatureId(selectedFeatures[index].id)}</small></> : <em>{datumOperators.has(tool as MeasurementTool) && index === 0 ? 'Click a datum feature' : 'Click a feature'}</em>}</div>
          </div>)}
        </div>
        {spec?.count === 1
          ? <p className="hint">Click a feature to add the measurement.</p>
          : <button className="primary-button" disabled={draft.length !== spec?.count || duplicateSelection} onClick={() => runRelationship()}>{duplicateSelection ? 'Measurement already added' : 'Add measurement (Enter)'}</button>}
      </>}
      {error && <div className="inline-error">{error}</div>}
    </section>

    <section className="panel-section results-section">
      <div className="eyebrow">Measurements{relationships.length > 0 && <span>{relationships.length}</span>}</div>
      {suspendedRelationships.length > 0 && <p className="hint">{suspendedRelationships.length} measurement{suspendedRelationships.length === 1 ? ' is' : 's are'} hidden because {suspendedRelationships.length === 1 ? 'its features are' : 'their features are'} unavailable at this corner exclusion.</p>}
      {!relationships.length ? (!suspendedRelationships.length && <p className="hint">Relative measurements will appear here and on the image.</p>) : relationships.map((result) => {
        const hidden = hiddenRelationshipIds.includes(result.id)
        const detail = isMetricTable(result) || result.operator === 'distance_construction'
        const construction = result.definition?.construction
        const kind = result.operator === 'datum' ? `Datum ${result.datum_label || '?'}`
          : result.operator === 'distance_construction' ? `Distance${construction ? ` · ${constructionNames[construction]}` : ''}`
          : specs[result.operator]?.label || result.operator.replaceAll('_', ' ')
        const refs = datumOperators.has(result.operator as MeasurementTool)
          ? `${(result.feature_refs[1] ? shortFeatureId(result.feature_refs[1]) : undefined)} | datum ${result.datum_label || '?'}`
          : result.feature_refs.map(shortFeatureId).join(' ↔ ')
        return <article className={`result-row ${hidden ? 'hidden-annotation' : ''} ${detail ? 'has-detail' : ''}`} key={result.id}>
          <span className="result-symbol" aria-hidden="true"><ResultSymbol result={result}/></span>
          <div className="result-kind">{kind}<span className="result-refs">{refs}{hidden ? ' · hidden on image' : ''}</span></div>
          {result.operator !== 'datum' && !isMetricTable(result) ? <ResultValue value={result.value} units={result.units}/> : <span/>}
          {detail && <div className="result-detail">{isMetricTable(result) ? <MetricTable measurement={result}/> : <DistanceResult measurement={result}/>}</div>}
          <div className="result-actions">
            {result.operator === 'distance_construction' && <button className="icon-button" title="Edit distance" aria-label="Edit distance" onClick={() => editMeasurement(result.id)}><Pencil size={14}/></button>}
            {isAnnotatedMeasurement(result) && <button className="icon-button visibility-button"
              title={hidden ? 'Show annotation' : 'Hide annotation'}
              aria-label={hidden ? 'Show annotation' : 'Hide annotation'}
              onClick={() => toggleRelationshipVisibility(result.id)}>
              {hidden ? <EyeOff size={14}/> : <Eye size={14}/>}
            </button>}
            <button className="icon-button" title="Remove measurement" aria-label="Remove measurement" onClick={() => removeRelationship(result.id)}><X size={14}/></button>
          </div>
        </article>
      })}
    </section>
    <details className="inspector-settings"><summary>Image and edge settings<ChevronDown size={15}/></summary>
    <section className="panel-section">
      <div className="eyebrow">Measurement accuracy</div>
      <div className="measurement-control standalone">
        <div><span>Corner exclusion</span><b>{cornerExclusion.toFixed(0)} px</b></div>
        <input type="range" min="0" max="5" step="1" value={cornerExclusion}
          disabled={!analysis}
          aria-label="Corner exclusion radius in pixels"
          onChange={(event) => {
            const radius = Number(event.target.value)
            setCornerExclusion(radius)
            void updateCornerExclusion(radius)
          }}/>
        <div className="range-labels"><span>0 px</span><span>5 px</span></div>
        <p>Remove corners from measurements. Recommended for highest accuracy.</p>
      </div>
    </section>

    <section className="panel-section">
      <div className="eyebrow">Display</div>
      <div className="deformation-control standalone">
        <div><span>Accentuate deformation</span><b>{deformationFactor.toFixed(deformationFactor < 10 ? 1 : 0)}×</b></div>
        <input type="range" min="1" max={deformationMax} step={Math.max(.1, (deformationMax - 1) / 100)}
          disabled={deformationMax <= 1}
          value={Math.min(deformationFactor, deformationMax)} onChange={(event) => {
            const value = Number(event.target.value)
            const snap = 1 + Math.max(.15, (deformationMax - 1) * .035)
            setDeformationFactor(value <= snap ? 1 : value)
          }}/>
        <div className="range-labels"><span>1×</span><span>{deformationMax.toFixed(deformationMax < 10 ? 1 : 0)}×</span></div>
        <p>Display only; measurements use their selected geometry basis at the original scale.</p>
      </div>
    </section>

    </details>
  </aside>
}

export default function App() {
  const store = useInspectionStore()
  const { analysis, extractionFailure, loading, tool, setAnalysis, setExtractionFailure, setLoading, setError, setTool, togglePinned, addDraft, toggleDraft, clearSelections, clearDraft, addRelationship, setOverrideDraft, applyGeometryOverride, applyCornerExclusion } = store
  const [controlDown, setControlDown] = useState(false)
  const [cameraState, setCameraState] = useState<CameraState>({ phase: 'idle', devices: [], camera: null, controls: null, error: null })
  const [pausedImageUrl, setPausedImageUrl] = useState<string | null>(null)
  const [roiTool, setRoiTool] = useState<RoiTool>(null)
  const [cameraRoi, setCameraRoi] = useState<CameraRoi | null>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [inspectorWidth, setInspectorWidth] = useState(inspectorWidthRange.initial)
  // Escape closes Settings on its own; it must not also leave the active tool.
  const settingsOpenRef = useRef(false)
  settingsOpenRef.current = settingsOpen
  const geometryRequestPending = useRef(false)
  const geometryRequestVersion = useRef(0)
  const cornerRequestVersion = useRef(0)
  const sideTrails = useRef<Point[][]>([])

  const loadImage = useCallback(async (path: string, validMask: CameraAnalysisMask | null = null) => {
    cornerRequestVersion.current += 1
    setLoading(true); setError(null); setAnalysis(null)
    try {
      const result = await window.inspection.analyze(path, validMask)
      if ('error' in result) { setExtractionFailure(result); setError(result.error) }
      else setAnalysis(result)
    }
    catch (error) { setError(error instanceof Error ? error.message : String(error)) }
    finally { setLoading(false) }
  }, [setAnalysis, setError, setExtractionFailure, setLoading])

  useEffect(() => { window.inspection.defaultSample().then(loadImage) }, [loadImage])
  useEffect(() => {
    void window.inspection.getState().then((state) => setInspectorWidth(state.preferences.values.inspectorWidth ?? inspectorWidthRange.initial))
  }, [])
  const saveInspectorWidth = useCallback((width: number) => {
    void window.inspection.updatePreferences({ inspectorWidth: width }).catch((error) => setError(error instanceof Error ? error.message : String(error)))
  }, [setError])
  useEffect(() => {
    void window.inspection.cameraState().then(setCameraState)
    return window.inspection.onCameraState(setCameraState)
  }, [])
  useEffect(() => {
    const onDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        geometryRequestVersion.current += 1
        const state = useInspectionStore.getState()
        const hasSelection = state.draft.length || state.overrideDraft.length || state.pinned.length ||
          state.distancePreview || state.editingMeasurementId
        // Escape first clears a selection; with nothing selected it leaves the tool.
        if (hasSelection) clearSelections()
        else if (!settingsOpenRef.current && state.tool !== 'select') setTool('select')
      }
      if (event.key === 'Control') setControlDown(true)
    }
    const onUp = (event: KeyboardEvent) => { if (event.key === 'Control') setControlDown(false) }
    const onBlur = () => setControlDown(false)
    window.addEventListener('keydown', onDown); window.addEventListener('keyup', onUp); window.addEventListener('blur', onBlur)
    return () => { window.removeEventListener('keydown', onDown); window.removeEventListener('keyup', onUp); window.removeEventListener('blur', onBlur) }
  }, [clearSelections, setTool])

  useEffect(() => {
    void window.inspection.setCameraRoi(cameraRoi).catch((error) => setError(String(error)))
  }, [cameraRoi, setError])

  const openImage = async () => { const path = await window.inspection.chooseImage(); if (path) { setPausedImageUrl(null); setCameraRoi(null); setRoiTool(null); await loadImage(path) } }
  const cameraLive = ['starting', 'streaming', 'pausing'].includes(cameraState.phase)
  const toggleCamera = async () => {
    setError(null)
    try {
      if (cameraState.phase === 'streaming') {
        const snapshot = await window.inspection.pauseCamera(cameraRoi)
        setRoiTool(null)
        setPausedImageUrl(snapshot.imageUrl)
        await loadImage(snapshot.snapshotPath, snapshot.analysisMask)
      } else if (cameraState.phase !== 'starting' && cameraState.phase !== 'pausing') {
        clearSelections(); setTool('select')
        await window.inspection.startCamera(cameraState.camera?.id)
      }
    } catch (error) { setError(error instanceof Error ? error.message : String(error)) }
  }
  const setCameraControl = async (name: 'exposureAuto' | 'exposure' | 'gainAuto' | 'gain', value: boolean | number) => {
    try { await window.inspection.setCameraControl(name, value) }
    catch (error) { setError(error instanceof Error ? error.message : String(error)) }
  }
  const removeCompound = async (feature: FeatureRecord) => {
    if (geometryRequestPending.current) return
    const sourceAnalysis = useInspectionStore.getState().analysis
    geometryRequestPending.current = true
    setError(null)
    try {
      const result = await window.inspection.removeCompound({ featureId: feature.id })
      if (!result.prediction || !result.report) throw new Error('Compound removal returned no updated analysis')
      if (useInspectionStore.getState().analysis === sourceAnalysis) applyGeometryOverride(result.prediction, result.report, result.invalidated_feature_ids)
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setError(message.replace(/^Error invoking remote method '[^']+': (Error: )?/, ''))
    } finally {
      geometryRequestPending.current = false
    }
  }
  const onFeatureClick = (feature: FeatureRecord) => {
    if (isContourPicking(tool)) return
    if (tool === 'remove_compound') {
      if (feature.label_level === 'compound') void removeCompound(feature)
      return
    }
    if (isCompoundOverride(tool)) {
      if ((feature.label_level || 'base') !== 'base' || feature.type === 'point') setError('Compounds group contour segments; select base lines, arcs, circles, or profiles.')
      else { setError(null); toggleDraft(feature.id) }
      return
    }
    if (tool === 'select') togglePinned(feature.id)
    else if (feature.status !== 'fitted') setError(feature.reason || 'This feature is unavailable for measurement.')
    else {
      const count = analysis?.report.interaction?.measurement_specs[tool]?.count || fallbackSpecs[tool].count
      const target = compoundMeasurementTargets[tool]
      if (target && !target.accepts(feature)) {
        setError(target.message); return
      }
      // Single-selection tools apply on click; there is nothing to confirm.
      if (count === 1) clearDraft()
      setError(null); addDraft(feature.id, count)
      if (count === 1) void runRelationship()
    }
  }
  const runRelationship = useCallback(async (params: Record<string, unknown> = {}) => {
    const draft = useInspectionStore.getState().draft
    const activeTool = useInspectionStore.getState().tool
    if (activeTool === 'select' || activeTool === 'remove_compound' || isGeometryOverride(activeTool) || isCompoundOverride(activeTool)) return
    const currentAnalysis = useInspectionStore.getState().analysis
    const currentRelationships = useInspectionStore.getState().relationships
    const count = currentAnalysis?.report.interaction?.measurement_specs[activeTool]?.count || fallbackSpecs[activeTool].count
    if (draft.length !== count) return
    // A single selection applied on click has no Add button to retry; drop it on failure.
    const failed = (message: string) => {
      setError(message.replace(/^Error invoking remote method '[^']+': (Error: )?/, ''))
      if (count === 1) clearDraft()
    }
    if (currentRelationships.some((result) => measurementIdentity(result.operator, result.feature_refs, result) === measurementIdentity(activeTool, draft, params))) {
      failed('That measurement has already been added for the selected feature set.')
      return
    }
    setError(null)
    try {
      const measured = await window.inspection.measureRelationship({ id: `user_${Date.now()}`, operator: activeTool, featureRefs: draft, params })
      let referenceDatum = datumOperators.has(activeTool) ? currentRelationships.find(
        (item) => item.operator === 'datum' && item.feature_refs[0] === draft[0],
      ) : undefined
      if (datumOperators.has(activeTool) && !referenceDatum) {
        const label = nextDatumLabel(currentRelationships)
        const measuredDatum = await window.inspection.measureRelationship({
          id: `user_datum_${Date.now()}`, operator: 'datum', featureRefs: [draft[0]], params: {},
        })
        referenceDatum = { ...measuredDatum, datum_label: label, datum_feature_ref: draft[0] }
        addRelationship(referenceDatum)
      }
      const result = activeTool === 'datum'
        ? { ...measured, datum_label: nextDatumLabel(currentRelationships), datum_feature_ref: draft[0] }
        : datumOperators.has(activeTool)
          ? { ...measured, datum_label: referenceDatum?.datum_label, datum_feature_ref: draft[0], datum: referenceDatum?.datum }
          : measured
      if (useInspectionStore.getState().analysis !== currentAnalysis) return
      addRelationship(result); clearDraft()
    } catch (error) { failed(error instanceof Error ? error.message : String(error)) }
  }, [addRelationship, clearDraft, setError])
  const updateCornerExclusion = useCallback(async (radius: number) => {
    const state = useInspectionStore.getState()
    const sourcePath = state.analysis?.path
    if (!sourcePath) return
    const requestVersion = ++cornerRequestVersion.current
    setError(null)
    try {
      // Saved measurements are re-evaluated with the new retained support. Those whose
      // features drop out are suspended and return when a later radius restores them.
      const result = await window.inspection.setCornerExclusion(radius, allRelationships(state).map(remeasureRequest))
      const current = useInspectionStore.getState().analysis
      if (cornerRequestVersion.current === requestVersion && current?.path === sourcePath) {
        applyCornerExclusion(result.prediction, result.report, result.measurements)
      }
    } catch (error) {
      if (cornerRequestVersion.current === requestVersion) {
        setError(error instanceof Error ? error.message : String(error))
      }
    }
  }, [applyCornerExclusion, setError])
  const applyCompound = useCallback(async () => {
    const state = useInspectionStore.getState()
    if (!isCompoundOverride(state.tool) || !state.draft.length || geometryRequestPending.current) return
    const sourceAnalysis = state.analysis
    geometryRequestPending.current = true
    setError(null)
    try {
      const result = await window.inspection.redefineCompound({ semantic: compoundOverrideKinds[state.tool].semantic, featureIds: state.draft })
      if (!result.prediction || !result.report) throw new Error('Compound redefinition returned no updated analysis')
      if (useInspectionStore.getState().analysis === sourceAnalysis) applyGeometryOverride(result.prediction, result.report, result.invalidated_feature_ids)
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setError(message.replace(/^Error invoking remote method '[^']+': (Error: )?/, ''))
    } finally {
      geometryRequestPending.current = false
    }
  }, [applyGeometryOverride, setError])
  const redefineGeometry = useCallback(async (point: Point, cursorPath: Point[]) => {
    const state = useInspectionStore.getState()
    const activeTool = state.tool
    if (!isContourPicking(activeTool) || geometryRequestPending.current) return
    const points = [...state.overrideDraft, point]
    if (isSideCompound(activeTool)) {
      // Keep each side's cursor trail; it chooses the route around a closed contour.
      if (points.length === 1) sideTrails.current = []
      if (points.length % 2 === 0) sideTrails.current = [...sideTrails.current.slice(0, points.length / 2 - 1), cursorPath]
    }
    const requestVersion = geometryRequestVersion.current
    geometryRequestPending.current = true
    setError(null)
    try {
      const result = isSideCompound(activeTool)
        ? await window.inspection.redefineCompound({ semantic: compoundOverrideKinds[activeTool].semantic, points, cursorPaths: sideTrails.current })
        : isGeometryOverride(activeTool)
          ? await window.inspection.redefineGeometry({ type: geometryOverrideKinds[activeTool], points, cursorPath })
          : null
      if (!result) return
      if (result.complete) {
        if (!result.prediction || !result.report) throw new Error('Geometry redefinition returned no updated analysis')
        applyGeometryOverride(result.prediction, result.report, result.invalidated_feature_ids)
      } else {
        if (geometryRequestVersion.current === requestVersion && useInspectionStore.getState().tool === activeTool) {
          setOverrideDraft(result.snapped_points)
        }
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setError(message.replace(/^Error invoking remote method '[^']+': (Error: )?/, ''))
    } finally {
      geometryRequestPending.current = false
    }
  }, [applyGeometryOverride, setError, setOverrideDraft])
  const totalMs = useMemo(() => analysis ? Object.values(analysis.timingsMs).reduce((sum, value) => sum + value, 0) : 0, [analysis])

  return <main className="app-shell">
    <header className="top-bar">
      <div className="brand"><span className="brand-mark"><CircleDotDashed size={16}/></span>Measurific</div>
      <div className={`image-status ${!cameraLive && !analysis && extractionFailure ? 'error' : ''}`}>{cameraLive && cameraState.camera ? <>{cameraState.camera.model}<small>{cameraState.camera.width} × {cameraState.camera.height} · {cameraState.camera.sensorBits}-bit {cameraState.camera.channel === 'camera' ? 'grayscale' : `${cameraState.camera.channel} channel`}</small></> : analysis ? <>{analysis.name}<small>{analysis.width} × {analysis.height} · {analysis.report.summary.measured} measurements{analysis.report.summary.failed_or_unavailable ? ` · ${analysis.report.summary.failed_or_unavailable} unavailable` : ''} · {(totalMs / 1000).toFixed(1)} s</small></> : extractionFailure && <>{extractionFailure.name}<small>Extraction failed · diagnostic overlay shown</small></>}</div>
      <div className="top-actions">
        <button className={`stream-button ${cameraLive ? 'active' : ''}`} onClick={toggleCamera} disabled={cameraState.phase === 'starting' || cameraState.phase === 'pausing'}>
          {cameraState.phase === 'starting' || cameraState.phase === 'pausing' ? <LoaderCircle className="spinner" size={15}/> : cameraLive ? <Pause size={15}/> : <span className="live-dot"/>}{cameraLive ? 'Pause' : 'Live camera'}
        </button>
        <button className="open-button" onClick={openImage} disabled={cameraLive}><FolderOpen size={15}/>Open image</button>
        <Tooltip.Root><Tooltip.Trigger asChild><button className="stream-button settings-button" aria-label="Settings" onClick={() => setSettingsOpen(true)}><Settings size={16}/></button></Tooltip.Trigger><Tooltip.Portal><Tooltip.Content className="radix-tooltip" side="bottom">Settings</Tooltip.Content></Tooltip.Portal></Tooltip.Root>
      </div>
    </header>
    <nav className="tool-bar">
      {cameraLive && <ToolGroup label="Region of interest">
        <ToolButton tool="roi_rectangle" active={roiTool === 'rectangle'} icon={<Square size={18}/>} label="Rectangular region of interest" onClick={() => setRoiTool(roiTool === 'rectangle' ? null : 'rectangle')}/>
        <ToolButton tool="roi_circle" active={roiTool === 'circle'} icon={<Circle size={18}/>} label="Circular region of interest" onClick={() => setRoiTool(roiTool === 'circle' ? null : 'circle')}/>
      </ToolGroup>}
      <ToolGroup label="Inspect">
        <ToolButton tool="select" active={!cameraLive && tool === 'select'} disabled={cameraLive} icon={<MousePointer2 size={18}/>} label="Inspect and pin features" onClick={() => setTool('select')}/>
      </ToolGroup>
      <ToolGroup label="Redefine">
        <ToolButton tool="redefine_line" active={tool === 'redefine_line'} disabled={cameraLive} icon={<Minus size={18}/>} label="Redefine contour as line" onClick={() => setTool('redefine_line')}/>
        <ToolButton tool="redefine_arc" active={tool === 'redefine_arc'} disabled={cameraLive} icon={<span className="tool-glyph">⌒</span>} label="Redefine contour as arc" onClick={() => setTool('redefine_arc')}/>
        <ToolButton tool="redefine_circle" active={tool === 'redefine_circle'} disabled={cameraLive} icon={<Circle size={18}/>} label="Redefine closed ring as circle" onClick={() => setTool('redefine_circle')}/>
        <div className="tool-separator"/>
        <ToolButton tool="redefine_full_round_slot" active={tool === 'redefine_full_round_slot'} disabled={cameraLive} icon={<CompoundGlyph kind="slot"/>} label="Group segments as full round slot" onClick={() => setTool('redefine_full_round_slot')}/>
        <ToolButton tool="redefine_triangular_thread" active={tool === 'redefine_triangular_thread'} disabled={cameraLive} icon={<CompoundGlyph kind="triangular"/>} label="Redefine thread (triangular): click start and end of each side" onClick={() => setTool('redefine_triangular_thread')}/>
        <ToolButton tool="redefine_trapezoidal_thread" active={tool === 'redefine_trapezoidal_thread'} disabled={cameraLive} icon={<CompoundGlyph kind="trapezoidal"/>} label="Redefine thread (trapezoidal): click start and end of each side" onClick={() => setTool('redefine_trapezoidal_thread')}/>
        <ToolButton tool="redefine_tooth_profile" active={tool === 'redefine_tooth_profile'} disabled={cameraLive} icon={<Cog size={18}/>} label="Group segments as tooth profile" onClick={() => setTool('redefine_tooth_profile')}/>
        <ToolButton tool="remove_compound" active={tool === 'remove_compound'} disabled={cameraLive} icon={<Ungroup size={18}/>} label="Remove compound: click a highlighted compound to restore its primitives" onClick={() => setTool('remove_compound')}/>
      </ToolGroup>
      <ToolGroup label="Measure">
        <ToolButton tool="distance_construction" active={tool === 'distance_construction'} disabled={cameraLive} icon={<Ruler size={18}/>} label="Distance" onClick={() => setTool('distance_construction')}/>
        <ToolButton tool="angle" active={tool === 'angle'} disabled={cameraLive} icon={<TriangleRight size={18}/>} label="Angle between lines" onClick={() => setTool('angle')}/>
      </ToolGroup>
      <ToolGroup label="Size">
        <ToolButton tool="least_squares_diameter" active={tool === 'least_squares_diameter'} disabled={cameraLive} icon={<span className="tool-glyph">Ø</span>} label="Diameter of a circle" onClick={() => setTool('least_squares_diameter')}/>
        <ToolButton tool="radius" active={tool === 'radius'} disabled={cameraLive} icon={<span className="tool-glyph">R</span>} label="Radius of an arc or circle" onClick={() => setTool('radius')}/>
        <ToolButton tool="length" active={tool === 'length'} disabled={cameraLive} icon={<span className="tool-glyph">↔</span>} label="Length of a line" onClick={() => setTool('length')}/>
        <ToolButton tool="angular_span" active={tool === 'angular_span'} disabled={cameraLive} icon={<span className="tool-glyph">⌒</span>} label="Arc angle" onClick={() => setTool('angular_span')}/>
      </ToolGroup>
      <ToolGroup label="Compound measurements">
        <ToolButton tool="thread" active={tool === 'thread'} disabled={cameraLive} icon={<span className="tool-glyph">≋</span>} label="Thread measurements" onClick={() => setTool('thread')}/>
        <ToolButton tool="slot" active={tool === 'slot'} disabled={cameraLive} icon={<CompoundGlyph kind="slot"/>} label="Slot measurements" onClick={() => setTool('slot')}/>
        <ToolButton tool="tooth_profile" active={tool === 'tooth_profile'} disabled={cameraLive} icon={<Cog size={18}/>} label="Tooth profile measurements" onClick={() => setTool('tooth_profile')}/>
      </ToolGroup>
      <ToolGroup label="Planar GD&T">
        <ToolButton tool="datum" active={tool === 'datum'} disabled={cameraLive} icon={<span className="tool-glyph datum-tool-glyph">A</span>} label="Define datum feature" onClick={() => setTool('datum')}/>
        <ToolButton tool="concentricity" active={tool === 'concentricity'} disabled={cameraLive} icon={<span className="tool-glyph">◎</span>} label="Concentricity" onClick={() => setTool('concentricity')}/>
        <ToolButton tool="perpendicularity" active={tool === 'perpendicularity'} disabled={cameraLive} icon={<span className="tool-glyph">⊥</span>} label="Perpendicularity" onClick={() => setTool('perpendicularity')}/>
        <ToolButton tool="parallelism" active={tool === 'parallelism'} disabled={cameraLive} icon={<span className="tool-glyph">∥</span>} label="Parallelism" onClick={() => setTool('parallelism')}/>
        <ToolButton tool="circularity_ls" active={tool === 'circularity_ls'} disabled={cameraLive} icon={<span className="tool-glyph">○</span>} label="Circularity" onClick={() => setTool('circularity_ls')}/>
        <ToolButton tool="straightness" active={tool === 'straightness'} disabled={cameraLive} icon={<span className="tool-glyph">—</span>} label="Straightness" onClick={() => setTool('straightness')}/>
        <ToolButton tool="symmetry" active={tool === 'symmetry'} disabled={cameraLive} icon={<span className="tool-glyph">⌯</span>} label="Symmetry" onClick={() => setTool('symmetry')}/>
      </ToolGroup>
      <div className="tool-bar-spacer"/>
      <ToolGroup label="Clear">
        <Tooltip.Root><Tooltip.Trigger asChild><button className="tool-button" aria-label={cameraLive ? 'Clear region of interest' : 'Clear selections'} disabled={cameraLive ? !cameraRoi : false} onClick={cameraLive ? () => { setCameraRoi(null); setRoiTool(null) } : clearSelections}><Trash2 size={18}/></button></Tooltip.Trigger><Tooltip.Portal><Tooltip.Content className="radix-tooltip" side="bottom">{cameraLive ? 'Clear region of interest' : 'Clear selections (Esc)'}</Tooltip.Content></Tooltip.Portal></Tooltip.Root>
      </ToolGroup>
    </nav>
    <div className="workspace" style={{ '--inspector-width': `${inspectorWidth}px` } as CSSProperties}>
      <ImageStage onFeatureClick={onFeatureClick} onGeometryClick={redefineGeometry} controlDown={controlDown} cameraLive={cameraLive} pausedImageUrl={pausedImageUrl} roiTool={roiTool} cameraRoi={cameraRoi} onRoiChange={setCameraRoi}/>
      {cameraLive
        ? <CameraPanel state={cameraState} roi={cameraRoi} onDiscover={() => { void window.inspection.discoverCameras() }} onSet={(name, value) => { void setCameraControl(name, value) }}/>
        : <InspectorPanel runRelationship={runRelationship} updateCornerExclusion={updateCornerExclusion} applyCompound={applyCompound}/>}
      <PanelResizer width={inspectorWidth} onChange={setInspectorWidth} onCommit={saveInspectorWidth}/>
    </div>
    {settingsOpen && <SettingsPanel onClose={() => setSettingsOpen(false)} canReanalyze={Boolean(analysis) && !cameraLive}
      onReanalyze={() => { setSettingsOpen(false); if (analysis) void loadImage(analysis.path) }}/>}
    {loading && <div className="loading-screen"><LoaderCircle className="spinner" size={30}/><strong>Analyzing image</strong><span>Subpixel edges · primitives and compounds · measurements</span></div>}
  </main>
}
