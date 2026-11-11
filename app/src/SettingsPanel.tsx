import { useCallback, useEffect, useState } from 'react'
import { Download, FolderOpen, LoaderCircle, Lock, RefreshCw, Trash2, Upload, X } from 'lucide-react'
import type { CalibrationRecord, PersistentStateView, Settings, WorkerHealth } from './types'

function errorText(error: unknown) {
  const message = error instanceof Error ? error.message : String(error)
  return message.replace(/^Error invoking remote method '[^']+': (Error: )?/, '')
}

function edgeModelName(format: string) {
  if (format === 'edgesr-coarse-pae-v1') return 'PAE + coarse model'
  if (format.startsWith('edgesr-sdf-')) return 'SDF v5'
  return format
}

function Segmented<T extends string>({ value, options, disabled, onChange }: {
  value: T
  options: [T, string][]
  disabled?: boolean
  onChange: (value: T) => void
}) {
  return <div className="segmented">
    {options.map(([option, label]) => <button key={option} className={value === option ? 'active' : ''} disabled={disabled}
      onClick={() => value !== option && onChange(option)}>{label}</button>)}
  </div>
}

function SettingRow({ label, locked, children }: { label: string; locked: boolean; children: React.ReactNode }) {
  return <div className="setting-row">
    <span>{label}{locked && <Lock size={10} aria-label="Set by an environment variable"/>}</span>
    {children}
  </div>
}

export function SettingsPanel({ onClose, onReanalyze, canReanalyze }: {
  onClose: () => void
  onReanalyze: () => void
  canReanalyze: boolean
}) {
  const [state, setState] = useState<PersistentStateView | null>(null)
  const [health, setHealth] = useState<WorkerHealth | null>(null)
  const [calibrations, setCalibrations] = useState<CalibrationRecord[]>([])
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      const [nextState, nextCalibrations] = await Promise.all([window.inspection.getState(), window.inspection.listCalibrations()])
      setState(nextState)
      setCalibrations(nextCalibrations)
    } catch (caught) { setError(errorText(caught)) }
    window.inspection.workerHealth().then(setHealth, (caught) => setError(`Inspection worker: ${errorText(caught)}`))
  }, [])

  useEffect(() => { void refresh() }, [refresh])
  useEffect(() => window.inspection.onState(setState), [])
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const update = async (patch: Partial<Omit<Settings, 'schemaVersion'>>) => {
    setError(null)
    try {
      setState(await window.inspection.updateSettings(patch))
      setHealth((current) => current && { ...current, pendingRestart: true })
    } catch (caught) { setError(errorText(caught)) }
  }
  const run = async (action: () => Promise<string | null | void>) => {
    setError(null); setMessage(null)
    try {
      const result = await action()
      if (result) setMessage(result)
      await refresh()
    } catch (caught) { setError(errorText(caught)) }
  }

  if (!state) return <div className="settings-overlay"><div className="settings-dialog loading"><LoaderCircle className="spinner" size={20}/></div></div>

  const { effective, overridden } = state.settings
  const locked = (key: keyof Settings) => overridden.includes(key)
  const gpuProvider = health?.availableProviders.find((name) => name !== 'CPUExecutionProvider' && name !== 'AzureExecutionProvider')
  const activeProvider = health?.model?.providers?.[0]
  const calibrationScope = state.calibration.scope === 'machine'
    ? 'Saved for every user of this computer.'
    : 'Saved for your account only: this account cannot write the shared calibration folder.'

  return <div className="settings-overlay" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <div className="settings-dialog" role="dialog" aria-label="Settings">
      <header>
        <h2>Settings</h2>
        <button className="icon-button" aria-label="Close settings" onClick={onClose}><X size={16}/></button>
      </header>
      <div className="settings-body">
        {(error || message || state.warnings.length > 0) && <section className="settings-notices">
          {error && <div className="inline-error">{error}</div>}
          {message && <div className="inline-note">{message}</div>}
          {state.warnings.map((warning) => <div key={warning} className="inline-error">{warning}</div>)}
        </section>}

        <section className="panel-section">
          <div className="eyebrow">Analysis</div>
          <SettingRow label="Edges" locked={locked('edgeMethod')}>
            <Segmented value={effective.edgeMethod} disabled={locked('edgeMethod')}
              options={[['pae', 'PAE'], ['deep-learning', 'Deep Learning']]} onChange={(edgeMethod) => void update({ edgeMethod })}/>
          </SettingRow>
          <p className="hint">
            PAE is much faster and good for sharp edges. Deep Learning works best for blurry edges and very fine detail.
          </p>
          <SettingRow label="Compute" locked={locked('computeDevice')}>
            <Segmented value={effective.computeDevice} disabled={locked('computeDevice')}
              options={[['cpu', 'CPU'], ['gpu', 'GPU when available']]} onChange={(computeDevice) => void update({ computeDevice })}/>
          </SettingRow>
          <p className="hint">
            {gpuProvider ? `GPU inference is available through ${gpuProvider.replace('ExecutionProvider', '')}. ` : 'No GPU inference provider is available on this computer; GPU falls back to CPU. '}
            Changes apply to the next analyzed image.
          </p>
          {health?.pendingRestart && canReanalyze && <button className="secondary-button" onClick={onReanalyze}><RefreshCw size={13}/>Re-analyze current image</button>}
        </section>

        <section className="panel-section">
          <div className="eyebrow">Runtime</div>
          {health ? <dl className="settings-facts">
            <div><dt>App</dt><dd>{health.appVersion}</dd></div>
            <div><dt>Python</dt><dd>{health.python}</dd></div>
            <div><dt>Recognition</dt><dd>ONNX Runtime {health.packages.onnxruntime ?? health.packages['onnxruntime-directml']} · {activeProvider?.replace('ExecutionProvider', '') ?? health.model?.format}</dd></div>
            {health.edgeModel && <div><dt>Edges</dt><dd>{edgeModelName(health.edgeModel.format)} · {health.edgeModel.format.includes('torch') ? 'PyTorch' : 'ONNX Runtime'} · {health.edgeModel.providers?.[0]?.replace('ExecutionProvider', '')}</dd></div>}
            {health.model?.source_checkpoint_sha256 && <div><dt>Model</dt><dd title={health.model.source_checkpoint_sha256}>{health.model.source_checkpoint_sha256.slice(0, 12)}</dd></div>}
          </dl> : <p className="hint">Starting inspection worker…</p>}
        </section>

        <section className="panel-section">
          <div className="eyebrow">Calibrations</div>
          <p className="hint">{calibrationScope}</p>
          {Object.entries(state.calibration.errors).map(([scope, text]) => <div key={scope} className="inline-error">{text}</div>)}
          {calibrations.length === 0
            ? <p className="hint">No calibrations have been saved.</p>
            : <ul className="calibration-list">
              {calibrations.map((record) => <li key={record.id}>
                <div><strong>{new Date(record.savedAt).toLocaleString()}</strong>
                  <small>{record.scope === 'machine' ? 'All users' : 'This user'} · {record.savedBy}</small></div>
                <button className="icon-button" aria-label={`Delete calibration ${record.id}`} disabled={!record.writable}
                  onClick={() => { if (confirm('Delete this calibration? This cannot be undone.')) void run(async () => { await window.inspection.deleteCalibration(record.id) }) }}>
                  <Trash2 size={13}/></button>
              </li>)}
            </ul>}
        </section>

        <section className="panel-section">
          <div className="eyebrow">Data</div>
          <div className="folder-row"><span>Settings</span><code className="path" title={state.userDirectory}>{state.userDirectory}</code>
            <button className="icon-button" aria-label="Open settings folder" onClick={() => void run(() => window.inspection.openDataFolder('user'))}><FolderOpen size={14}/></button></div>
          <div className="folder-row"><span>Calibrations</span><code className="path" title={state.calibration.directory}>{state.calibration.directory}</code>
            <button className="icon-button" aria-label="Open calibration folder" onClick={() => void run(() => window.inspection.openDataFolder('calibration'))}><FolderOpen size={14}/></button></div>
          <div className="settings-actions">
            <button className="secondary-button" onClick={() => void run(async () => {
              const file = await window.inspection.exportState()
              return file ? `Exported to ${file}` : null
            })}><Download size={13}/>Export…</button>
            <button className="secondary-button" onClick={() => void run(async () => {
              const summary = await window.inspection.importState()
              if (!summary) return null
              const conflicts = summary.conflicts.length ? `; ${summary.conflicts.length} with an existing id but different values were skipped` : ''
              return `Imported settings and ${summary.added.length} calibration${summary.added.length === 1 ? '' : 's'}${conflicts}.`
            })}><Upload size={13}/>Import…</button>
          </div>
          <p className="hint">Exports hold settings, preferences and every calibration as one JSON file. Importing replaces settings and adds calibrations; existing calibrations are never overwritten.</p>
        </section>
      </div>
    </div>
  </div>
}
