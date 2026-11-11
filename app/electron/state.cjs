// Persistent settings, preferences and calibrations (electron-store JSON files).
//
// Settings and preferences are per user. Calibrations belong to the machine:
// they are saved in a machine-wide folder when this user can write it and in
// the user's folder otherwise, and both folders are always read.
const { execFile } = require('node:child_process')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')

const SETTINGS_VERSION = 1
const PREFERENCES_VERSION = 1
const CALIBRATIONS_VERSION = 1
const EXPORT_FORMAT = 'measurific-state'
// Exports made before the rename to Measurific.
const LEGACY_EXPORT_FORMATS = ['telecentric-inspector-state']

const settingsSchema = {
  schemaVersion: { type: 'integer', default: SETTINGS_VERSION },
  computeDevice: { enum: ['cpu', 'gpu'], default: 'cpu' },
  // Subpixel edges: the coarse model + PAE (fast, sharp edges) or the full
  // coarse + fine deep learning edge model (blurred edges, very fine detail).
  edgeMethod: { enum: ['pae', 'deep-learning'], default: 'pae' },
  // Camera image: one Bayer color site at half resolution, or the camera's own
  // grayscale conversion (demosaiced and color-processed).
  cameraChannel: { enum: ['red', 'green', 'blue', 'camera'], default: 'red' },
}
const preferencesSchema = {
  schemaVersion: { type: 'integer', default: PREFERENCES_VERSION },
  lastImageDirectory: { type: ['string', 'null'], default: null },
  // Width of the right-hand inspector panel in CSS pixels.
  inspectorWidth: { type: 'integer', minimum: 260, maximum: 720, default: 336 },
}
const calibrationsSchema = {
  schemaVersion: { type: 'integer', default: CALIBRATIONS_VERSION },
  calibration: { type: 'object', additionalProperties: { type: 'object' } },
}

// Development and CI overrides; the settings UI shows these as locked.
const settingsEnvironment = {
  computeDevice: ['INSPECTION_DEVICE', (value) => value],
  edgeMethod: ['INSPECTION_EDGE_METHOD', (value) => value],
  cameraChannel: ['INSPECTION_CAMERA_CHANNEL', (value) => value],
}
// Settings the Python worker is started with; changing one restarts it.
const workerSettingKeys = ['computeDevice', 'edgeMethod']

function machineDataDirectory(appName, platform = process.platform, env = process.env) {
  if (env.INSPECTION_MACHINE_DATA_DIR) return env.INSPECTION_MACHINE_DATA_DIR
  // Locations every account can reach without administrator rights, except on
  // Linux, where an administrator must create the folder (see README).
  if (platform === 'darwin') return path.join('/Users/Shared', appName)
  if (platform === 'win32') return path.join(env.ProgramData || 'C:\\ProgramData', appName)
  return path.join('/var/lib', appName.toLowerCase().replace(/\s+/g, '-'))
}

function canWrite(directory) {
  try {
    fs.mkdirSync(directory, { recursive: true })
    const probe = path.join(directory, `.write-test-${process.pid}-${Date.now()}`)
    fs.writeFileSync(probe, '')
    fs.rmSync(probe)
    return true
  } catch {
    return false
  }
}

// Folders created by one account must stay writable by the machine's other accounts.
function shareDirectory(directory, platform = process.platform) {
  if (platform === 'win32') {
    // S-1-5-32-545 is the local Users group; only the folder's creator may change its ACL.
    execFile('icacls', [directory, '/grant', '*S-1-5-32-545:(OI)(CI)M', '/Q'], { windowsHide: true }, () => {})
    return
  }
  try { fs.chmodSync(directory, 0o777) } catch { /* owned by another account */ }
}

function shareFile(file, platform = process.platform) {
  if (platform === 'win32') return
  try { fs.chmodSync(file, 0o666) } catch { /* owned by another account */ }
}

function calibrationId(existing, now = new Date()) {
  // electron-store treats dots as key paths, so the ISO timestamp drops them.
  let time = now.getTime()
  let id
  do { id = new Date(time++).toISOString().replace(/[:.]/g, '-') } while (existing.has(id))
  return id
}

function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

class PersistentState {
  constructor({ Store, userDirectory, machineDirectory, appVersion = '0.0.0', platform = process.platform, env = process.env, now = () => new Date() }) {
    this.Store = Store
    this.platform = platform
    this.env = env
    this.now = now
    this.appVersion = appVersion
    this.userDirectory = userDirectory
    this.machineDirectory = machineDirectory
    this.warnings = []
    this.settings = this.openRecoverable('settings', settingsSchema)
    this.preferences = this.openRecoverable('preferences', preferencesSchema)
    this.openCalibrations()
  }

  // A damaged settings file is set aside rather than blocking startup.
  openRecoverable(name, schema) {
    const options = { name, cwd: this.userDirectory, schema, clearInvalidConfig: false }
    try {
      const store = new this.Store(options)
      // Keys a newer schema no longer has (e.g. recognition options) are dropped.
      const known = Object.fromEntries(Object.entries(store.store).filter(([key]) => Object.hasOwn(schema, key)))
      if (Object.keys(known).length !== Object.keys(store.store).length) store.store = known
      return store
    } catch (error) {
      const file = path.join(this.userDirectory, `${name}.json`)
      const aside = path.join(this.userDirectory, `${name}.invalid-${Date.now()}.json`)
      try { fs.renameSync(file, aside) } catch { /* nothing to set aside */ }
      this.warnings.push(`${name}.json could not be read (${error.message}); defaults are in use and the old file was kept as ${path.basename(aside)}`)
      return new this.Store(options)
    }
  }

  openCalibrations() {
    this.calibrationStores = {}
    this.calibrationErrors = {}
    const machineWritable = canWrite(this.machineDirectory)
    if (machineWritable) shareDirectory(this.machineDirectory, this.platform)
    this.calibrationScope = machineWritable ? 'machine' : 'user'
    for (const [scope, directory] of [['machine', this.machineDirectory], ['user', this.userDirectory]]) {
      const writable = scope === 'user' || machineWritable
      // Only the save target is created; other folders are read when they have calibrations.
      if (scope !== this.calibrationScope && !fs.existsSync(path.join(directory, 'calibrations.json'))) continue
      try {
        this.calibrationStores[scope] = { writable, store: new this.Store({ name: 'calibrations', cwd: directory, schema: calibrationsSchema }) }
      } catch (error) {
        // Never reset calibrations: report the problem and leave the file for inspection.
        this.calibrationErrors[scope] = `${path.join(directory, 'calibrations.json')}: ${error.message}`
      }
    }
  }

  environmentOverrides() {
    const overrides = {}
    for (const [key, [name, parse]] of Object.entries(settingsEnvironment)) {
      if (this.env[name] !== undefined && this.env[name] !== '') overrides[key] = parse(this.env[name])
    }
    return overrides
  }

  effectiveSettings() {
    const overrides = this.environmentOverrides()
    return { ...this.settings.store, ...overrides }
  }

  describe() {
    return {
      settings: { values: this.settings.store, effective: this.effectiveSettings(), overridden: Object.keys(this.environmentOverrides()), path: this.settings.path },
      preferences: { values: this.preferences.store, path: this.preferences.path },
      calibration: {
        scope: this.calibrationScope,
        machineDirectory: this.machineDirectory,
        userDirectory: this.userDirectory,
        directory: this.calibrationScope === 'machine' ? this.machineDirectory : this.userDirectory,
        errors: this.calibrationErrors,
      },
      userDirectory: this.userDirectory,
      warnings: this.warnings,
    }
  }

  updateSettings(patch) {
    const allowed = Object.keys(settingsSchema).filter((key) => key !== 'schemaVersion')
    const next = { ...this.settings.store }
    for (const [key, value] of Object.entries(patch || {})) {
      if (!allowed.includes(key)) throw new Error(`Unknown setting: ${key}`)
      next[key] = value
    }
    this.settings.store = next // validated as a whole; rejects without writing
    return this.describe()
  }

  updatePreferences(patch) {
    const allowed = Object.keys(preferencesSchema).filter((key) => key !== 'schemaVersion')
    const next = { ...this.preferences.store }
    for (const [key, value] of Object.entries(patch || {})) {
      if (!allowed.includes(key)) throw new Error(`Unknown preference: ${key}`)
      next[key] = value
    }
    this.preferences.store = next
    return this.describe()
  }

  listCalibrations() {
    const records = []
    for (const [scope, { store, writable }] of Object.entries(this.calibrationStores)) {
      for (const [id, record] of Object.entries(store.get('calibration', {}))) records.push({ ...record, id, scope, writable })
    }
    return records.sort((a, b) => b.id.localeCompare(a.id))
  }

  getCalibration(id) {
    return this.listCalibrations().find((record) => record.id === id) || null
  }

  saveCalibration(calibration) {
    if (!isPlainObject(calibration)) throw new Error('A calibration must be a JSON object')
    const target = this.calibrationStores[this.calibrationScope]
    if (!target) throw new Error(`Calibrations cannot be saved: ${this.calibrationErrors[this.calibrationScope]}`)
    const { id: _id, scope: _scope, writable: _writable, ...values } = calibration
    const now = this.now()
    const id = calibrationId(new Set(this.listCalibrations().map((record) => record.id)), now)
    const record = { ...values, savedAt: now.toISOString(), savedBy: os.userInfo().username, appVersion: this.appVersion }
    // The id is a key segment, never a dotted path, so records are written whole.
    target.store.set('calibration', { ...target.store.get('calibration', {}), [id]: record })
    if (this.calibrationScope === 'machine') shareFile(target.store.path, this.platform)
    return { ...record, id, scope: this.calibrationScope, writable: true }
  }

  deleteCalibration(id) {
    for (const [scope, { store, writable }] of Object.entries(this.calibrationStores)) {
      const all = store.get('calibration', {})
      if (!Object.hasOwn(all, id)) continue
      if (!writable) throw new Error(`Calibration ${id} is shared by all users of this machine and this account cannot change it`)
      const { [id]: _removed, ...rest } = all
      store.set('calibration', rest)
      if (scope === 'machine') shareFile(store.path, this.platform)
      return true
    }
    return false
  }

  exportBundle() {
    const calibration = {}
    for (const { id, scope: _scope, writable: _writable, ...record } of this.listCalibrations()) calibration[id] = record
    return {
      format: EXPORT_FORMAT, version: 1, exportedAt: this.now().toISOString(), appVersion: this.appVersion,
      settings: this.settings.store, preferences: this.preferences.store, calibration,
    }
  }

  // Settings and preferences are replaced; calibrations are only ever added.
  importBundle(bundle) {
    if (!isPlainObject(bundle) || (bundle.format !== EXPORT_FORMAT && !LEGACY_EXPORT_FORMATS.includes(bundle.format))) throw new Error('This file is not a Measurific settings export')
    if (bundle.version !== 1) throw new Error(`Unsupported export version ${bundle.version}`)
    // Exports from older versions may carry settings that no longer exist.
    const known = (values, schema) => Object.fromEntries(Object.entries(isPlainObject(values) ? values : {})
      .filter(([key]) => key !== 'schemaVersion' && Object.hasOwn(schema, key)))
    this.updateSettings(known(bundle.settings, settingsSchema))
    this.updatePreferences(known(bundle.preferences, preferencesSchema))
    const target = this.calibrationStores[this.calibrationScope]
    const existing = new Map(this.listCalibrations().map((record) => [record.id, record]))
    const added = []
    const conflicts = []
    const incoming = isPlainObject(bundle.calibration) ? bundle.calibration : {}
    const additions = {}
    for (const [id, record] of Object.entries(incoming)) {
      if (!isPlainObject(record) || id.includes('.')) { conflicts.push(id); continue }
      const current = existing.get(id)
      if (!current) { additions[id] = record; added.push(id); continue }
      const { scope: _scope, writable: _writable, id: _id, ...stored } = current
      if (JSON.stringify(stored) !== JSON.stringify(record)) conflicts.push(id)
    }
    if (added.length) {
      if (!target) throw new Error(`Calibrations cannot be saved: ${this.calibrationErrors[this.calibrationScope]}`)
      target.store.set('calibration', { ...target.store.get('calibration', {}), ...additions })
      if (this.calibrationScope === 'machine') shareFile(target.store.path, this.platform)
    }
    return { added, conflicts }
  }
}

module.exports = { PersistentState, machineDataDirectory, workerSettingKeys, calibrationId, EXPORT_FORMAT }
