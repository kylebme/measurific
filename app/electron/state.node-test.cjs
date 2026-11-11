const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { before, test } = require('node:test')
const { PersistentState, calibrationId, machineDataDirectory } = require('./state.cjs')

let Store
before(async () => { ({ default: Store } = await import('electron-store')) })

function directories() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'inspector-state-'))
  return { userDirectory: path.join(root, 'user'), machineDirectory: path.join(root, 'machine'), root }
}

function open(dirs, options = {}) {
  return new PersistentState({ Store, ...dirs, appVersion: '1.2.3', env: {}, ...options })
}

test('settings persist, validate and report environment overrides', () => {
  const dirs = directories()
  const state = open(dirs)
  assert.equal(state.describe().settings.values.computeDevice, 'cpu')
  state.updateSettings({ computeDevice: 'gpu' })
  assert.throws(() => state.updateSettings({ computeDevice: 'tpu' }), /schema violation/)
  assert.throws(() => state.updateSettings({ recognition: 'geometry' }), /Unknown setting/)
  const reopened = open(dirs, { env: { INSPECTION_DEVICE: 'cpu' } })
  const { settings } = reopened.describe()
  assert.equal(settings.values.computeDevice, 'gpu')
  assert.equal(settings.effective.computeDevice, 'cpu')
  assert.deepEqual(settings.overridden, ['computeDevice'])
})

test('the camera channel defaults to red, validates and can be overridden', () => {
  const dirs = directories()
  const state = open(dirs)
  assert.equal(state.describe().settings.values.cameraChannel, 'red')
  state.updateSettings({ cameraChannel: 'camera' })
  assert.throws(() => state.updateSettings({ cameraChannel: 'infrared' }))
  assert.equal(open(dirs).describe().settings.values.cameraChannel, 'camera')
  const overridden = open(dirs, { env: { INSPECTION_CAMERA_CHANNEL: 'blue' } }).describe().settings
  assert.equal(overridden.effective.cameraChannel, 'blue')
  assert.ok(overridden.overridden.includes('cameraChannel'))
})

test('edge extraction defaults to PAE, validates and can be overridden', () => {
  const dirs = directories()
  const state = open(dirs)
  assert.equal(state.describe().settings.values.edgeMethod, 'pae')
  state.updateSettings({ edgeMethod: 'deep-learning' })
  assert.throws(() => state.updateSettings({ edgeMethod: 'threshold' }), /schema violation/)
  assert.equal(open(dirs).describe().settings.values.edgeMethod, 'deep-learning')
  const overridden = open(dirs, { env: { INSPECTION_EDGE_METHOD: 'pae' } }).describe().settings
  assert.equal(overridden.effective.edgeMethod, 'pae')
  assert.ok(overridden.overridden.includes('edgeMethod'))
})

test('settings that no longer exist are dropped from saved files', () => {
  const dirs = directories()
  fs.mkdirSync(dirs.userDirectory, { recursive: true })
  fs.writeFileSync(path.join(dirs.userDirectory, 'settings.json'), JSON.stringify({
    schemaVersion: 1, recognition: 'geometry', modelPath: null, geometryTolerancePx: 0.5, computeDevice: 'gpu',
  }))
  const state = open(dirs)
  const expected = { schemaVersion: 1, computeDevice: 'gpu', edgeMethod: 'pae', cameraChannel: 'red' }
  assert.deepEqual({ ...state.describe().settings.values }, expected)
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(dirs.userDirectory, 'settings.json'), 'utf8')), expected)
  assert.deepEqual(state.warnings, [])
})

test('a damaged settings file is set aside and defaults are used', () => {
  const dirs = directories()
  fs.mkdirSync(dirs.userDirectory, { recursive: true })
  fs.writeFileSync(path.join(dirs.userDirectory, 'settings.json'), '{ not json')
  const state = open(dirs)
  assert.equal(state.describe().settings.values.computeDevice, 'cpu')
  assert.equal(state.warnings.length, 1)
  assert.ok(fs.readdirSync(dirs.userDirectory).some((name) => name.startsWith('settings.invalid-')))
})

test('calibrations are keyed by timestamp and stored per machine when writable', () => {
  const dirs = directories()
  const times = [new Date('2026-09-26T14:03:11.123Z'), new Date('2026-09-26T14:03:11.123Z')]
  const state = open(dirs, { now: () => times.shift() || new Date('2026-09-27T00:00:00.000Z') })
  const first = state.saveCalibration({ cameraSerial: 'A1', micronsPerPixel: 5.2 })
  const second = state.saveCalibration({ cameraSerial: 'A1', micronsPerPixel: 5.3 })
  assert.equal(first.id, '2026-09-26T14-03-11-123Z')
  assert.equal(second.id, '2026-09-26T14-03-11-124Z')
  assert.equal(first.scope, 'machine')
  const file = JSON.parse(fs.readFileSync(path.join(dirs.machineDirectory, 'calibrations.json'), 'utf8'))
  assert.equal(file.calibration[first.id].micronsPerPixel, 5.2)
  assert.equal(file.calibration[first.id].appVersion, '1.2.3')
  assert.deepEqual(open(dirs).listCalibrations().map((record) => record.id), [second.id, first.id])
  assert.equal(state.deleteCalibration(first.id), true)
  assert.equal(state.getCalibration(first.id), null)
})

test('calibrations fall back to the user folder and still read the machine folder', { skip: process.platform === 'win32' || process.getuid?.() === 0 }, () => {
  const dirs = directories()
  const machine = open(dirs)
  const shared = machine.saveCalibration({ micronsPerPixel: 4 })
  fs.chmodSync(dirs.machineDirectory, 0o555)
  try {
    const state = open(dirs)
    assert.equal(state.describe().calibration.scope, 'user')
    const own = state.saveCalibration({ micronsPerPixel: 5 })
    assert.equal(own.scope, 'user')
    assert.deepEqual(state.listCalibrations().map((record) => [record.id, record.scope, record.writable]).sort(),
      [[own.id, 'user', true], [shared.id, 'machine', false]].sort())
    assert.throws(() => state.deleteCalibration(shared.id), /shared by all users/)
  } finally {
    fs.chmodSync(dirs.machineDirectory, 0o755)
  }
})

test('a damaged calibration file is reported and never reset', () => {
  const dirs = directories()
  fs.mkdirSync(dirs.machineDirectory, { recursive: true })
  fs.writeFileSync(path.join(dirs.machineDirectory, 'calibrations.json'), '{ damaged')
  const state = open(dirs)
  assert.match(state.describe().calibration.errors.machine, /calibrations\.json/)
  assert.throws(() => state.saveCalibration({ micronsPerPixel: 1 }), /cannot be saved/)
  assert.equal(fs.readFileSync(path.join(dirs.machineDirectory, 'calibrations.json'), 'utf8'), '{ damaged')
})

test('the inspector width defaults, persists and validates', () => {
  const dirs = directories()
  const state = open(dirs)
  assert.equal(state.describe().preferences.values.inspectorWidth, 336)
  state.updatePreferences({ inspectorWidth: 420 })
  assert.throws(() => state.updatePreferences({ inspectorWidth: 100 }), /schema violation/)
  assert.equal(open(dirs).describe().preferences.values.inspectorWidth, 420)
})

test('export and import round-trip settings and add only new calibrations', () => {
  const source = open(directories())
  source.updateSettings({ computeDevice: 'gpu', edgeMethod: 'deep-learning' })
  source.updatePreferences({ lastImageDirectory: '/parts' })
  const calibration = source.saveCalibration({ micronsPerPixel: 7 })
  const bundle = JSON.parse(JSON.stringify(source.exportBundle()))
  bundle.settings.recognition = 'geometry' // exported by an older version
  const target = open(directories())
  assert.deepEqual(target.importBundle(bundle), { added: [calibration.id], conflicts: [] })
  assert.deepEqual({ ...target.describe().settings.values }, { schemaVersion: 1, computeDevice: 'gpu', edgeMethod: 'deep-learning', cameraChannel: 'red' })
  assert.equal(target.describe().preferences.values.lastImageDirectory, '/parts')
  bundle.calibration[calibration.id].micronsPerPixel = 8
  assert.deepEqual(target.importBundle(bundle), { added: [], conflicts: [calibration.id] })
  assert.equal(target.getCalibration(calibration.id).micronsPerPixel, 7)
  assert.throws(() => target.importBundle({ format: 'other' }), /not a Measurific/)
  // Exports from before the rename to Measurific still import.
  const legacy = open(directories())
  assert.deepEqual(legacy.importBundle({ ...bundle, format: 'telecentric-inspector-state' }), { added: [calibration.id], conflicts: [] })
})

test('machine folders do not need administrator rights on macOS and Windows', () => {
  assert.equal(machineDataDirectory('Measurific', 'darwin', {}), '/Users/Shared/Measurific')
  assert.equal(machineDataDirectory('Measurific', 'win32', { ProgramData: 'D:\\ProgramData' }), path.join('D:\\ProgramData', 'Measurific'))
  assert.equal(machineDataDirectory('Measurific', 'linux', {}), '/var/lib/measurific')
  assert.equal(machineDataDirectory('X', 'linux', { INSPECTION_MACHINE_DATA_DIR: '/srv/x' }), '/srv/x')
  assert.equal(calibrationId(new Set(), new Date('2026-01-02T03:04:05.006Z')), '2026-01-02T03-04-05-006Z')
})
