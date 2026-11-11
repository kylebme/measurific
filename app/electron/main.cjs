const { app, BrowserWindow, dialog, ipcMain, shell } = require('electron')
const { spawn } = require('node:child_process')
const { readFile, writeFile } = require('node:fs/promises')
const path = require('node:path')
const { CameraService } = require('./camera.cjs')
const { PersistentState, machineDataDirectory, workerSettingKeys } = require('./state.cjs')

const root = path.resolve(__dirname, '..', '..')
const guiRoot = path.resolve(__dirname, '..')
const appName = app.isPackaged ? 'Measurific' : 'Measurific Dev'
let worker
let workerConfig
let nextId = 1
let cameraService
let state

// Must run before anything reads userData. Development builds keep their own
// folder so experiments never touch an installed app's settings.
if (process.env.INSPECTION_DATA_DIR) app.setPath('userData', path.resolve(process.env.INSPECTION_DATA_DIR))
else if (!app.isPackaged) app.setPath('userData', path.join(app.getPath('appData'), appName))

function broadcast(channel, ...args) {
  for (const window of BrowserWindow.getAllWindows()) {
    if (!window.isDestroyed()) window.webContents.send(channel, ...args)
  }
}

// Packaged builds carry a relocatable Python runtime, the worker, the ONNX
// models and demo images as extraResources, outside app.asar.
function runtimePaths() {
  const windows = process.platform === 'win32'
  if (app.isPackaged) {
    const resources = process.resourcesPath
    return {
      python: process.env.INSPECTION_PYTHON || (windows ? path.join(resources, 'python', 'python.exe') : path.join(resources, 'python', 'bin', 'python3')),
      worker: path.join(resources, 'worker', 'worker.py'),
      model: path.join(resources, 'models', 'contour'),
      edgeModel: path.join(resources, 'models', 'edge'),
      demo: path.join(resources, 'demo-data'),
      root: null,
    }
  }
  const paths = {
    python: process.env.INSPECTION_PYTHON || (windows ? path.join(guiRoot, '.venv', 'Scripts', 'python.exe') : path.join(guiRoot, '.venv', 'bin', 'python')),
    worker: path.join(guiRoot, 'python', 'worker.py'),
    model: path.join(root, 'packages', 'contour-model', 'release-v6', 'onnx'),
    edgeModel: path.join(root, 'packages', 'edge-model', 'models', 'sdf_v5-onnx'),
    demo: path.join(guiRoot, 'demo-data'),
    root,
  }
  // Development only: try another exported ONNX folder (or a .pt checkpoint) before bundling it.
  if (process.env.INSPECTION_CHECKPOINT) paths.model = path.resolve(process.env.INSPECTION_CHECKPOINT)
  if (process.env.INSPECTION_EDGE_MODEL) paths.edgeModel = path.resolve(process.env.INSPECTION_EDGE_MODEL)
  return paths
}

function desiredWorkerConfig() {
  const settings = state.effectiveSettings()
  return Object.fromEntries(workerSettingKeys.map((key) => [key, settings[key] ?? null]))
}

function stopWorker() {
  if (!worker) return
  const stopping = worker
  worker = undefined
  stopping.kill()
}

function ensureWorker() {
  if (worker) return
  const paths = runtimePaths()
  const config = desiredWorkerConfig()
  const args = [paths.worker, '--device', config.computeDevice, '--edge-method', config.edgeMethod,
    '--model', paths.model, '--edge-model', paths.edgeModel]
  if (paths.root) args.push('--root', paths.root)
  const env = { ...process.env }
  if (app.isPackaged) {
    // The bundled interpreter must not see a user's own Python installation.
    delete env.PYTHONHOME
    delete env.PYTHONPATH
    Object.assign(env, { PYTHONNOUSERSITE: '1', PYTHONDONTWRITEBYTECODE: '1' })
  }
  const child = spawn(paths.python, args, { cwd: path.dirname(paths.worker), env, stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true })
  // Requests belong to the process that received them; a replaced worker's
  // exit must not reject requests sent to its successor.
  const pending = new Map()
  child.pending = pending
  worker = child
  workerConfig = config
  let buffered = ''
  child.stdout.setEncoding('utf8')
  child.stdout.on('data', (chunk) => {
    buffered += chunk
    let newline
    while ((newline = buffered.indexOf('\n')) >= 0) {
      const line = buffered.slice(0, newline)
      buffered = buffered.slice(newline + 1)
      if (!line.trim()) continue
      try {
        const response = JSON.parse(line)
        const slot = pending.get(response.id)
        if (slot) {
          pending.delete(response.id)
          response.error ? slot.reject(new Error(response.error)) : slot.resolve(response.result)
        }
      } catch (error) {
        console.error('Invalid inspection worker response', line, error)
      }
    }
  })
  child.stderr.on('data', (chunk) => console.error(`[inspection-python] ${chunk}`))
  const stopped = (error) => {
    for (const slot of pending.values()) slot.reject(error)
    pending.clear()
    if (worker === child) worker = undefined
  }
  child.on('error', (error) => stopped(new Error(`Inspection worker could not start (${paths.python}): ${error.message}`)))
  child.stdin.on('error', (error) => stopped(new Error(`Inspection worker input failed: ${error.message}`)))
  child.on('exit', (code, signal) => stopped(new Error(`Inspection worker stopped (${signal || `exit ${code}`})`)))
}

function callWorker(command, payload = {}) {
  ensureWorker()
  const id = nextId++
  return new Promise((resolve, reject) => {
    worker.pending.set(id, { resolve, reject })
    worker.stdin.write(`${JSON.stringify({ id, command, ...payload })}\n`)
  })
}

function workerIsStale() {
  return Boolean(worker) && JSON.stringify(workerConfig) !== JSON.stringify(desiredWorkerConfig())
}

async function analyze(filePath, validMask) {
  // Compute settings take effect with the next analysis, which discards
  // the worker's per-image state anyway.
  if (workerIsStale()) stopWorker()
  const result = await callWorker('analyze', { path: filePath, validMask })
  const bytes = await readFile(filePath)
  const ext = path.extname(filePath).toLowerCase()
  const mime = ext === '.jpg' || ext === '.jpeg' ? 'image/jpeg' : ext === '.bmp' ? 'image/bmp' : 'image/png'
  return { ...result, imageUrl: `data:${mime};base64,${bytes.toString('base64')}` }
}

function publishState() {
  const described = state.describe()
  broadcast('inspection:state', described)
  return described
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1600,
    height: 1000,
    minWidth: 1100,
    minHeight: 700,
    backgroundColor: '#f7f7f8',
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    webPreferences: { preload: path.join(__dirname, 'preload.cjs'), contextIsolation: true, nodeIntegration: false },
  })
  win.maximize()
  if (process.env.VITE_DEV_SERVER_URL) win.loadURL(process.env.VITE_DEV_SERVER_URL)
  else win.loadFile(path.join(guiRoot, 'dist', 'index.html'))
}

ipcMain.handle('inspection:choose-image', async () => {
  const lastDirectory = state.describe().preferences.values.lastImageDirectory
  const result = await dialog.showOpenDialog({
    title: 'Open inspection image',
    ...(lastDirectory ? { defaultPath: lastDirectory } : {}),
    properties: ['openFile'],
    filters: [{ name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'] }],
  })
  if (result.canceled) return null
  state.updatePreferences({ lastImageDirectory: path.dirname(result.filePaths[0]) })
  return result.filePaths[0]
})
ipcMain.handle('inspection:analyze', (_event, filePath, validMask) => analyze(filePath, validMask))
ipcMain.handle('inspection:relationship', (_event, request) => callWorker('relationship', { request }))
ipcMain.handle('inspection:set-corner-exclusion', (_event, radius, measurements) => callWorker('set_corner_exclusion', { radius, measurements }))
ipcMain.handle('inspection:redefine-geometry', (_event, request) => callWorker('redefine_geometry', { request }))
ipcMain.handle('inspection:redefine-compound', (_event, request) => callWorker('redefine_compound', { request }))
ipcMain.handle('inspection:remove-compound', (_event, request) => callWorker('remove_compound', { request }))
ipcMain.handle('inspection:default-sample', () => path.join(runtimePaths().demo, 'drilled_plate.png'))
ipcMain.handle('inspection:camera-state', () => cameraService.publicState())
ipcMain.handle('inspection:camera-discover', () => cameraService.discover())
ipcMain.handle('inspection:camera-start', (_event, deviceId) => cameraService.start(deviceId))
ipcMain.handle('inspection:camera-roi', (_event, roi) => cameraService.setRoi(roi))
ipcMain.handle('inspection:camera-focus-reset', () => cameraService.resetFocus())
ipcMain.handle('inspection:camera-pause', (_event, roi) => cameraService.pause(roi))
ipcMain.handle('inspection:camera-control', (_event, name, value) => cameraService.setControl(name, value))

ipcMain.handle('inspection:state', () => state.describe())
ipcMain.handle('inspection:update-settings', async (_event, patch) => {
  state.updateSettings(patch)
  const described = publishState()
  const channel = state.effectiveSettings().cameraChannel
  if (cameraService.channel !== channel) await cameraService.setChannel(channel)
  return described
})
ipcMain.handle('inspection:update-preferences', (_event, patch) => { state.updatePreferences(patch); return publishState() })
ipcMain.handle('inspection:calibrations', () => state.listCalibrations())
ipcMain.handle('inspection:calibration', (_event, id) => state.getCalibration(id))
ipcMain.handle('inspection:save-calibration', (_event, calibration) => state.saveCalibration(calibration))
ipcMain.handle('inspection:delete-calibration', (_event, id) => state.deleteCalibration(id))
ipcMain.handle('inspection:worker-health', async () => ({
  ...(await callWorker('health')), pendingRestart: workerIsStale(), appVersion: app.getVersion(),
}))
ipcMain.handle('inspection:open-folder', async (_event, which) => {
  const { calibration, userDirectory } = state.describe()
  const error = await shell.openPath(which === 'calibration' ? calibration.directory : userDirectory)
  if (error) throw new Error(error)
})
ipcMain.handle('inspection:export-state', async () => {
  const result = await dialog.showSaveDialog({
    title: 'Export settings and calibrations',
    defaultPath: `measurific-${new Date().toISOString().slice(0, 10)}.json`,
    filters: [{ name: 'JSON', extensions: ['json'] }],
  })
  if (result.canceled || !result.filePath) return null
  await writeFile(result.filePath, `${JSON.stringify(state.exportBundle(), null, 2)}\n`)
  return result.filePath
})
ipcMain.handle('inspection:import-state', async () => {
  const result = await dialog.showOpenDialog({
    title: 'Import settings and calibrations',
    properties: ['openFile'],
    filters: [{ name: 'JSON', extensions: ['json'] }],
  })
  if (result.canceled) return null
  const summary = state.importBundle(JSON.parse(await readFile(result.filePaths[0], 'utf8')))
  publishState()
  return summary
})

app.whenReady().then(async () => {
  const { default: Store } = await import('electron-store')
  state = new PersistentState({
    Store,
    userDirectory: app.getPath('userData'),
    machineDirectory: machineDataDirectory(appName),
    appVersion: app.getVersion(),
  })
  for (const warning of state.warnings) console.warn(warning)
  cameraService = new CameraService({ app, guiRoot, projectRoot: root, send: broadcast, channel: state.effectiveSettings().cameraChannel })
  createWindow()
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow() })
})
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit() })
app.on('before-quit', () => {
  stopWorker()
  if (cameraService) void cameraService.close()
})
