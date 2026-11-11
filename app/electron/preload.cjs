const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('inspection', {
  chooseImage: () => ipcRenderer.invoke('inspection:choose-image'),
  analyze: (path, validMask) => ipcRenderer.invoke('inspection:analyze', path, validMask),
  measureRelationship: (request) => ipcRenderer.invoke('inspection:relationship', request),
  setCornerExclusion: (radius, measurements) => ipcRenderer.invoke('inspection:set-corner-exclusion', radius, measurements),
  redefineGeometry: (request) => ipcRenderer.invoke('inspection:redefine-geometry', request),
  redefineCompound: (request) => ipcRenderer.invoke('inspection:redefine-compound', request),
  removeCompound: (request) => ipcRenderer.invoke('inspection:remove-compound', request),
  defaultSample: () => ipcRenderer.invoke('inspection:default-sample'),
  cameraState: () => ipcRenderer.invoke('inspection:camera-state'),
  discoverCameras: () => ipcRenderer.invoke('inspection:camera-discover'),
  startCamera: (deviceId) => ipcRenderer.invoke('inspection:camera-start', deviceId),
  setCameraRoi: (roi) => ipcRenderer.invoke('inspection:camera-roi', roi),
  resetCameraFocus: () => ipcRenderer.invoke('inspection:camera-focus-reset'),
  pauseCamera: (roi) => ipcRenderer.invoke('inspection:camera-pause', roi),
  setCameraControl: (name, value) => ipcRenderer.invoke('inspection:camera-control', name, value),
  getState: () => ipcRenderer.invoke('inspection:state'),
  updateSettings: (patch) => ipcRenderer.invoke('inspection:update-settings', patch),
  updatePreferences: (patch) => ipcRenderer.invoke('inspection:update-preferences', patch),
  listCalibrations: () => ipcRenderer.invoke('inspection:calibrations'),
  getCalibration: (id) => ipcRenderer.invoke('inspection:calibration', id),
  saveCalibration: (calibration) => ipcRenderer.invoke('inspection:save-calibration', calibration),
  deleteCalibration: (id) => ipcRenderer.invoke('inspection:delete-calibration', id),
  workerHealth: () => ipcRenderer.invoke('inspection:worker-health'),
  openDataFolder: (which) => ipcRenderer.invoke('inspection:open-folder', which),
  exportState: () => ipcRenderer.invoke('inspection:export-state'),
  importState: () => ipcRenderer.invoke('inspection:import-state'),
  onState: (callback) => {
    const listener = (_event, state) => callback(state)
    ipcRenderer.on('inspection:state', listener)
    return () => ipcRenderer.removeListener('inspection:state', listener)
  },
  onCameraState: (callback) => {
    const listener = (_event, state) => callback(state)
    ipcRenderer.on('inspection:camera-state', listener)
    return () => ipcRenderer.removeListener('inspection:camera-state', listener)
  },
  onCameraFocus: (callback) => {
    const listener = (_event, focus) => callback(focus)
    ipcRenderer.on('inspection:camera-focus', listener)
    return () => ipcRenderer.removeListener('inspection:camera-focus', listener)
  },
  onCameraFrame: (callback) => {
    const listener = (_event, meta, pixels) => callback(meta, pixels)
    ipcRenderer.on('inspection:camera-frame', listener)
    return () => ipcRenderer.removeListener('inspection:camera-frame', listener)
  },
})
