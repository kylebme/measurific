#!/usr/bin/env node
// Builds the relocatable Python runtime bundled with the app (build/python-runtime).
//
// A python-build-standalone interpreter (via uv) gets the locked third-party
// requirements plus the pipeline packages installed from source with
// --no-deps, so training-only dependencies such as torch never enter the app.
// Standard-library and package test suites are pruned and bytecode is
// precompiled, because the installed app directory is read-only.
//
//   node scripts/build-python-runtime.mjs [--python 3.13.12]
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const guiRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const repoRoot = path.resolve(guiRoot, '..')
const output = path.join(guiRoot, 'build', 'python-runtime')
const windows = process.platform === 'win32'
// Pinned so every platform ships the same interpreter; bump deliberately.
const version = process.argv.includes('--python') ? process.argv[process.argv.indexOf('--python') + 1] : '3.13.12'
const pipelinePackages = ['edge-model', 'measurement', 'contour-model', 'pae'].map((name) => path.join(repoRoot, 'packages', name))

function run(command, args, options = {}) {
  console.log(`$ ${command} ${args.join(' ')}`)
  return execFileSync(command, args, { stdio: 'inherit', ...options })
}

function remove(target) {
  fs.rmSync(target, { recursive: true, force: true })
}

function pythonExecutable(directory) {
  return windows ? path.join(directory, 'python.exe') : path.join(directory, 'bin', 'python3')
}

function standardLibrary(directory) {
  if (windows) return path.join(directory, 'Lib')
  const lib = path.join(directory, 'lib')
  const name = fs.readdirSync(lib).find((entry) => /^python3\.\d+$/.test(entry))
  return path.join(lib, name)
}

// Test suites and development-only files are dead weight in a shipped app.
function prune(directory) {
  const stdlib = standardLibrary(directory)
  for (const name of ['test', 'idlelib', 'tkinter', 'turtledemo', 'ensurepip', 'lib2to3', 'pydoc_data', 'EXTERNALLY-MANAGED']) {
    remove(path.join(stdlib, name))
  }
  for (const name of ['BUILD', 'include', 'share', 'tcl']) remove(path.join(directory, name))
  // Tcl/Tk only serve tkinter, which is removed above.
  for (const folder of [path.join(directory, 'lib'), path.join(directory, 'DLLs')]) {
    if (!fs.existsSync(folder)) continue
    for (const entry of fs.readdirSync(folder)) {
      if (/^(lib)?(tcl|tk|itcl|thread)\d|^_tkinter|^(tcl|tk)\d.*\.dll$/i.test(entry)) remove(path.join(folder, entry))
    }
  }
  const sitePackages = path.join(stdlib, 'site-packages')
  const walk = (current) => {
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const full = path.join(current, entry.name)
      if (!entry.isDirectory()) continue
      if (entry.name === '__pycache__' || entry.name === 'tests') remove(full)
      else walk(full)
    }
  }
  walk(stdlib)
  // pip and its launchers are not needed at runtime; uv installs the packages.
  for (const entry of fs.readdirSync(sitePackages)) {
    if (/^pip(-|$)/.test(entry)) remove(path.join(sitePackages, entry))
  }
}

function directorySize(directory) {
  let total = 0
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const full = path.join(directory, entry.name)
    if (entry.isDirectory()) total += directorySize(full)
    else if (entry.isFile()) total += fs.statSync(full).size
  }
  return total
}

const staging = fs.mkdtempSync(path.join(os.tmpdir(), 'inspector-python-'))
try {
  run('uv', ['python', 'install', version, '--install-dir', staging, '--no-bin'])
  // uv also adds a minor-version symlink next to the real directory.
  const installed = fs.readdirSync(staging, { withFileTypes: true }).find((entry) => entry.name.startsWith('cpython-') && entry.isDirectory())?.name
  if (!installed) throw new Error(`uv did not install CPython ${version} into ${staging}`)
  remove(output)
  fs.mkdirSync(path.dirname(output), { recursive: true })
  fs.cpSync(path.join(staging, installed), output, { recursive: true, verbatimSymlinks: true })
} finally {
  remove(staging)
}

const python = pythonExecutable(output)
remove(path.join(standardLibrary(output), 'EXTERNALLY-MANAGED'))
const install = ['pip', 'install', '--python', python, '--no-cache', '--link-mode', 'copy']
run('uv', [...install, '--require-hashes', '-r', path.join(guiRoot, 'python', 'runtime', 'requirements.lock')])
run('uv', [...install, '--no-deps', ...pipelinePackages])
prune(output)
// Precompile with hashes that are never re-checked against mtimes: the files
// are read-only once installed, and timestamps change when the app is copied.
run(python, ['-m', 'compileall', '-q', '-j', '0', '--invalidation-mode', 'unchecked-hash', standardLibrary(output)])
run(python, ['-c', 'import numpy, scipy, shapely, cv2, PIL, onnxruntime, numba, pae, inspection_measurement.pae_edge, contour_model.onnx_runtime, edgesr.runtime, edgesr.sdf_runtime; print("runtime imports ok")'], {
  env: { ...process.env, PYTHONNOUSERSITE: '1', PYTHONDONTWRITEBYTECODE: '1' },
})
console.log(`Python runtime: ${output} (${Math.round(directorySize(output) / 1e6)} MB)`)
