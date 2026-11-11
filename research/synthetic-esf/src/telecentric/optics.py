"""Accepted radiance quadrature and locally varying intensity PSF proxies."""
from dataclasses import replace
import numpy as np
from scipy.stats import qmc
from scipy.special import j1
from scipy.ndimage import map_coordinates


def accepted_directions(source, optics):
    if source.kind == 'collimated' or source.half_angle_deg == 0 and source.kind != 'diffuse':
        return np.array([[0., 0., 1.]]), np.ones(1)
    # Uniform projected solid angle: cos(theta) dOmega = du dv, where u,v
    # are transverse direction cosines. Constant radiance is Lambertian.
    radius = optics.object_na
    if source.kind != 'diffuse':
        radius = min(radius, np.sin(np.deg2rad(source.half_angle_deg)))
    uv = qmc.Sobol(2, scramble=True, seed=source.seed).random_base2(int(np.log2(source.samples)))
    r = radius * np.sqrt(uv[:, 0])
    phi = 2*np.pi*uv[:, 1]
    z = np.sqrt(1-r*r)
    directions = np.column_stack((r*np.cos(phi)/z, r*np.sin(phi)/z, np.ones(len(r))))
    weights = np.ones(len(r))
    if source.kind == 'gaussian':
        exponent = -0.5*(np.arcsin(r)/np.deg2rad(source.gaussian_sigma_deg))**2
        weights = np.exp(exponent-exponent.max())
    return directions, weights/weights.sum()


def psf_kernel(optics, camera, supersample):
    if optics.psf == 'none':
        return np.ones((1, 1)), {'kind': 'none', 'support_px': 0}
    pitch_mm = camera.object_pitch_mm/supersample
    half = int(np.ceil(optics.support_px*supersample))
    axis_px = np.arange(-half, half+1)/supersample
    x, y = np.meshgrid(axis_px, axis_px)
    info = {'kind': optics.psf, 'support_px': half/supersample}
    if optics.psf in {'gaussian', 'mixture'}:
        angle = np.deg2rad(optics.angle_deg)
        u, v = np.cos(angle)*x+np.sin(angle)*y, -np.sin(angle)*x+np.cos(angle)*y
        sy = optics.sigma_px if optics.sigma_y_px is None else optics.sigma_y_px
        kernel = np.exp(-0.5*((u/optics.sigma_px)**2+(v/sy)**2))/(2*np.pi*optics.sigma_px*sy)
        if optics.psf == 'mixture':
            tx, ty = optics.tail_offset_px
            tail = np.exp(-((x-tx)**2+(y-ty)**2)/(2*optics.tail_sigma_px**2))/(2*np.pi*optics.tail_sigma_px**2)
            kernel = (1-optics.tail_weight)*kernel + optics.tail_weight*tail
        info['retained_mass_estimate'] = float(kernel.sum()/supersample**2)
    elif optics.psf == 'airy':
        # Incoherent circular-pupil intensity PSF, object-space units.
        scale = 2*np.pi*optics.object_na/(optics.wavelength_um*1e-3)
        v = scale*np.hypot(x, y)*camera.object_pitch_mm
        kernel = np.ones_like(v)
        np.divide(2*j1(v), v, out=kernel, where=v != 0)
        kernel = kernel**2 * scale**2/(4*np.pi)
        info['retained_mass_estimate'] = float(kernel.sum()*pitch_mm**2)
    else:
        data = np.load(optics.measured_path, allow_pickle=False)
        if data.ndim != 2 or any(s % 2 == 0 for s in data.shape) or not np.isfinite(data).all() or np.any(data < 0) or data.sum() <= 0:
            raise ValueError('Measured PSF must be finite, nonnegative, odd-sized 2D .npy with positive sum')
        step = optics.measured_pitch_um*1e-3
        coords = np.array([y*camera.object_pitch_mm/step+(data.shape[0]-1)/2, x*camera.object_pitch_mm/step+(data.shape[1]-1)/2])
        kernel = map_coordinates(data, coords, order=1, mode='constant', cval=0)
        info['retained_mass_estimate'] = float(kernel.sum()*pitch_mm**2/(data.sum()*step**2))
        import hashlib
        from pathlib import Path
        info['sha256'] = hashlib.sha256(Path(optics.measured_path).read_bytes()).hexdigest()
    if not np.isfinite(kernel).all() or kernel.sum() <= 0:
        raise ValueError('PSF has no sampled energy; increase support or check sample pitch')
    kernel /= kernel.sum()
    info['centroid_px'] = [float((kernel*x).sum()), float((kernel*y).sum())]
    info['note'] = 'Finite support normalized to unit energy; test support convergence. Intensity PSF is an approximation for partially coherent backlighting.'
    return kernel, info


def field_weight(node_index, optics, camera, xs, ys):
    """Bilinear partition of unity at output coordinates; independent of tiles."""
    nx, ny = optics.field_grid
    col, row = node_index % nx, node_index // nx
    wx = np.ones_like(xs) if nx == 1 else np.maximum(0, 1-np.abs(np.clip(xs/camera.width, 0, 1)*(nx-1)-col))
    wy = np.ones_like(ys) if ny == 1 else np.maximum(0, 1-np.abs(np.clip(ys/camera.height, 0, 1)*(ny-1)-row))
    return wy[:, None]*wx[None, :]


def field_kernels(optics, camera, supersample):
    if not optics.field_nodes:
        kernel, info = psf_kernel(optics, camera, supersample)
        return [kernel], {'mode': 'constant', **info}
    pairs = [psf_kernel(replace(optics, field_grid=(1, 1), field_nodes=(), **node), camera, supersample) for node in optics.field_nodes]
    half = max(kernel.shape[0]//2 for kernel, _ in pairs)
    kernels = []
    for kernel, _ in pairs:
        padding = half-kernel.shape[0]//2
        kernels.append(np.pad(kernel, padding))
    return kernels, dict(mode='field', grid=list(optics.field_grid), nodes=[info for _, info in pairs],
                         convention='output-position bilinear mixture of normalized kernels; DC-preserving local imaging approximation, not an energy-conserving point-source propagation solver')
