"""Memory-bounded rendering, native pixel integration, and independent labels."""
from dataclasses import dataclass
import numpy as np
from scipy.signal import fftconvolve
from .optics import accepted_directions, field_kernels, field_weight


@dataclass
class RenderResult:
    irradiance: np.ndarray
    geometric_coverage: np.ndarray
    center_mask: np.ndarray
    metadata: dict


def validate_roi(camera, roi):
    roi = (0, 0, camera.width, camera.height) if roi is None else tuple(roi)
    if len(roi) != 4 or any(isinstance(v, bool) or not isinstance(v, int) for v in roi):
        raise ValueError('ROI is four integers: x, y, width, height')
    x, y, w, h = roi
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > camera.width or y+h > camera.height:
        raise ValueError('ROI must be inside the sensor')
    return roi


def render(scene, config, roi=None, progress=None):
    camera, sampling = config.camera, config.sampling
    roi = validate_roi(camera, roi)
    rx, ry, width, height = roi
    ss = sampling.supersample
    kernels, psf_info = field_kernels(config.optics, camera, ss)
    pad = kernels[0].shape[0]//2
    directions, weights = accepted_directions(config.illumination, config.optics)
    irradiance = np.empty((height, width), np.float32)
    coverage = np.empty_like(irradiance)
    center_mask = np.empty((height, width), bool)
    tile = sampling.tile_px
    total = ((height+tile-1)//tile)*((width+tile-1)//tile)
    completed = 0
    for ty in range(0, height, tile):
        for tx in range(0, width, tile):
            h, w = min(tile, height-ty), min(tile, width-tx)
            # Sample centers on a GLOBAL grid. Each tile evaluates true geometry
            # beyond its border by the complete PSF support, including off-FOV.
            xs = rx+tx+(np.arange(-pad, w*ss+pad)+0.5)/ss
            ys = ry+ty+(np.arange(-pad, h*ss+pad)+0.5)/ss
            shape = (len(ys), len(xs))
            shadow = np.empty(len(xs)*len(ys), np.float64)
            occupancy = np.empty_like(shadow)
            for start in range(0, len(shadow), sampling.ray_chunk):
                ids = np.arange(start, min(start+sampling.ray_chunk, len(shadow)))
                px, py = xs[ids % len(xs)], ys[ids // len(xs)]
                xy = camera.pixel_to_world(np.column_stack((px, py)))
                origins = np.column_stack((xy, np.full(len(ids), camera.focus_z_mm)))
                blocked = scene.blocked(origins, np.array([0., 0., 1.]))
                occupancy[ids] = blocked
                transmission = np.zeros(len(ids))
                for direction, weight in zip(directions, weights):
                    hits = blocked if np.array_equal(direction, [0, 0, 1]) else scene.blocked(origins, direction)
                    transmission += weight*(~hits)
                gradient = 1 + config.gradient_xy[0]*(px/camera.width-0.5) + config.gradient_xy[1]*(py/camera.height-0.5)
                shadow[ids] = config.foreground + (config.background-config.foreground)*transmission*np.maximum(gradient, 0)
            high = shadow.reshape(shape)
            optical = np.zeros((h*ss, w*ss))
            for node_index, kernel in enumerate(kernels):
                weight = field_weight(node_index, config.optics, camera, xs[pad:pad+w*ss], ys[pad:pad+h*ss]) if config.optics.field_nodes else 1.0
                if np.any(weight):
                    optical += weight*(fftconvolve(high, kernel, mode='valid') if pad else high)
            irradiance[ty:ty+h, tx:tx+w] = np.maximum(optical.reshape(h, ss, w, ss).mean(axis=(1, 3)), 0)
            occ = occupancy.reshape(shape)[pad:pad+h*ss, pad:pad+w*ss]
            coverage[ty:ty+h, tx:tx+w] = occ.reshape(h, ss, w, ss).mean(axis=(1, 3))
            px, py = np.meshgrid(rx+tx+np.arange(w)+0.5, ry+ty+np.arange(h)+0.5)
            xy = camera.pixel_to_world(np.column_stack((px.ravel(), py.ravel())))
            origins = np.column_stack((xy, np.full(len(xy), camera.focus_z_mm)))
            center_mask[ty:ty+h, tx:tx+w] = scene.blocked(origins, np.array([0., 0., 1.])).reshape(h, w)
            completed += 1
            if progress:
                progress(completed, total)
    meta = dict(config=config.to_dict(), roi_xywh=list(roi), psf=psf_info, angular_samples=len(weights), angular_effective_samples=float(1/(weights@weights)), object_pitch_mm=camera.object_pitch_mm,
                coordinates='pixel edges at integers, centers at i+0.5; full sensor coordinates; x right, y down; world mm; z along rays',
                model='opaque geometric radiance through a fixed-focus telecentric pupil; inverse distortion before pixel integration; output-indexed local intensity PSF',
                radiometry='bright_e_per_ms specifies unobstructed detected flux; angular weights normalized after pupil acceptance',
                truth='physical and ideal-pixel geometry plus distorted geometric image labels; masks/coverage use distorted geometry; observed optical edges are estimator outputs')
    return RenderResult(irradiance, coverage, center_mask, meta)
