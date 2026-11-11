"""SDF edge inference with the ordered-ring contract used by the image adapter."""
import time
from pathlib import Path

import cv2
import numpy as np


def field_rings(sdf, scale=16):
    """Trace the bilinear zero level on a sparse subpixel lattice.

    Only quads near a sign change need a dense canvas. The extra quad around them
    keeps samples continuous across canvas borders. SDF samples sit at source pixel
    centers, unlike CFI samples at subpixel centers; compensate that origin after
    using the shared occupancy tracer.
    """
    from edgesr.runtime import trace_rings
    s = np.asarray(sdf, np.float32)
    inside = s >= 0
    mixed = np.zeros(s.shape, np.uint8)
    corners = (inside[:-1, :-1], inside[:-1, 1:], inside[1:, :-1], inside[1:, 1:])
    mixed[:-1, :-1] = np.logical_or.reduce(corners) & ~np.logical_and.reduce(corners)
    active = cv2.dilate(mixed, np.ones((3, 3), np.uint8))
    active[-1] = 0
    active[:, -1] = 0
    y, x = np.nonzero(active)
    u = np.arange(scale, dtype=np.float32) / scale
    a, b, c, d = (v[y, x][:, None, None] for v in (s[:-1, :-1], s[:-1, 1:], s[1:, :-1], s[1:, 1:]))
    top = a + (b - a) * u[None, None, :]
    bottom = c + (d - c) * u[None, None, :]
    canvas = top + (bottom - top) * u[None, :, None]
    result = trace_rings({'canvas': canvas, 'cy': y, 'cx': x, 'mask': inside}, 1, scale)
    result['rings'] = [q + (0.5 - 0.5 / scale) for q in result['rings']]
    return result


class SDFEdgeModel:
    """ONNX or development torch SDF model, interchangeable with EdgeModel.rings."""

    def __init__(self, path, device='cpu', threads=None):
        from edgesr.runtime import select_providers
        self.path = Path(path)
        if self.path.is_dir():
            from edgesr.sdf_runtime import SDFRuntime
            self.runtime = SDFRuntime(self.path, threads=threads, providers=select_providers(device))
            self.format = self.runtime.meta['format']
            self.providers = self.runtime.session.get_providers()
            self.provenance = self.runtime.meta.get('provenance', {})
            self.defaults = self.runtime.min_area
        else:
            import torch
            from edgesr.sdf_infer import SDFPredictor
            if device not in ('cpu', 'gpu'):
                raise ValueError('Compute device must be cpu or gpu')
            target = 'cuda' if device == 'gpu' and torch.cuda.is_available() else 'cpu'
            self.runtime = SDFPredictor(self.path, coarse=self.path.parent / 'coarse_v2.pt', device=target,
                                        amp=False)
            self.format, self.providers, self.provenance = 'edgesr-sdf-torch-v1', [target], {}
            self.defaults = self.runtime.min_area

    def describe(self):
        return {'format': self.format, 'directory': str(self.path), 'providers': self.providers,
                'source_checkpoint_sha256': self.provenance.get('source_checkpoint_sha256')}

    def rings(self, image, max_value=None, valid_mask=None, min_object_area=None, min_hole_area=None,
              apron=16):
        from edgesr.runtime import TRUST_MARGIN, backlit_frame, nest_rings, reading
        started = time.perf_counter()
        img = reading(image, max_value)
        valid = np.ones(img.shape, bool) if valid_mask is None else np.asarray(valid_mask, bool)
        if valid.shape != img.shape or not valid.any():
            raise ValueError('valid_mask must match the image and contain pixels')
        framed, backlight, gain = backlit_frame(img, valid, apron)
        sdf = self.runtime.field(framed)['sdf'].copy()
        if not np.isfinite(sdf).all():
            raise ValueError('SDF model returned non-finite distances')
        trusted = np.pad(valid.astype(np.uint8), apron)
        trusted = cv2.dilate(trusted, np.ones((2 * TRUST_MARGIN + 1,) * 2, np.uint8)).astype(bool)
        sdf[~trusted] = -np.maximum(np.abs(sdf[~trusted]), 1e-3)
        # A negative outer row guarantees every ring closes within the backlight apron.
        sdf[[0, -1], :] = -8
        sdf[:, [0, -1]] = -8
        min_area = tuple(float(default if value is None else value)
                         for default, value in zip(self.defaults, (min_object_area, min_hole_area)))
        t = time.perf_counter()
        traced = field_rings(sdf)
        nested = nest_rings([q - apron for q in traced['rings']], traced['is_hole'], min_area)
        times = {**self.runtime.times, 'trace': time.perf_counter() - t}
        return {**nested, 'lattice_crossings': traced['crossings'], 'apron': apron, 'min_area': min_area,
                'backlight': backlight, 'gain': gain, 'stage_seconds': times,
                'seconds': time.perf_counter() - started}
