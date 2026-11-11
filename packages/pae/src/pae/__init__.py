# SPDX-License-Identifier: MIT
# Derived in part from gravi-toni/subpixel-edges commit a4f8f1a (2020).
# Copyright (c) 2020 Giuseppe Vitucci and Nicola Vitucci.
"""CPU implementation of PAE subpixel edge localization.

The numerical detector is derived from ``gravi-toni/subpixel-edges`` and the
MATLAB implementation by Trujillo-Pino et al. See ``NOTICE.md`` in the source
distribution for the complete provenance.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike

from .edgepixel import EdgePixel, Edges
from .final_detector_iter0 import main_iter0
from .final_detector_iter1 import main_iter1
from .final_detector_iterN import main_iterN

from .silhouette import SilhouetteWorkspace, detect_seeded, detect_silhouette

__all__ = [
    "EdgePixel",
    "Edges",
    "SilhouetteWorkspace",
    "detect",
    "detect_seeded",
    "detect_silhouette",
    "subpixel_edges",
    "warmup",
]
__version__ = "0.1.0"


def detect(
    image: ArrayLike,
    threshold: float,
    *,
    smoothing: int = 0,
    order: int = 2,
) -> Edges:
    """Locate grayscale-image edges with subpixel precision.

    Parameters
    ----------
    image:
        A finite, two-dimensional grayscale image. Integer and floating-point
        arrays are accepted; calculations use float64 to match the reference.
    threshold:
        Minimum gradient magnitude for an edge candidate, in the same
        intensity units as ``image`` (for example 25 for an 8-bit image).
    smoothing:
        ``0`` for clean images, ``1`` for the reference 3x3 low-noise path,
        or more than one for iterative high-noise reconstruction.
    order:
        ``1`` for locally straight edges or ``2`` for locally curved edges.

    Returns
    -------
    Edges
        One-dimensional arrays ``x``, ``y``, ``nx``, ``ny``, ``curv``,
        ``i0``, ``i1``, and ``position``. Coordinates are zero-based with x
        increasing by column and y increasing by row. Normals point toward
        increasing intensity; ``i0 <= i1``.

    Notes
    -----
    The first call compiles the per-edge kernels with Numba. Call
    :func:`warmup` during application startup when latency must be predictable.
    """
    array = _validate_image(image)
    threshold_value = _validate_threshold(threshold)
    smoothing_value = _validate_integer("smoothing", smoothing, minimum=0)
    order_value = _validate_integer("order", order, minimum=1, maximum=2)

    if smoothing_value == 0:
        return main_iter0(array, threshold_value, 0, order_value)
    if smoothing_value == 1:
        return main_iter1(array, threshold_value, 1, order_value)

    current = array
    edges = Edges()
    for _ in range(smoothing_value):
        edges, current = main_iterN(
            current, threshold_value, smoothing_value, order_value
        )
    return edges


def subpixel_edges(
    img: ArrayLike, threshold: float, iters: int = 0, order: int = 2
) -> Edges:
    """Compatibility spelling matching the upstream Python port."""
    return detect(img, threshold, smoothing=iters, order=order)


def warmup(
    *, include_iterative: bool = False, include_silhouette: bool = False
) -> None:
    """Compile the Numba kernels before latency-sensitive processing."""
    image = np.zeros((17, 17), dtype=np.float64)
    image[:, 9:] = 255.0
    detect(image, 25.0, smoothing=0, order=2)
    detect(image, 25.0, smoothing=1, order=2)
    if include_iterative:
        detect(image, 25.0, smoothing=2, order=2)
    if include_silhouette:
        detect_silhouette(
            image,
            25.0,
            silhouette_threshold=127.0,
            polarity="light-on-dark",
        )


def _validate_image(image: ArrayLike) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"image must be a 2-D grayscale array; got shape {array.shape}")
    if min(array.shape) < 11:
        raise ValueError("image dimensions must both be at least 11 pixels")
    if array.dtype.kind not in "uif":
        raise TypeError(f"image dtype must be real numeric; got {array.dtype}")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError("image values must all be finite")
    # The reference implementation performs its numerical work in float64.
    # The inherited kernels use MATLAB-compatible column-major linear indices.
    # Building float64 directly in Fortran order avoids several full-image
    # transposition copies in the latency-sensitive smoothing=0 path.
    return np.asarray(array, dtype=np.float64, order="F")


def _validate_threshold(threshold: float) -> float:
    if not isinstance(threshold, Real) or isinstance(threshold, (bool, np.bool_)):
        raise TypeError("threshold must be a real number")
    value = float(threshold)
    if not np.isfinite(value) or value < 0:
        raise ValueError("threshold must be finite and non-negative")
    return value


def _validate_integer(
    name: str, value: int, *, minimum: int, maximum: int | None = None
) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        bounds = f"{minimum}..{maximum}" if maximum is not None else f">={minimum}"
        raise ValueError(f"{name} must be {bounds}")
    return result
