# SPDX-License-Identifier: MIT
# Derived in part from gravi-toni/subpixel-edges commit a4f8f1a (2020).
# Copyright (c) 2020 Giuseppe Vitucci and Nicola Vitucci.
"""Threshold-seeded fast path for controlled telecentric silhouettes."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Literal

import cv2
import numpy as np
from numpy.typing import ArrayLike

from .edgepixel import EdgePixel, Edges
from .edges_iter0_seeded import h_edges_seeded, v_edges_seeded
from .seeded import (
    classify_candidates,
    expand_seed_band,
    gradient_signs,
)

Polarity = Literal["dark-on-light", "light-on-dark"]


@dataclass(slots=True)
class SilhouetteWorkspace:
    """Reusable full-frame mask that avoids allocation on every frame."""

    shape: tuple[int, int]
    mask: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or min(self.shape) < 11:
            raise ValueError("workspace shape must contain two dimensions >= 11")
        self.shape = (int(self.shape[0]), int(self.shape[1]))
        self.mask = np.empty(self.shape, dtype=np.uint8)


def detect_silhouette(
    image: ArrayLike,
    pae_threshold: float,
    *,
    silhouette_threshold: float,
    polarity: Polarity = "dark-on-light",
    search_radius: int = 3,
    min_area: float = 0.0,
    max_area: float | None = None,
    order: int = 2,
    smoothing: int = 0,
    workspace: SilhouetteWorkspace | None = None,
) -> Edges:
    """Locate PAE edges near thresholded telecentric-silhouette contours.

    Thresholding only supplies coarse candidate seeds. Positions and all other
    reported features are calculated from the original grayscale intensities
    using the same PAE equations and candidate predicates as :func:`detect`.
    ``smoothing=1`` selects the noise-compensated equations; larger values
    select iterative restoration. These modes currently use the full reference
    detector and then retain candidates in the seed band.

    ``search_radius`` must cover the distance between the threshold contour
    and the maximum-gradient pixel. Three pixels is appropriate for a sharply
    focused silhouette; validate larger values for wider optical blur.
    ``min_area`` and ``max_area`` filter dust or unrelated components while
    retaining exterior contours and holes that satisfy those limits.
    """
    source = _validate_image(image)
    pae_threshold = _validate_real("pae_threshold", pae_threshold, minimum=0.0)
    silhouette_threshold = _validate_real(
        "silhouette_threshold", silhouette_threshold
    )
    search_radius = _validate_integer(
        "search_radius", search_radius, minimum=0, maximum=32
    )
    order = _validate_integer("order", order, minimum=1, maximum=2)
    smoothing = _validate_integer("smoothing", smoothing, minimum=0, maximum=1000)
    min_area = _validate_real("min_area", min_area, minimum=0.0)
    if max_area is not None:
        max_area = _validate_real("max_area", max_area, minimum=min_area)
    if polarity not in ("dark-on-light", "light-on-dark"):
        raise ValueError("polarity must be 'dark-on-light' or 'light-on-dark'")

    if workspace is None:
        mask = np.empty(source.shape, dtype=np.uint8)
    else:
        if workspace.shape != source.shape:
            raise ValueError(
                f"workspace shape {workspace.shape} does not match image {source.shape}"
            )
        mask = workspace.mask

    comparison = cv2.CMP_LT if polarity == "dark-on-light" else cv2.CMP_GT
    cv2.compare(source, silhouette_threshold, comparison, mask)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )

    selected = []
    for contour in contours:
        area = abs(cv2.contourArea(contour))
        if area < min_area or (max_area is not None and area > max_area):
            continue
        selected.append(contour.reshape(-1, 2))
    if not selected:
        return _empty_edges()

    seeds = np.ascontiguousarray(np.concatenate(selected), dtype=np.int64)
    return _detect_seeded(source, seeds, pae_threshold, search_radius, order, smoothing)


def detect_seeded(
    image: ArrayLike,
    pae_threshold: float,
    *,
    seeds: ArrayLike,
    search_radius: int = 3,
    order: int = 2,
    smoothing: int = 0,
) -> Edges:
    """Locate PAE edges near supplied integer ``[x, y]`` seed pixels.

    This is the topology-neutral seeded detector used by higher-level image
    adapters whose coarse boundaries come from Canny or another segmenter.
    """
    source = _validate_image(image)
    pae_threshold = _validate_real("pae_threshold", pae_threshold, minimum=0.0)
    search_radius = _validate_integer("search_radius", search_radius, minimum=0, maximum=32)
    order = _validate_integer("order", order, minimum=1, maximum=2)
    smoothing = _validate_integer("smoothing", smoothing, minimum=0, maximum=1000)
    points = np.asarray(seeds)
    if points.ndim != 2 or points.shape[1] != 2 or points.dtype.kind not in "iu":
        raise ValueError("seeds must be an integer [N, 2] array")
    if len(points) == 0:
        return _empty_edges()
    if np.any(points < 0) or np.any(points[:, 0] >= source.shape[1]) or np.any(points[:, 1] >= source.shape[0]):
        raise ValueError("seeds must lie inside the image")
    return _detect_seeded(source, np.ascontiguousarray(points, dtype=np.int64),
                          pae_threshold, search_radius, order, smoothing)


def _detect_seeded(source, seeds, pae_threshold, search_radius, order, smoothing=0):
    expanded = expand_seed_band(seeds, source.shape[0], source.shape[1], search_radius)
    if smoothing:
        from . import detect
        edges = detect(source, pae_threshold, smoothing=smoothing, order=order)
        keep = np.isin(edges.position, expanded)
        return EdgePixel(**{name: getattr(edges, name)[keep] for name in
                            ('position', 'x', 'y', 'nx', 'ny', 'curv', 'i0', 'i1')})
    # Classifying before sorting shrinks the duplicate set dramatically: only
    # actual NMS maxima survive. np.unique then restores the deterministic
    # column-major order of the full reference detector.
    edges_y, edges_x = classify_candidates(source, expanded, pae_threshold)
    edges_y = np.unique(edges_y)
    edges_x = np.unique(edges_x)
    return _fit_selected_edges(source, edges_y, edges_x, order)


def _fit_selected_edges(image, edges_y, edges_x, order):
    rows, cols = image.shape
    if edges_y.size + edges_x.size == 0:
        return _empty_edges()
    ny_sign, nx_sign = gradient_signs(image, edges_y, edges_x)
    Ay, By, ay, by, cy = h_edges_seeded(image, rows, edges_y, order)
    Ax, Bx, ax, bx, cx = v_edges_seeded(image, rows, edges_x, order)
    Ay, By, ay, by, cy = (value.ravel("F") for value in (Ay, By, ay, by, cy))
    Ax, Bx, ax, bx, cx = (value.ravel("F") for value in (Ax, Bx, ax, bx, cx))

    result = EdgePixel()
    result.position = np.r_[edges_y, edges_x]
    result.x = np.r_[edges_y // rows, edges_x // rows - ax]
    result.y = np.r_[edges_y % rows - ay, edges_x % rows]
    result.nx = np.r_[
        np.sign(Ay - By) / np.sqrt(1 + by**2) * by,
        np.sign(Ax - Bx) / np.sqrt(1 + bx**2),
    ]
    result.ny = np.r_[
        np.sign(Ay - By) / np.sqrt(1 + by**2),
        np.sign(Ax - Bx) / np.sqrt(1 + bx**2) * bx,
    ]
    result.curv = np.r_[
        2 * cy * ny_sign / ((1 + by**2) ** 1.5),
        2 * cx * nx_sign / ((1 + bx**2) ** 1.5),
    ]
    result.i0 = np.r_[np.minimum(Ay, By), np.minimum(Ax, Bx)]
    result.i1 = np.r_[np.maximum(Ay, By), np.maximum(Ax, Bx)]

    invalid = np.union1d(
        np.union1d(np.where(result.x > cols), np.where(result.y > rows)),
        np.union1d(np.where(result.x < 0), np.where(result.y < 0)),
    )
    if invalid.size:
        for name in ("position", "x", "y", "nx", "ny", "curv", "i0", "i1"):
            setattr(result, name, np.delete(getattr(result, name), invalid))
    return result


def _empty_edges() -> Edges:
    return EdgePixel(
        position=np.empty(0, dtype=np.int64),
        x=np.empty(0, dtype=np.float64),
        y=np.empty(0, dtype=np.float64),
        nx=np.empty(0, dtype=np.float64),
        ny=np.empty(0, dtype=np.float64),
        curv=np.empty(0, dtype=np.float64),
        i0=np.empty(0, dtype=np.float64),
        i1=np.empty(0, dtype=np.float64),
    )


def _validate_image(image: ArrayLike) -> np.ndarray:
    source = np.asarray(image)
    if source.ndim != 2:
        raise ValueError(f"image must be a 2-D grayscale array; got {source.shape}")
    if min(source.shape) < 11:
        raise ValueError("image dimensions must both be at least 11 pixels")
    if source.dtype.kind not in "uif":
        raise TypeError(f"image dtype must be real numeric; got {source.dtype}")
    if source.dtype.kind == "f" and not np.isfinite(source).all():
        raise ValueError("image values must all be finite")
    return np.ascontiguousarray(source)


def _validate_real(name: str, value: float, *, minimum: float | None = None) -> float:
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or (minimum is not None and result < minimum):
        constraint = f" and >= {minimum}" if minimum is not None else ""
        raise ValueError(f"{name} must be finite{constraint}")
    return result


def _validate_integer(
    name: str, value: int, *, minimum: int, maximum: int
) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or result > maximum:
        raise ValueError(f"{name} must be {minimum}..{maximum}")
    return result
