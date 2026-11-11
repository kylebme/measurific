# SPDX-License-Identifier: MIT
# Derived in part from gravi-toni/subpixel-edges commit a4f8f1a (2020).
# Copyright (c) 2020 Giuseppe Vitucci and Nicola Vitucci.
"""Reference-compatible PAE candidate selection near coarse contour seeds."""

import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def _fx(image, y, x):
    return 0.5 * (float(image[y, x + 1]) - float(image[y, x - 1]))


@njit(cache=True, inline="always")
def _fy(image, y, x):
    return 0.5 * (float(image[y + 1, x]) - float(image[y - 1, x]))


@njit(cache=True)
def expand_seed_band(seeds, rows, cols, radius):
    """Expand contour coordinates into an in-bounds square search band."""
    width = 2 * radius + 1
    candidates = np.empty(seeds.shape[0] * width * width, dtype=np.int64)
    count = 0
    for k in range(seeds.shape[0]):
        seed_x = seeds[k, 0]
        seed_y = seeds[k, 1]
        for dx in range(-radius, radius + 1):
            x = seed_x + dx
            if x < 1 or x >= cols - 1:
                continue
            for dy in range(-radius, radius + 1):
                y = seed_y + dy
                if y < 1 or y >= rows - 1:
                    continue
                candidates[count] = x * rows + y
                count += 1
    return candidates[:count]


@njit(cache=True)
def classify_candidates(image, candidates, threshold):
    """Apply the full detector's exact orientation and NMS predicates."""
    rows, cols = image.shape
    edges_y = np.empty(candidates.size, dtype=np.int64)
    edges_x = np.empty(candidates.size, dtype=np.int64)
    count_y = 0
    count_x = 0

    for edge in candidates:
        x = edge // rows
        y = edge - x * rows
        fx = _fx(image, y, x)
        fy = _fy(image, y, x)
        magnitude = np.sqrt(fx * fx + fy * fy)

        if y >= 5 and y < rows - 5 and x >= 2 and x < cols - 2:
            afy = abs(fy)
            if (
                magnitude > threshold
                and afy >= abs(fx)
                and afy >= abs(_fy(image, y - 1, x))
                and afy > abs(_fy(image, y + 1, x))
            ):
                edges_y[count_y] = edge
                count_y += 1

        if y >= 2 and y < rows - 2 and x >= 5 and x < cols - 5:
            afx = abs(fx)
            if (
                magnitude > threshold
                and afx > abs(fy)
                and afx >= abs(_fx(image, y, x - 1))
                and afx > abs(_fx(image, y, x + 1))
            ):
                edges_x[count_x] = edge
                count_x += 1

    return edges_y[:count_y], edges_x[:count_x]


@njit(cache=True)
def gradient_signs(image, edges_y, edges_x):
    """Return reference derivative signs used to orient curvature."""
    rows = image.shape[0]
    ny = np.ones(edges_y.size, dtype=np.float64)
    nx = np.ones(edges_x.size, dtype=np.float64)
    for k in range(edges_y.size):
        edge = edges_y[k]
        x = edge // rows
        y = edge - x * rows
        if _fy(image, y, x) < 0:
            ny[k] = -1.0
    for k in range(edges_x.size):
        edge = edges_x[k]
        x = edge // rows
        y = edge - x * rows
        if _fx(image, y, x) < 0:
            nx[k] = -1.0
    return ny, nx
