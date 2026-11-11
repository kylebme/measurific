# SPDX-License-Identifier: MIT
# Derived in part from gravi-toni/subpixel-edges commit a4f8f1a (2020).
# Copyright (c) 2020 Giuseppe Vitucci and Nicola Vitucci.
"""Fused full-frame derivative and non-maximum-suppression pass."""

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def find_candidates(F, threshold):
    """Return reference derivatives and candidates in column-major order.

    This is algebraically identical to the NumPy expressions in the upstream
    port. Fusing their temporary arrays is important at multi-megapixel sizes:
    the reference formulation otherwise allocates a gradient image, two masks,
    and several absolute-value/comparison arrays.
    """
    rows, cols = F.shape
    Fx = np.zeros_like(F)
    Fy = np.zeros_like(F)

    # Column-major traversal matches the input layout and the MATLAB indexing
    # used by the inherited per-edge kernels.
    for x in prange(1, cols - 1):
        for y in range(1, rows - 1):
            Fx[y, x] = 0.5 * (F[y, x + 1] - F[y, x - 1])
            Fy[y, x] = 0.5 * (F[y + 1, x] - F[y - 1, x])

    count_y = np.zeros(cols, dtype=np.int64)
    for x in prange(2, cols - 2):
        count = 0
        for y in range(5, rows - 5):
            fx = Fx[y, x]
            fy = Fy[y, x]
            afy = abs(fy)
            if (
                np.sqrt(fx * fx + fy * fy) > threshold
                and afy >= abs(fx)
                and afy >= abs(Fy[y - 1, x])
                and afy > abs(Fy[y + 1, x])
            ):
                count += 1
        count_y[x] = count

    count_x = np.zeros(cols, dtype=np.int64)
    for x in prange(5, cols - 5):
        count = 0
        for y in range(2, rows - 2):
            fx = Fx[y, x]
            fy = Fy[y, x]
            afx = abs(fx)
            if (
                np.sqrt(fx * fx + fy * fy) > threshold
                and afx > abs(fy)
                and afx >= abs(Fx[y, x - 1])
                and afx > abs(Fx[y, x + 1])
            ):
                count += 1
        count_x[x] = count

    offset_y = np.empty(cols, dtype=np.int64)
    offset_x = np.empty(cols, dtype=np.int64)
    total_y = 0
    total_x = 0
    for x in range(cols):
        offset_y[x] = total_y
        offset_x[x] = total_x
        total_y += count_y[x]
        total_x += count_x[x]

    edges_y = np.empty(total_y, dtype=np.int64)
    edges_x = np.empty(total_x, dtype=np.int64)
    for x in prange(2, cols - 2):
        out = offset_y[x]
        for y in range(5, rows - 5):
            fx = Fx[y, x]
            fy = Fy[y, x]
            afy = abs(fy)
            if (
                np.sqrt(fx * fx + fy * fy) > threshold
                and afy >= abs(fx)
                and afy >= abs(Fy[y - 1, x])
                and afy > abs(Fy[y + 1, x])
            ):
                edges_y[out] = x * rows + y
                out += 1

    for x in prange(5, cols - 5):
        out = offset_x[x]
        for y in range(2, rows - 2):
            fx = Fx[y, x]
            fy = Fy[y, x]
            afx = abs(fx)
            if (
                np.sqrt(fx * fx + fy * fy) > threshold
                and afx > abs(fy)
                and afx >= abs(Fx[y, x - 1])
                and afx > abs(Fx[y, x + 1])
            ):
                edges_x[out] = x * rows + y
                out += 1

    return Fx, Fy, edges_y, edges_x
