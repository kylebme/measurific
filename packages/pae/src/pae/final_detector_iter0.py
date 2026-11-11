# SPDX-License-Identifier: MIT
# Derived from gravi-toni/subpixel-edges commit a4f8f1a (2020).
# Copyright (c) 2020 Giuseppe Vitucci and Nicola Vitucci.

import numpy as np

from pae.candidates import find_candidates
from pae.edgepixel import EdgePixel
from pae.edges_iter0 import h_edges, v_edges


def main_iter0(F, threshold, iters, order):
    ep = EdgePixel()

    rows, cols = np.shape(F)

    Fx, Fy, edges_y, edges_x = find_candidates(F, threshold)

    F = F.ravel('F')
    Fx = Fx.ravel('F')
    Fy = Fy.ravel('F')

    ny = np.ones((np.shape(edges_y)[0], 1))
    ny[Fy[edges_y] < 0] = -1
    nx = np.ones((np.shape(edges_x)[0], 1))
    nx[Fx[edges_x] < 0] = -1

    Ay, By, ay, by, cy = h_edges(F, rows, Fx, Fy, edges_y, order)
    Ax, Bx, ax, bx, cx = v_edges(F, rows, Fx, Fy, edges_x, order)

    Ay, By, ay, by, cy = (v.ravel('F') for v in (Ay, By, ay, by, cy))
    Ax, Bx, ax, bx, cx = (v.ravel('F') for v in (Ax, Bx, ax, bx, cx))
    ny = ny.ravel('F')
    nx = nx.ravel('F')

    ep.position = np.r_[edges_y, edges_x]
    ep.x = np.r_[edges_y // rows, edges_x // rows - ax]
    ep.y = np.r_[edges_y % rows - ay, edges_x % rows]

    ep.nx = np.r_[
        np.sign(Ay - By) / np.sqrt(1 + by ** 2) * by,
        np.sign(Ax - Bx) / np.sqrt(1 + bx ** 2),
    ]
    ep.ny = np.r_[
        np.sign(Ay - By) / np.sqrt(1 + by ** 2),
        np.sign(Ax - Bx) / np.sqrt(1 + bx ** 2) * bx,
    ]
    ep.curv = np.r_[
        2 * cy * ny / ((1 + by ** 2) ** 1.5),
        2 * cx * nx / ((1 + bx ** 2) ** 1.5),
    ]
    ep.i0 = np.r_[np.minimum(Ay, By), np.minimum(Ax, Bx)]
    ep.i1 = np.r_[np.maximum(Ay, By), np.maximum(Ax, Bx)]

    # # erase elements outside the image size
    index_to_erase1 = np.union1d(np.where(ep.x > cols), np.where(ep.y > rows))
    index_to_erase2 = np.union1d(np.where(ep.x < 0), np.where(ep.y < 0))

    index_to_erase = np.union1d(index_to_erase1, index_to_erase2)

    ep.x = np.delete(ep.x, index_to_erase)
    ep.y = np.delete(ep.y, index_to_erase)
    ep.nx = np.delete(ep.nx, index_to_erase)
    ep.ny = np.delete(ep.ny, index_to_erase)
    ep.i0 = np.delete(ep.i0, index_to_erase)
    ep.i1 = np.delete(ep.i1, index_to_erase)
    ep.position = np.delete(ep.position, index_to_erase)
    ep.curv = np.delete(ep.curv, index_to_erase)

    return ep
