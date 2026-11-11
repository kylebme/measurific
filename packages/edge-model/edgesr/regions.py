"""Torch-free minimum-area region filter for per-cell hi-res occupancy canvases.

Shared by the torch inference (cfi_infer) and the ONNX runtime (runtime)."""

import numpy as np

from . import SCALE


def filter_small_regions_exact(ins, cy, cx, cell, min_area, scale=SCALE):
    """min_area: scalar (both classes) or (min_object_area, min_hole_area) in px^2."""
    obj_a, hole_a = (min_area, min_area) if np.isscalar(min_area) else min_area
    """Remove closed connected regions (object specks and holes) smaller than min_area
    (source px^2) from per-cell hi-res occupancy canvases, like the adapter's per-contour
    `min_area`. ins: (N, cell*s, cell*s) bool canvases of the active cells at (cy, cx).
    Components are merged across adjacent active cells (union-find on shared borders,
    incl. diagonal contacts for 8-connected object regions); a component touching an
    inactive neighbour or the image frame continues into the large fill region and is
    kept. All canvases are labelled in one cv2 call per class via a separator mosaic.
    Returns the filtered canvases (numpy bool)."""
    import cv2
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    ins = np.asarray(ins, bool)
    N, S, _ = ins.shape
    cy, cx = np.asarray(cy, np.int64), np.asarray(cx, np.int64)
    # neighbour index lookup via a dense cell grid
    gh, gw = int(cy.max()) + 2, int(cx.max()) + 2
    grid = np.full((gh + 1, gw + 1), -1, np.int64)
    grid[cy + 1, cx + 1] = np.arange(N)
    right, left = grid[cy + 1, cx + 2], grid[cy + 1, cx]
    down, up = grid[cy + 2, cx + 1], grid[cy, cx + 1]
    G = int(np.ceil(np.sqrt(N)))
    R = int(np.ceil(N / G))
    T = S + 1
    out = ins.copy()
    for value, conn in ((True, 8), (False, 4)):
        min_px = (obj_a if value else hole_a) * scale * scale
        if min_px <= 0:
            continue
        tiles = np.zeros((R * G, T, T), np.uint8)
        tiles[:N, :S, :S] = ins == value
        mosaic = tiles.reshape(R, G, T, T).transpose(0, 2, 1, 3).reshape(R * T, G * T)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mosaic, connectivity=conn)
        if n <= 1:
            continue
        labels = lab.reshape(R, T, G, T).transpose(0, 2, 1, 3)   # view (R, G, T, T)
        labels = labels.reshape(R * G, T, T)[:N, :S, :S]
        labels = labels - 1                                       # -1 = not this class (int32)
        area = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
        total = n - 1
        a_idx, b_idx = [], []
        for nbr, mine, theirs in ((right, (slice(None), slice(None), -1), (slice(None), slice(None), 0)),
                                  (down, (slice(None), -1, slice(None)), (slice(None), 0, slice(None)))):
            has = nbr >= 0
            if not has.any():
                continue
            la = labels[mine][has]                 # slice the border first, then select cells
            lb = labels[theirs][nbr[has]]
            pairs = [(la, lb)]
            if conn == 8:
                pairs += [(la[:, :-1], lb[:, 1:]), (la[:, 1:], lb[:, :-1])]
            for x, y in pairs:
                m = (x >= 0) & (y >= 0)
                a_idx.append(x[m])
                b_idx.append(y[m])
        open_ = np.zeros(total, bool)
        for nbr, edge in ((left, (slice(None), 0)), (right, (slice(None), -1)),
                          (up, (0, slice(None))), (down, (-1, slice(None)))):
            e = labels[(slice(None),) + edge][nbr < 0].ravel()
            open_[e[e >= 0]] = True
        a_all = np.concatenate(a_idx) if a_idx else np.zeros(0, np.int64)
        b_all = np.concatenate(b_idx) if b_idx else np.zeros(0, np.int64)
        g = coo_matrix((np.ones(len(a_all)), (a_all, b_all)), shape=(total, total))
        _, comp = connected_components(g, directed=False)
        comp_area = np.bincount(comp, weights=area)
        comp_open = np.bincount(comp, weights=open_.astype(float)) > 0
        kill = ((comp_area < min_px) & ~comp_open)[comp]         # per mosaic label
        if kill.any():
            # flip only inside the bounding boxes of removed components
            for k in np.nonzero(kill)[0]:
                x, y, w, h = stats[k + 1, :4]
                ti, ty0 = divmod(y, T)
                tj, tx0 = divmod(x, T)
                idx = ti * G + tj
                sub = labels[idx, ty0:ty0 + h, tx0:tx0 + w]
                out[idx, ty0:ty0 + h, tx0:tx0 + w][sub == k] = not value
    return out
