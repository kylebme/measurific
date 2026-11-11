"""Measurement-only corner exclusion, with original contour identities intact."""

import numpy as np


def corner_exclusions(contours, radius=1.0):
    """Find localized turns and mask nearby vertices by contour arclength.

    Chords at two scales distinguish localized corners from gradual curvature.
    Interpolated positions are used only to estimate directions, never returned
    as measurement samples. The radius is a fixed policy, not a PSF estimate.
    """
    if (
        isinstance(radius, (bool, np.bool_))
        or not isinstance(radius, (int, float, np.integer, np.floating))
        or not np.isfinite(radius)
        or radius < 0
    ):
        raise ValueError("corner_exclusion_px must be a finite nonnegative number")
    radius = float(radius)
    excluded = np.zeros(len(contours.xy), dtype=bool)
    corners = []
    if radius:
        for path, closed in enumerate(contours.closed):
            q = contours.path(path)
            lengths = np.linalg.norm(np.diff(q, axis=0), axis=1)
            s = np.r_[0.0, np.cumsum(lengths)]
            total = s[-1] + (np.linalg.norm(q[-1] - q[0]) if closed else 0.0)
            if total < 16:
                continue  # Insufficient support for the two-scale test.
            source_s = np.r_[s, total] if closed else s
            source_q = np.vstack([q, q[0]]) if closed else q

            def at(offset):
                target = (s + offset) % total if closed else s + offset
                return np.column_stack(
                    [np.interp(target, source_s, source_q[:, axis]) for axis in (0, 1)]
                )

            def turn(inner, outer):
                before = at(-inner) - at(-outer)
                after = at(outer) - at(inner)
                cross = before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]
                return np.arctan2(cross, np.sum(before * after, axis=1))

            small, large = turn(2.0, 4.0), turn(4.0, 8.0)
            sharp = (
                (abs(small) >= np.radians(35))
                & (abs(large) >= np.radians(35))
                & (abs(small) >= 0.8 * abs(large))
                & (small * large > 0)
            )
            if not closed:
                sharp &= (s >= 8.0) & (s <= total - 8.0)
            candidates = np.flatnonzero(sharp)
            if not len(candidates):
                continue
            # Join nearby responses to one rounded/noisy corner. Handle a corner
            # straddling the stored origin of a closed ring in the same way.
            positions = s[candidates]
            if closed:
                gaps = np.diff(np.r_[positions, positions[0] + total])
                start = (int(np.argmax(gaps)) + 1) % len(positions)
                candidates = np.roll(candidates, -start)
                positions = s[candidates]
                positions = np.r_[
                    positions[0], positions[0] + np.cumsum(np.diff(positions) % total)
                ]
            groups = np.split(
                np.arange(len(candidates)), np.flatnonzero(np.diff(positions) > 2.0) + 1
            )
            for group in groups:
                center = float(
                    np.average(positions[group], weights=abs(small[candidates[group]]))
                )
                delta = abs(s - center)
                if closed:
                    delta = abs((s - center + total / 2) % total - total / 2)
                nearest = int(np.argmin(delta))
                lo = contours.offsets[path]
                excluded[lo : lo + len(q)] |= delta <= radius + 1e-9
                corners.append(
                    {
                        "path": path,
                        "nearest_vertex_index": int(lo + nearest),
                        "arclength_px": center % total if closed else center,
                        "observed_xy": q[nearest].tolist(),
                    }
                )
    # An edge is measurable only when both of its original endpoints survive.
    eligible = np.zeros(len(contours.xy), dtype=bool)
    for path, (lo, hi) in enumerate(zip(contours.offsets[:-1], contours.offsets[1:])):
        keep = ~excluded[lo:hi]
        eligible[lo:hi] = keep & np.roll(keep, -1)
        if not contours.closed[path]:
            eligible[hi - 1] = False
    return eligible, {
        "method": "fixed_arclength_corner_exclusion",
        "radius_px": radius,
        "blur_sigma_px": None,
        "corners": corners,
        "excluded_vertex_indices": np.flatnonzero(excluded).tolist(),
        "excluded_vertex_count": int(excluded.sum()),
    }
