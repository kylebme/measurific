"""Offline corner recovery; neither routine is a production extraction mode.

``repair_corners`` retains the earlier whole-polygon experiment. The newer
``recover_pae_corners`` fits raw pixels near PAE corners with locally estimated
Gaussian blur, multiple patch sizes, residual checks, and a competing bevel
model. Rejected neighborhoods retain their original PAE samples. Conditional
covariance and model-comparison thresholds are heuristics, not calibrated
accuracy guarantees. See corner-recovery-20260910/README.md for held-out results.
"""
import cv2
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
import numpy as np
from scipy.optimize import least_squares
from experimental_optical_fit import _polygon_model
(nodes, weights) = np.polynomial.legendre.leggauss(6)
nodes = (nodes + 1) / 2
weights = weights / 2

def fit_corner_pixels(image, center, before, after, sigma, noise, *, patch_radius=9, audit=None, check_chamfer=False):
    """Fit a local wedge; sigma=None estimates blur from the same raw pixels."""
    evidence = {} if audit is None else audit
    evidence['patch_radius'] = patch_radius
    def reject(reason):
        evidence['reason'] = reason
        return None
    a = before - center
    a /= np.linalg.norm(a)
    b = after - center
    b /= np.linalg.norm(b)
    theta = np.arctan2([a[1], b[1]], [a[0], b[0]])
    sign = np.sign(np.linalg.det(np.array([-a, b])))
    (yy, xx) = np.mgrid[-patch_radius:patch_radius+1, -patch_radius:patch_radius+1]
    xy = np.c_[xx.ravel(), yy.ravel()] + np.rint(center)
    xy = xy.astype(float)
    if np.any(xy.min(axis=0) < 0) or np.any(xy.max(axis=0) > np.array(image.shape[::-1]) - 1):
        return reject('patch_touches_border')
    observed = image[xy[:, 1].astype(int), xy[:, 0].astype(int)].astype(float)
    initial = np.r_[center, theta, np.quantile(observed, 0.1), np.quantile(observed, 0.9)]
    estimate_blur = sigma is None
    if estimate_blur:
        initial = np.r_[initial, 1.]
    previous = None
    cached = None

    def evaluate(p):
        nonlocal previous, cached
        if previous is None or not np.array_equal(previous, p):
            rays = np.c_[np.cos(p[2:4]), np.sin(p[2:4])] * 200
            verts = np.array([p[:2] + rays[0], p[:2], p[:2] + rays[1]])
            blur = p[6] if estimate_blur else sigma
            (value, js, r, c, d) = _polygon_model(xy, verts, np.array([1, 2, 0]), blur, nodes, weights)
            coverage = value + (1 if sign < 0 else 0)
            shape = np.zeros((len(xy), 6))
            np.add.at(shape, (r, c), d)
            jc = shape.reshape(-1, 3, 2).sum(axis=1)
            ja = shape[:, :2] @ np.array([-rays[0, 1], rays[0, 0]])
            jb = shape[:, 4:] @ np.array([-rays[1, 1], rays[1, 0]])
            jac = np.c_[jc, ja, jb] * (p[4] - p[5])
            jac = np.c_[jac, coverage, 1 - coverage]
            if estimate_blur:
                jac = np.c_[jac, js*(p[4]-p[5])]
            cached = (p[5] + (p[4] - p[5]) * coverage - observed, jac)
            previous = p.copy()
        return cached
    lower = np.r_[center - 3, theta - 0.3, 0, 0]
    upper = np.r_[center + 3, theta + 0.3, 255, 255]
    if estimate_blur:
        lower, upper = np.r_[lower, .15], np.r_[upper, 4.]
    initial = np.clip(initial, lower+1e-5, upper-1e-5)
    sol = least_squares(lambda p: evaluate(p)[0], initial, jac=lambda p: evaluate(p)[1], bounds=(lower, upper), max_nfev=60, x_scale='jac', loss='soft_l1', f_scale=max(0.5, noise), ftol=1e-07)
    (residual, jac) = evaluate(sol.x)
    rms = np.sqrt(np.mean(residual ** 2))
    evidence.update(rms_dn=float(rms), blur_sigma_px=float(sol.x[6] if estimate_blur else sigma),
                    iterations=int(sol.nfev))
    if not sol.success or rms > max(0.75, 1.6 * noise):
        return reject('nonconvergence' if not sol.success else 'incompatible_image_patch')
    # Mean RMS over a mostly flat patch can hide a wrong chamfer/corner model.
    # Test coherent local residuals against independent pixel-noise variance.
    from scipy.signal import convolve2d
    axis = np.arange(-2, 3, dtype=float)
    kernel = np.exp(-(axis[:, None]**2+axis[None, :]**2)/(2*.9**2))
    kernel /= kernel.sum()
    local = convolve2d(residual.reshape(2*patch_radius+1, -1), kernel, mode='valid')
    z = float(np.max(abs(local))/(max(noise, .5)*np.sqrt(np.sum(kernel**2))))
    evidence['maximum_coherent_residual_z'] = z
    if z > 4.5:
        return reject('structured_image_residual')
    if np.any(sol.active_mask) or sol.x[5]-sol.x[4] < 40:
        return reject('parameter_bound_or_low_contrast')
    weight = 1/np.sqrt(1+(residual/max(noise, .5))**2)
    covariance = np.linalg.pinv(jac.T @ (jac*weight[:, None])) * max(noise, 0.5) ** 2
    uncertainty = 3 * np.sqrt(np.linalg.eigvalsh(covariance[:2, :2]).max())
    if uncertainty > 0.15 or np.linalg.cond(jac) > 10000:
        evidence['three_sigma_position_px'] = float(uncertainty)
        return reject('weak_corner_localization')
    if check_chamfer:
        # A small covariance assumes a sharp vertex. Challenge that assumption:
        # can a bevel displaced 0.3 px from the intersection explain the pixels?
        # Refit the nuisance parameters; a fixed-parameter comparison is biased.
        def bevel_residual(p):
            directions = np.c_[np.cos(p[2:4]), np.sin(p[2:4])]
            cosine = np.sqrt(max(1e-4, (1+directions[0]@directions[1])/2))
            cut = .3/cosine
            vertices = p[:2]+directions[[0, 0, 1, 1]]*np.array([200., cut, cut, 200.])[:, None]
            value = _polygon_model(xy, vertices, np.array([1, 2, 3, 0]),
                                   p[6] if estimate_blur else sigma, nodes, weights)[0]
            coverage = value+(1 if sign < 0 else 0)
            return p[5]+(p[4]-p[5])*coverage-observed
        bevel = least_squares(bevel_residual, sol.x, bounds=(lower, upper),
                              max_nfev=40, x_scale='jac', loss='soft_l1',
                              f_scale=max(.5, noise), ftol=1e-7)
        evidence['bevel_cost_increase_noise_units'] = float((bevel.cost-sol.cost)/max(.5, noise)**2)
        if not bevel.success or evidence['bevel_cost_increase_noise_units'] < 4.5:
            return reject('sharp_corner_not_identifiable')
    evidence.update(reason='accepted', three_sigma_position_px=float(uncertainty))
    return (sol.x[:2], dict(evidence))

def repair_corners(q, image=None, sigma=1.0, noise=2.0):
    lengths = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)
    s = np.r_[0, np.cumsum(lengths[:-1])]
    L = lengths.sum()
    simple = cv2.approxPolyDP(q.astype(np.float32), 1.0, True)[:, 0].astype(float)
    incoming = simple - np.roll(simple, 1, axis=0)
    outgoing = np.roll(simple, -1, axis=0) - simple
    angle = np.arccos(np.clip(np.sum(incoming * outgoing, axis=1) / np.linalg.norm(incoming, axis=1) / np.linalg.norm(outgoing, axis=1), -1, 1))
    simple = simple[angle > 0.5]
    if not len(simple):
        return (q, 0)
    (_, ids) = cKDTree(q).query(simple)
    cs = np.sort(s[ids])
    gaps = (np.roll(cs, -1) - cs) % L
    start = (np.argmax(gaps) + 1) % len(cs)
    cs = np.roll(cs, -start)
    cs = np.r_[cs[0], cs[0] + np.cumsum(np.diff(cs) % L)]
    groups = np.split(cs, np.flatnonzero(np.diff(cs) > 7) + 1)
    cs = np.array([g.mean() % L for g in groups])
    cs.sort()
    (_, ids) = cKDTree(s[:, None]).query(cs[:, None])
    simple = q[ids]
    changes = []
    for (i, p) in enumerate(simple):
        delta = (s - cs[i] + L / 2) % L - L / 2
        limits = [min(12, (cs[i] - cs[i - 1]) % L * 0.65), min(12, (cs[(i + 1) % len(cs)] - cs[i]) % L * 0.65)]
        fits = []
        for (sign, limit) in zip([-1, 1], limits):
            pts = q[(delta * sign > 4) & (delta * sign < limit)]
            if len(pts) < 4:
                break
            center = pts.mean(axis=0)
            (_, _, v) = np.linalg.svd(pts - center, full_matrices=False)
            normal = v[1]
            if np.quantile(abs((pts - center) @ normal), 0.95) > 0.1:
                break
            fits.append((normal, center @ normal))
        if len(fits) != 2:
            continue
        normals = np.array([a for (a, b) in fits])
        offsets = np.array([b for (a, b) in fits])
        if abs(np.linalg.det(normals)) < 0.3:
            continue
        corner = np.linalg.solve(normals, offsets)
        if np.linalg.norm(corner - p) > 3:
            continue
        if image is not None:
            before = q[(delta < -4) & (delta > -12)].mean(axis=0)
            after = q[(delta > 4) & (delta < 12)].mean(axis=0)
            result = fit_corner_pixels(image, corner, before, after, sigma, noise)
            if result is None:
                continue
            corner = result[0]
        changes.append((cs[i], corner))
    keep = np.ones(len(q), bool)
    for (position, corner) in changes:
        delta = (s - position + L / 2) % L - L / 2
        keep[abs(delta) < 3.5] = False
    if not changes:
        return (q, 0)
    ss = np.r_[s[keep], [a for (a, b) in changes]]
    new = np.vstack([q[keep], [b for (a, b) in changes]])[np.argsort(ss)]
    if not Polygon(new).is_valid:
        return (q, 0)
    return (new, len(changes))


def recover_pae_corners(image, contours, diagnostics, *, polarity='dark-on-light'):
    """Experimental local correction; raw PAE and the production adapter are unchanged.

    Only accepted corner neighborhoods are replaced. Blur is estimated locally;
    no renderer parameters, CAD, primitive labels, or truth are supplied here.
    """
    from inspection_measurement import Contours
    canonical = image if polarity == 'dark-on-light' else 255-image
    f = canonical.astype(float)
    difference = f[:, :-2]-2*f[:, 1:-1]+f[:, 2:]
    noise = max(.5, float(np.median(abs(difference-np.median(difference)))/(.67449*np.sqrt(6))))
    result, accepted, rejected = [], [], []
    for path in range(len(contours.closed)):
        q = contours.path(path)
        lengths = np.linalg.norm(np.roll(q, -1, axis=0)-q, axis=1)
        s = np.r_[0, np.cumsum(lengths[:-1])]
        total = float(lengths.sum())
        simplified = cv2.approxPolyDP(q.astype(np.float32), 1., True)[:, 0].astype(float)
        incoming, outgoing = simplified-np.roll(simplified, 1, axis=0), np.roll(simplified, -1, axis=0)-simplified
        angles = np.arctan2(incoming[:, 0]*outgoing[:, 1]-incoming[:, 1]*outgoing[:, 0], (incoming*outgoing).sum(axis=1))
        sharp = abs(angles) > .5
        simplified, signs = simplified[sharp], np.sign(angles[sharp])
        if not len(simplified):
            result.append(q.copy())
            continue
        positions = s[cKDTree(q).query(simplified)[1]]
        order = np.argsort(positions)
        positions, signs = positions[order], signs[order]
        gaps = np.diff(np.r_[positions, positions[0]+total])
        shift = -(int(np.argmax(gaps))+1) % len(positions)
        positions, signs = np.roll(positions, shift), np.roll(signs, shift)
        positions = np.r_[positions[0], positions[0]+np.cumsum(np.diff(positions) % total)]
        starts = [0]
        for j in range(1, len(positions)):
            if signs[j] != signs[j-1] or positions[j]-positions[starts[-1]] > 7.:
                starts.append(j)
        groups = np.split(positions, starts[1:])
        centers = np.sort(np.array([g.mean() % total for g in groups]))
        changes = []
        for k, position in enumerate(centers):
            delta = (s-position+total/2) % total-total/2
            p = q[np.argmin(abs(delta))]
            audit = dict(path=path, original_xy=p.tolist(), arclength_px=float(position))
            preceding = (position-centers[k-1]) % total if len(centers) > 1 else total
            following = (centers[(k+1) % len(centers)]-position) % total if len(centers) > 1 else total
            limits = [min(15., preceding*.65), min(15., following*.65)]
            inner_limits = [min(4., preceding*.25), min(4., following*.25)]
            fits = []
            for direction, inner, limit in zip([-1., 1.], inner_limits, limits):
                support = q[(delta*direction > inner) & (delta*direction < limit)]
                if len(support) < 3 or np.linalg.norm(np.ptp(support, axis=0)) < 1.5:
                    break
                center = support.mean(axis=0)
                _, _, vectors = np.linalg.svd(support-center, full_matrices=False)
                normal = vectors[1]
                if np.quantile(abs((support-center)@normal), .95) > .25:
                    break
                fits.append((normal, float(center@normal), center))
            if len(fits) != 2:
                rejected.append(dict(audit, reason='insufficient_straight_support'))
                continue
            matrix = np.array([row[0] for row in fits])
            if abs(np.linalg.det(matrix)) < .3:
                rejected.append(dict(audit, reason='nearly_parallel_sides'))
                continue
            initial = np.linalg.solve(matrix, np.array([row[1] for row in fits]))
            if np.linalg.norm(initial-p) > 4.:
                rejected.append(dict(audit, reason='distant_line_intersection'))
                continue
            trials, candidates = [], []
            for radius in (3, 6, 9):
                trial = {}
                fit = fit_corner_pixels(canonical, initial, fits[0][2], fits[1][2], None, noise,
                                        patch_radius=radius, audit=trial, check_chamfer=True)
                trials.append(trial)
                if fit is not None:
                    candidates.append(fit)
            if not candidates:
                rejected.append(dict(audit, reason='image_fit_rejected', trials=trials))
                continue
            best = min(candidates, key=lambda item: item[1]['three_sigma_position_px'])
            if any(np.linalg.norm(other[0]-best[0]) > .15 for other in candidates):
                rejected.append(dict(audit, reason='patch_scale_disagreement', trials=trials))
                continue
            point, evidence = best
            radius = max(3.5, 3*evidence['blur_sigma_px'])
            if radius > min(limits)-1:
                rejected.append(dict(audit, reason='neighboring_corner_too_close', trials=trials))
                continue
            changes.append((position, point, radius, dict(audit, **evidence, xy=point.tolist(), replacement_radius_px=radius)))
        conflicts = set()
        for i, a in enumerate(changes):
            for j in range(i):
                b = changes[j]
                if abs((a[0]-b[0]+total/2) % total-total/2) <= a[2]+b[2]:
                    conflicts.update([i, j])
        rejected.extend(dict(changes[i][3], reason='overlapping_corner_regions') for i in sorted(conflicts))
        changes = [item for i, item in enumerate(changes) if i not in conflicts]
        keep = np.ones(len(q), bool)
        for position, _, radius, _ in changes:
            keep[abs((s-position+total/2) % total-total/2) < radius] = False
        if not changes:
            result.append(q.copy())
            continue
        order = np.argsort(np.r_[s[keep], [item[0] for item in changes]])
        fitted = np.vstack([q[keep], [item[1] for item in changes]])[order]
        if not Polygon(fitted).is_valid:
            result.append(q.copy())
            rejected.extend(dict(item[3], reason='ring_topology_rejected') for item in changes)
            continue
        result.append(fitted)
        accepted.extend(item[3] for item in changes)
    # Roll back corrections if they change containment or touch another ring.
    original_polygons, new_polygons = [Polygon(contours.path(i)) for i in range(len(result))], [Polygon(q) for q in result]
    valid = True
    for i in range(len(result)):
        for j in range(i):
            if (new_polygons[i].boundary.intersects(new_polygons[j].boundary) or
                new_polygons[i].contains(new_polygons[j]) != original_polygons[i].contains(original_polygons[j]) or
                new_polygons[j].contains(new_polygons[i]) != original_polygons[j].contains(original_polygons[i])):
                valid = False
    if not valid:
        rejected.extend(dict(item, reason='nesting_rejected') for item in accepted)
        accepted = []
        result = [contours.path(i).copy() for i in range(len(result))]
    fitted = Contours(np.concatenate(result), np.r_[0, np.cumsum([len(q) for q in result])])
    info = dict(diagnostics)
    info['method'] = 'experimental_pae_with_local_gaussian_corners'
    info.pop('pae_source_indices', None)
    info['rings'] = [dict(ring, point_count=len(result[i])) for i, ring in enumerate(info['rings'])]
    info['corner_recovery'] = dict(noise_sigma_dn=noise, accepted=accepted, rejected=rejected,
                                   original_vertex_count=len(contours.xy), final_vertex_count=len(fitted.xy),
                                   coordinate_source='original_pae_except_image_fitted_corner_neighborhoods')
    return fitted, info
