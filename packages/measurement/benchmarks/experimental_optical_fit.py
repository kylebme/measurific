"""EXPERIMENTAL benchmark only: image-supported silhouette fitting with a Gaussian optical forward model.

The boundary integral integrates the PSF over finite square sensor pixels. Its
shape derivative is another boundary integral, giving a sparse analytic
Jacobian without rasterizing candidate polygons during optimization. This is a
separate measurement mode; the reference PAE detector is unchanged.
"""
from dataclasses import dataclass
import math
import numpy as np
import cv2
from numba import njit
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, hstack, vstack, eye

@njit(cache=True)
def _pixel_kernel(t, s):
    a = (t + 0.5) / s
    b = (t - 0.5) / s
    pa = 0.5 * math.erfc(-a / math.sqrt(2))
    pb = 0.5 * math.erfc(-b / math.sqrt(2))
    da = math.exp(-a * a / 2) / math.sqrt(2 * math.pi)
    db = math.exp(-b * b / 2) / math.sqrt(2 * math.pi)
    return ((t + 0.5) * pa - (t - 0.5) * pb + s * (da - db), pa - pb, da - db, (-a * da + b * db) / s)

@njit(cache=True)
def _polygon_model(xy, verts, ends, sigma, nodes, weights):
    n = len(xy)
    nv = len(verts)
    out = np.zeros(n)
    js = np.zeros(n)
    rows = [0]
    cols = [0]
    values = [0.0]
    for k in range(nv):
        a = verts[k]
        j = ends[k]
        b = verts[j]
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        steps = max(1, int(math.ceil(math.sqrt(dx * dx + dy * dy) / (2 * sigma))))
        for p in range(n):
            (x, y) = xy[p]
            if y < min(a[1], b[1]) - 8 * sigma - 1 or y > max(a[1], b[1]) + 8 * sigma + 1 or x > max(a[0], b[0]) + 8 * sigma + 1:
                continue
            if x < min(a[0], b[0]) - 8 * sigma - 1:
                (fa, ga, fsa, gsa) = _pixel_kernel(a[1] - y, sigma)
                (fb, gb, fsb, gsb) = _pixel_kernel(b[1] - y, sigma)
                out[p] += fb - fa
                js[p] += fsb - fsa
                continue
            jax = 0.0
            jay = 0.0
            jbx = 0.0
            jby = 0.0
            for st in range(steps):
                y0 = a[1] + dy * st / steps
                y1 = a[1] + dy * (st + 1) / steps
                x0 = a[0] + dx * st / steps
                x1 = a[0] + dx * (st + 1) / steps
                if y < min(y0, y1) - 8 * sigma - 1 or y > max(y0, y1) + 8 * sigma + 1 or x > max(x0, x1) + 8 * sigma + 1:
                    continue
                for g in range(len(nodes)):
                    t = (st + nodes[g]) / steps
                    w = weights[g] / steps
                    xx = a[0] + dx * t - x
                    yy = a[1] + dy * t - y
                    (fx, gx, fsx, gsx) = _pixel_kernel(xx, sigma)
                    (fy, gy, fsy, gsy) = _pixel_kernel(yy, sigma)
                    out[p] += w * dy * fx * gy
                    js[p] += w * dy * (fsx * gy + fx * gsy)
                    v = w * gx * gy
                    jax += v * dy * (1 - t)
                    jay -= v * dx * (1 - t)
                    jbx += v * dy * t
                    jby -= v * dx * t
            for (col, val) in ((2 * k, jax), (2 * k + 1, jay), (2 * j, jbx), (2 * j + 1, jby)):
                rows.append(p)
                cols.append(col)
                values.append(val)
    return (out, js, np.array(rows), np.array(cols), np.array(values))



@dataclass(frozen=True)
class SilhouetteFit:
    """Closed fitted polygon rings in image coordinates, with quality evidence."""

    rings: tuple[np.ndarray, ...]
    depths: tuple[int, ...]
    diagnostics: dict


def _initial_contours(image, threshold, min_area):
    smoothed = cv2.GaussianBlur(image, (0, 0), 1.)
    if threshold is None:
        threshold, mask = cv2.threshold(smoothed, 0, 255,
                                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    else:
        mask = (smoothed < threshold).astype(np.uint8)*255
    # Check the raw border too: smoothing must not hide clipped material.
    border = np.r_[image[0], image[-1], image[:, 0], image[:, -1]]
    if np.any(border < threshold):
        raise ValueError('Material touches image border; closed topology is not observable')
    contours, tree = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if tree is None:
        raise ValueError('No complete silhouette contours')
    selected = [i for i, q in enumerate(contours)
                if len(q) >= 3 and abs(cv2.contourArea(q)) >= min_area]
    if not selected:
        raise ValueError('No complete silhouette contours')
    depths = []
    for i in selected:
        parent = int(tree[0, i, 3])
        depth = 0
        while parent >= 0:
            if parent not in selected:
                raise ValueError('Area filtering removed a containing silhouette ring')
            depth += 1
            parent = int(tree[0, parent, 3])
        depths.append(depth)
    return [contours[i] for i in selected], tuple(depths), float(threshold), mask


def _knots(contours):
    vertices, ends, slices = [], [], []
    for contour in contours:
        # Remove raster stair steps and blur-rounded corners before fitting.
        # Subdivision then supplies degrees of freedom for curved/form-error edges.
        q = cv2.approxPolyDP(contour, 2., True)[:, 0].astype(float)
        if len(q) < 3:
            q = cv2.approxPolyDP(contour, .5, True)[:, 0].astype(float)
        if len(q) < 3:
            raise ValueError('Silhouette is too small to fit a closed boundary')
        ring = []
        for a, b in zip(q, np.roll(q, -1, axis=0)):
            length = np.linalg.norm(b-a)
            count = max(1, int(np.ceil(length/(8. if length > 30. else 3.))))
            ring.extend(a+np.arange(count)[:, None]/count*(b-a))
        start = len(vertices)
        vertices.extend(ring)
        stop = len(vertices)
        ends.extend(np.r_[np.arange(start+1, stop), start])
        slices.append(slice(start, stop))
    return np.asarray(vertices), np.asarray(ends), slices


def _smooth_indices(vertices, ends, slices):
    result = []
    for section in slices:
        for k in range(section.start, section.stop):
            previous = section.stop-1 if k == section.start else k-1
            following = ends[k]
            v0, v1 = vertices[k]-vertices[previous], vertices[following]-vertices[k]
            d0, d1 = np.linalg.norm(v0), np.linalg.norm(v1)
            angle = np.arccos(np.clip(v0@v1/d0/d1, -1, 1))
            if angle > .5 or max(d0, d1) > 12:
                continue
            result.append((previous, k, following))
    return np.asarray(result, dtype=int).reshape(-1, 3)


def _smooth_penalty(parameters, indices):
    """Normal curvature penalty, invariant to knot spacing along straight runs."""
    q = parameters[:-9].reshape(-1, 2)[indices]
    v0, v1 = q[:, 1]-q[:, 0], q[:, 2]-q[:, 1]
    d0 = np.maximum(np.linalg.norm(v0, axis=1), 1e-8)
    d1 = np.maximum(np.linalg.norm(v1, axis=1), 1e-8)
    denominator = d0+d1
    cross = v0[:, 0]*v1[:, 1]-v0[:, 1]*v1[:, 0]
    cross_grad = np.stack([np.c_[-v1[:, 1], v1[:, 0]],
                           np.c_[v1[:, 1]+v0[:, 1], -v1[:, 0]-v0[:, 0]],
                           np.c_[-v0[:, 1], v0[:, 0]]], axis=1)
    d_grad = np.stack([-v0/d0[:, None], v0/d0[:, None]-v1/d1[:, None],
                       v1/d1[:, None]], axis=1)
    gradient = 30*(cross_grad/denominator[:, None, None]-
                   cross[:, None, None]*d_grad/denominator[:, None, None]**2)
    columns = (2*indices[:, :, None]+np.arange(2)).ravel()
    rows = np.repeat(np.arange(len(indices)), 6)
    return 30*cross/denominator, csr_matrix((gradient.ravel(), (rows, columns)),
                                           shape=(len(indices), len(parameters)))


def fit_silhouette(image, *, threshold=None, polarity='dark-on-light',
                   min_area=8., max_iterations=100):
    """Fit closed silhouette geometry directly to original uint8 intensities.

    A smoothed threshold supplies topology only. The measurement fits polygon
    knots, Gaussian PSF sigma (0.2–4 px), and smooth foreground/background
    illumination jointly with a robust loss. Pixel area is integrated explicitly.
    Output vertices are model estimates, not original PAE observations. No CAD,
    primitive labels, ground-truth contours, or sidecars are used.

    The optical model assumes an isolated, high-contrast silhouette with a
    spatially constant isotropic Gaussian PSF. Unsupported fits and topology
    changes raise ValueError; diagnostics report residuals and estimated blur.
    Accuracy depends on noise, feature size, and optical model agreement.
    """
    from pae.silhouette import _validate_image, _validate_integer, _validate_real
    image = _validate_image(image)
    if image.dtype != np.uint8:
        raise ValueError('Silhouette fitting requires a grayscale uint8 image')
    min_area = _validate_real('min_area', min_area, minimum=0.)
    max_iterations = _validate_integer('max_iterations', max_iterations, minimum=1, maximum=500)
    if polarity not in ('dark-on-light', 'light-on-dark'):
        raise ValueError('Unsupported image polarity')
    if threshold is not None:
        threshold = _validate_real('threshold', threshold, minimum=0.)
        if threshold > 255:
            raise ValueError('threshold must be in [0,255]')
    if polarity == 'light-on-dark':
        image = 255-image
        threshold = None if threshold is None else 255-threshold
    contours, depths, level, mask = _initial_contours(image, threshold, min_area)
    vertices, ends, sections = _knots(contours)
    nv = len(vertices)
    edge_mask = np.zeros_like(image)
    cv2.drawContours(edge_mask, contours, -1, 255, 1)
    band = cv2.dilate(edge_mask, np.ones((15, 15), np.uint8)) > 0
    y, x = np.nonzero(band)
    xy = np.c_[x, y].astype(float)
    observed = image[y, x].astype(float)
    # Low-order fields accommodate illumination gradients and vignetting.
    u, v = (x-(image.shape[1]-1)/2)/max(image.shape), (y-(image.shape[0]-1)/2)/max(image.shape)
    basis = np.c_[np.ones(len(x)), u, v, u*u+v*v]
    dark, bright = np.median(image[mask > 0]), np.median(image[mask == 0])
    if bright-dark < 20:
        raise ValueError('Insufficient silhouette contrast')
    initial = np.r_[vertices.ravel(), 1., dark, 0., 0., 0., bright, 0., 0., 0.]
    smooth_indices = _smooth_indices(vertices, ends, sections)
    anchor = eye(2*nv, len(initial), format='csr')*.3
    nodes, weights = np.polynomial.legendre.leggauss(6)
    nodes, weights = (nodes+1)/2, weights/2
    reference_tangent = vertices[ends]-vertices
    reference_tangent /= np.linalg.norm(reference_tangent, axis=1)[:, None]
    order_rows = np.repeat(np.arange(nv), 4)
    order_cols = np.c_[2*np.arange(nv), 2*np.arange(nv)+1, 2*ends, 2*ends+1].ravel()
    last, cached = None, None

    def evaluate(parameters):
        nonlocal last, cached
        if last is None or not np.array_equal(last, parameters):
            occupancy, sigma_jac, rows, cols, values = _polygon_model(
                xy, parameters[:2*nv].reshape(-1, 2), ends, parameters[2*nv], nodes, weights)
            # OpenCV traverses exteriors clockwise; holes have opposite winding.
            occupancy = -occupancy
            shape_jac = hstack([
                csr_matrix((-values, (rows, cols)), shape=(len(xy), 2*nv)),
                csr_matrix(-sigma_jac[:, None])], format='csr')
            foreground = basis@parameters[2*nv+1:2*nv+5]
            background = basis@parameters[2*nv+5:]
            contrast = foreground-background
            prediction = background+contrast*occupancy
            jacobian = hstack([
                shape_jac.multiply(contrast[:, None]),
                csr_matrix(basis*occupancy[:, None]),
                csr_matrix(basis*(1-occupancy[:, None]))], format='csr')
            smooth_residual, smooth_jacobian = _smooth_penalty(parameters, smooth_indices)
            geometry = parameters[:2*nv].reshape(-1, 2)
            progression = np.sum((geometry[ends]-geometry)*reference_tangent, axis=1)
            active = progression < .01
            order_residual = 1000*np.minimum(progression-.01, 0.)
            order_values = 1000*np.c_[-reference_tangent, reference_tangent]*active[:, None]
            order_jacobian = csr_matrix((order_values.ravel(), (order_rows, order_cols)), shape=(nv, len(parameters)))
            cached = (np.r_[prediction-observed, anchor@(parameters-initial), smooth_residual, order_residual],
                      vstack([jacobian, anchor, smooth_jacobian, order_jacobian], format='csr'))
            last = parameters.copy()
        return cached

    # Smooth knots are observable only in their normal direction. Allowing
    # independent tangential motion makes the inverse problem rank deficient,
    # particularly on clean straight edges. Corners retain both coordinates.
    tr, tc, tv = [], [], []
    parameter = 0
    for section in sections:
        for k in range(section.start, section.stop):
            previous = section.stop-1 if k == section.start else k-1
            v0 = vertices[k]-vertices[previous]
            v1 = vertices[ends[k]]-vertices[k]
            v0 /= np.linalg.norm(v0)
            v1 /= np.linalg.norm(v1)
            angle = np.arccos(np.clip(v0@v1, -1, 1))
            if angle > .1:
                for axis in range(2):
                    tr.append(2*k+axis); tc.append(parameter); tv.append(1.)
                    parameter += 1
            else:
                tangent = v0+v1
                tangent /= np.linalg.norm(tangent)
                for axis, value in enumerate([-tangent[1], tangent[0]]):
                    tr.append(2*k+axis); tc.append(parameter); tv.append(value)
                parameter += 1
    geometry_count = parameter
    for k in range(9):
        tr.append(2*nv+k); tc.append(parameter); tv.append(1.)
        parameter += 1
    transform = csr_matrix((tv, (tr, tc)), shape=(len(initial), parameter))
    origin = np.r_[vertices.ravel(), np.zeros(9)]
    reduced_initial = np.r_[np.zeros(geometry_count), initial[-9:]]
    lower = np.r_[np.full(geometry_count, -4.), .2, -30., -255., -255., -255., 0., -255., -255., -255.]
    upper = np.r_[np.full(geometry_count, 4.), 4., 255., 255., 255., 255., 280., 255., 255., 255.]
    solution = least_squares(
        lambda p: evaluate(origin+transform@p)[0], reduced_initial,
        jac=lambda p: evaluate(origin+transform@p)[1]@transform,
        bounds=(lower, upper), loss='soft_l1', f_scale=3., max_nfev=max_iterations,
        ftol=1e-7, x_scale='jac', tr_options={'maxiter': 150})
    solution.x = origin+transform@solution.x
    residual, jacobian = evaluate(solution.x)
    residual = residual[:len(xy)]
    fitted = solution.x[:2*nv].reshape(-1, 2)
    if not solution.success:
        raise ValueError('Silhouette image fit did not converge')
    if not .205 < solution.x[2*nv] < 3.995:
        raise ValueError('Silhouette image fit reached its blur bound')
    noise = max(.5, float(np.median(np.abs(residual-np.median(residual)))/.67449))
    support = abs(jacobian[:len(xy), :2*nv])
    totals = np.asarray(support.sum(axis=0)).ravel()
    local_error = np.sqrt(np.asarray(support.T@(residual**2)).ravel()/np.maximum(totals, 1e-12))
    if np.any(totals.reshape(-1, 2).sum(axis=1) < 1.) or local_error.max() > max(8*noise, .12*(bright-dark)):
        raise ValueError(f'Silhouette boundary has missing image support or incompatible blur (local_rms={local_error.max():.2f}, noise={noise:.2f}, support={totals.reshape(-1,2).sum(axis=1).min():.2f})')
    # Validate continuous rings, not just the raster topology.
    # OpenCV intersection checks are insufficient for subpixel self-crossings.
    from shapely.geometry import Polygon
    polygons = [Polygon(fitted[section]) for section in sections]
    seed_polygons = [Polygon(c[:, 0]) for c in contours]
    for i, polygon in enumerate(polygons):
        if not polygon.is_valid or polygon.area < min_area:
            raise ValueError(f'Ring {i} fitted geometry is degenerate or self-intersecting')
        for j in range(i):
            if polygon.boundary.intersects(polygons[j].boundary):
                raise ValueError('Fitted silhouette rings cross or touch')
            if (polygon.contains(polygons[j]) != seed_polygons[i].contains(seed_polygons[j]) or
                    polygons[j].contains(polygon) != seed_polygons[j].contains(seed_polygons[i])):
                raise ValueError('Image fitting changed silhouette nesting')
    rings = tuple(fitted[section][::-1].copy() for section in sections)
    diagnostics = {
        'method': 'gaussian_pixel_integrated_silhouette_fit',
        'threshold': level if polarity == 'dark-on-light' else 255-level,
        'blur_sigma_px': float(solution.x[2*nv]),
        'residual_rms_dn': float(np.sqrt(np.mean(residual**2))),
        'residual_p95_dn': float(np.quantile(np.abs(residual), .95)),
        'maximum_local_residual_rms_dn': float(local_error.max()),
        'noise_sigma_dn': noise,
        'iterations': int(solution.nfev),
        'fit_pixel_count': len(xy),
        'model_knot_count': nv,
        'coordinate_source': 'image_fitted_polygon',
    }
    return SilhouetteFit(rings, depths, diagnostics)


def extract_for_benchmark(image, *, recover_corners=False, polarity='dark-on-light'):
    """Offline comparison only; fitted vertices do not preserve PAE identities."""
    from inspection_measurement import Contours
    fitted = fit_silhouette(image, polarity=polarity)
    rings = list(fitted.rings)
    corrections = 0
    if recover_corners:
        from experimental_corner_fit import repair_corners
        # Use dense support for the local straightness tests. This samples the
        # model's continuous segments; it does not add image observations.
        from evaluate_extraction_degradation import densify
        rings = [densify(q, .75) for q in rings]
        canonical = image if polarity == 'dark-on-light' else 255-image
        for i, ring in enumerate(rings):
            rings[i], n = repair_corners(ring, canonical,
                fitted.diagnostics['blur_sigma_px'], fitted.diagnostics['noise_sigma_dn'])
            corrections += n
    contours = Contours(np.concatenate(rings), np.r_[0, np.cumsum([len(q) for q in rings])])
    diagnostics = dict(fitted.diagnostics, experimental=True,
        corner_corrections=corrections,
        rings=[dict(path=i, depth=d, is_hole=bool(d % 2), point_count=len(q))
               for i, (q, d) in enumerate(zip(rings, fitted.depths))])
    # Topology must survive the additional corner model, including other rings.
    from shapely.geometry import Polygon
    before, after = [Polygon(q) for q in fitted.rings], [Polygon(q) for q in rings]
    for i in range(len(rings)):
        for j in range(i):
            if after[i].boundary.intersects(after[j].boundary):
                raise ValueError('Experimental corner fit made rings cross')
            if (after[i].contains(after[j]) != before[i].contains(before[j]) or
                    after[j].contains(after[i]) != before[j].contains(before[i])):
                raise ValueError('Experimental corner fit changed nesting')
    return contours, diagnostics
