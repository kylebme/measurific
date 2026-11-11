"""Independent optical integral checks for the offline experiment, not PAE modes."""
import numpy as np
import pytest
from scipy.special import ndtr

from experimental_optical_fit import _polygon_model
from experimental_corner_fit import fit_corner_pixels


def primitive(t, sigma):
    return t*ndtr(t/sigma)+sigma*np.exp(-.5*(t/sigma)**2)/np.sqrt(2*np.pi)


def rectangle_coverage(xy, rectangle, sigma):
    def integral(p, a, b):
        def f(t):
            return primitive(t+.5, sigma)-primitive(t-.5, sigma)
        return f(b-p)-f(a-p)
    x0, y0, x1, y1 = rectangle
    return integral(xy[:, 0], x0, x1)*integral(xy[:, 1], y0, y1)


@pytest.mark.parametrize('sigma', [.35, 1.8])
def test_polygon_integral_matches_independent_separable_rectangle(sigma):
    bounds = (22.23, 19.71, 92.64, 76.18)
    x0, y0, x1, y1 = bounds
    vertices = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    yy, xx = np.mgrid[17:23, 19:26]
    xy = np.c_[xx.ravel(), yy.ravel()].astype(float)
    nodes, weights = np.polynomial.legendre.leggauss(6)
    nodes, weights = (nodes+1)/2, weights/2
    predicted = _polygon_model(xy, vertices, np.array([1, 2, 3, 0]), sigma, nodes, weights)[0]
    np.testing.assert_allclose(predicted, rectangle_coverage(xy, bounds, sigma), atol=1e-8)


@pytest.mark.parametrize('sigma', [.35, 1.8])
def test_polygon_shape_and_sigma_jacobian(sigma):
    vertices = np.array([[4.2, 3.7], [14.6, 4.1], [13.8, 12.8], [3.6, 12.1]])
    xy = np.array([[4., 4.], [8., 4.], [14., 9.], [6., 10.]])
    ends = np.array([1, 2, 3, 0])
    nodes, weights = np.polynomial.legendre.leggauss(6)
    nodes, weights = (nodes+1)/2, weights/2
    _, js, rows, cols, data = _polygon_model(xy, vertices, ends, sigma, nodes, weights)
    jac = np.zeros((len(xy), 8))
    np.add.at(jac, (rows, cols), data)
    step = 1e-5
    for k in range(9):
        plus, minus = vertices.copy(), vertices.copy()
        if k < 8:
            plus.ravel()[k] += step
            minus.ravel()[k] -= step
        sp, sm = (sigma+step, sigma-step) if k == 8 else (sigma, sigma)
        numeric = (_polygon_model(xy, plus, ends, sp, nodes, weights)[0]-
                   _polygon_model(xy, minus, ends, sm, nodes, weights)[0])/(2*step)
        np.testing.assert_allclose(numeric, js if k == 8 else jac[:, k], atol=2e-5)


@pytest.mark.parametrize('sigma', [.35, 1.8])
@pytest.mark.parametrize('noise', [0., 2.])
def test_corner_location_against_independent_optical_image(sigma, noise):
    bounds = (22.23, 19.71, 92.64, 76.18)
    yy, xx = np.mgrid[:100, :120]
    xy = np.c_[xx.ravel(), yy.ravel()]
    image = (232-216*rectangle_coverage(xy, bounds, sigma)).reshape(100, 120)
    image += np.random.default_rng(144).normal(0, noise, image.shape)
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    truth = np.array(bounds[:2])
    initial = truth+[.4, -.3]
    result = fit_corner_pixels(image, initial, initial+[0, 10], initial+[10, 0], sigma, max(.5, noise))
    assert result is not None
    assert np.linalg.norm(result[0]-truth) < .1


@pytest.mark.parametrize('sigma', [.35, .8, 1.8])
@pytest.mark.parametrize('noise', [0., 2.])
def test_corner_estimates_unknown_blur_from_pixels(sigma, noise):
    bounds = (22.23, 19.71, 92.64, 76.18)
    yy, xx = np.mgrid[:100, :120]
    xy = np.c_[xx.ravel(), yy.ravel()]
    image = (232-216*rectangle_coverage(xy, bounds, sigma)).reshape(100, 120)
    image += np.random.default_rng(193).normal(0, noise, image.shape)
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    truth = np.array(bounds[:2])
    initial = truth+[.4, -.3]
    result = fit_corner_pixels(image, initial, initial+[0, 10], initial+[10, 0], None, max(.5, noise))
    assert result is not None
    assert np.linalg.norm(result[0]-truth) < .1
    assert abs(result[1]['blur_sigma_px']-sigma) < .08


def test_recovery_retains_raw_pae_and_rejects_smooth_circle_corners():
    from scipy.ndimage import gaussian_filter
    from inspection_measurement import contours_from_image
    from experimental_corner_fit import recover_pae_corners
    yy, xx = np.mgrid[:140, :160]
    image = np.rint(232-216*gaussian_filter(((xx-80)**2+(yy-70)**2 < 35**2).astype(float), 1.)).astype(np.uint8)
    contours, info = contours_from_image(image)
    raw = contours.xy.copy()
    fitted, diagnostics = recover_pae_corners(image, contours, info)
    np.testing.assert_array_equal(contours.xy, raw)
    assert diagnostics['corner_recovery']['accepted'] == []
    np.testing.assert_array_equal(fitted.xy, raw)


def test_chamfer_challenge_rejects_false_sharp_corner():
    # Independent source-area quadrature for a bevel cut from a rectangle.
    # Do not generate this negative control using the fitted polygon model.
    sigma, noise = 1.6, 2.
    bounds = (22.23, 19.71, 92.64, 76.18)
    yy, xx = np.mgrid[:100, :120]
    xy = np.c_[xx.ravel(), yy.ravel()]
    coverage = rectangle_coverage(xy, bounds, sigma)
    nodes, weights = np.polynomial.legendre.leggauss(96)
    width = 2.5
    for u, weight in zip((nodes+1)*width/2, weights*width/2):
        x = bounds[0]+u
        x_weight = ndtr((xy[:, 0]+.5-x)/sigma)-ndtr((xy[:, 0]-.5-x)/sigma)
        lo, hi = bounds[1], bounds[1]+width-u
        def f(t):
            return primitive(t+.5, sigma)-primitive(t-.5, sigma)
        y_weight = f(hi-xy[:, 1])-f(lo-xy[:, 1])
        coverage -= weight*x_weight*y_weight
    image = (232-216*coverage).reshape(100, 120)
    image += np.random.default_rng(193).normal(0, noise, image.shape)
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    initial = np.array(bounds[:2])+[.3, .3]
    audit = {}
    result = fit_corner_pixels(image, initial, initial+[0, 10], initial+[10, 0],
                               None, noise, audit=audit, check_chamfer=True)
    assert result is None, audit


def test_identifiable_sharp_corner_survives_chamfer_challenge():
    bounds = (22.23, 19.71, 92.64, 76.18)
    yy, xx = np.mgrid[:100, :120]
    image = np.rint(232-216*rectangle_coverage(np.c_[xx.ravel(), yy.ravel()], bounds, .4)).reshape(100, 120).astype(np.uint8)
    truth = np.array(bounds[:2])
    result = fit_corner_pixels(image, truth+[.2, -.2], truth+[0, 10], truth+[10, 0],
                               None, .5, check_chamfer=True)
    assert result is not None
    assert np.linalg.norm(result[0]-truth) < .05


def test_non_gaussian_psf_is_not_accepted_as_precise_gaussian_corner():
    bounds = (22.23, 19.71, 92.64, 76.18)
    yy, xx = np.mgrid[:100, :120]
    xy = np.c_[xx.ravel(), yy.ravel()]
    coverage = .65*rectangle_coverage(xy, bounds, .4)+.35*rectangle_coverage(xy, bounds, 2.)
    image = np.rint(232-216*coverage).reshape(100, 120).astype(np.uint8)
    truth = np.array(bounds[:2])
    audit = {}
    fit = fit_corner_pixels(image, truth+[.2, -.2], truth+[0, 10], truth+[10, 0],
                            None, .5, check_chamfer=True, audit=audit)
    assert fit is None, audit


def test_local_recovery_preserves_input_and_handles_inverted_polarity():
    from inspection_measurement import contours_from_image
    from experimental_corner_fit import recover_pae_corners
    bounds = (22.23, 19.71, 92.64, 76.18)
    yy, xx = np.mgrid[:100, :120]
    image = np.rint(232-216*rectangle_coverage(np.c_[xx.ravel(), yy.ravel()], bounds, .4)).reshape(100, 120).astype(np.uint8)
    outputs = []
    for pixels, polarity in [(image, 'dark-on-light'), (255-image, 'light-on-dark')]:
        raw, info = contours_from_image(pixels, polarity=polarity)
        before, metadata = raw.xy.copy(), repr(info)
        recovered, diagnostics = recover_pae_corners(pixels, raw, info, polarity=polarity)
        np.testing.assert_array_equal(raw.xy, before)
        assert repr(info) == metadata
        assert len(diagnostics['corner_recovery']['accepted']) == 4
        outputs.append(recovered.xy)
    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-8)
