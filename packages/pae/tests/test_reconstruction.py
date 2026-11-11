import math

import numpy as np
import pytest
from scipy.integrate import quad

from pae import detect
from pae.circle import circle_pixel_area, circle_horizontal_window


@pytest.mark.parametrize('radius', [4.5, 20., 10000.])
def test_circle_pixel_area_against_independent_quadrature(radius):
    rng = np.random.default_rng(98)
    for theta in np.r_[np.linspace(0, math.pi/2, 12), rng.uniform(0, 2*math.pi, 12)]:
        x, y = (radius+rng.uniform(-.3, .3))*np.array([np.cos(theta), np.sin(theta)])
        x0, x1, y0, y1 = x-.5, x+.5, y-.5, y+.5
        def height(u):
            h = math.sqrt(max(0., radius*radius-u*u))
            return max(0., min(y1, h)-max(y0, -h)) if abs(u) < radius else 0.
        cuts = [z for z in [-radius, radius] if x0 < z < x1]
        for v in [y0, y1]:
            if abs(v) < radius:
                c = math.sqrt((radius-v)*(radius+v))
                cuts.extend(z for z in [-c, c] if x0 < z < x1)
        expected = quad(height, x0, x1, points=cuts, epsabs=1e-10)[0]
        assert abs(circle_pixel_area(x0, y0, x1, y1, radius)-expected) < 2e-8


def test_tangent_crescent_is_not_missed():
    # All four corners are outside; the disk still intersects the pixel.
    assert circle_pixel_area(9.49, -.5, 10.49, .5, 9.5) > 0


def test_reconstruction_does_not_depend_on_sampling_grid_resolution():
    args = (0., 0., 7.22, .3, 7.4, 12., 230.)
    np.testing.assert_array_equal(circle_horizontal_window(*args, 10),
                                  circle_horizontal_window(*args, 100))


def test_iterating_a_perfect_fractional_step_does_not_introduce_grid_bias():
    image = np.full((24, 32), 16.)
    image[:, 17:] = 232
    image[:, 16] = 16+216*.37
    result = detect(image, 10, smoothing=10)
    keep = (result.y > 8) & (result.y < 16)
    assert np.max(abs(result.x[keep]-(16+.5-.37))) < .001
