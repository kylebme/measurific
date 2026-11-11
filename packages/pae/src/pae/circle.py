# SPDX-License-Identifier: MIT
# Derived API: Copyright (c) 2020 Giuseppe Vitucci and Nicola Vitucci.
"""Area-exact circle/pixel integration for PAE iterative reconstruction.

The inherited sampling grid used arange(-.5, .5, 1/(N-1)), excluding its positive
endpoint and shifting the area samples toward the upper left. Analytic clipping
removes that bias, sampling noise, and missed tangent crescents.
"""
import math
import numpy as np
from numba import njit


@njit(cache=True)
def _circle_integral(a, b, radius):
    # Stable definite integral of sqrt(R^2-x^2). Subtracting two O(R^2)
    # primitives loses area precision for the near-straight R=10000 case.
    ta = math.asin(min(1., max(-1., a/radius)))
    tb = math.asin(min(1., max(-1., b/radius)))
    delta = tb-ta
    if abs(delta) < .01:
        d2 = delta*delta
        difference = delta*d2*(1/6-d2/120+d2*d2/5040)
    else:
        difference = delta-math.sin(delta)
    return .5*radius*radius*(difference+2*math.cos((ta+tb)/2)**2*math.sin(delta))


@njit(cache=True)
def circle_pixel_area(x0, y0, x1, y1, radius):
    """Area of the intersection of a disk at the origin and a rectangle."""
    left, right = max(x0, -radius), min(x1, radius)
    if left >= right or y0 >= radius or y1 <= -radius:
        return 0.
    cuts = np.empty(6)
    cuts[0], cuts[1] = left, right
    count = 2
    for y in (y0, y1):
        if abs(y) < radius:
            crossing = math.sqrt(max(0., (radius-y)*(radius+y)))
            for x in (-crossing, crossing):
                if left < x < right:
                    cuts[count] = x
                    count += 1
    cuts = np.sort(cuts[:count])
    area = 0.
    for k in range(count-1):
        a, b = cuts[k], cuts[k+1]
        mid = (a+b)/2
        height = math.sqrt(max(0., (radius-mid)*(radius+mid)))
        if min(y1, height) <= max(y0, -height):
            continue
        curved = _circle_integral(a, b, radius)
        top = y1*(b-a) if y1 < height else curved
        bottom = y0*(b-a) if y0 > -height else -curved
        area += top-bottom
    return min((x1-x0)*(y1-y0), max(0., area))


@njit(cache=True)
def _circle_window(x_window_center, y_window_center, x_center, y_center,
                   radius, inner_intensity, outer_intensity, half_x, half_y):
    result = np.empty((2*half_y+1, 2*half_x+1))
    for row in range(result.shape[0]):
        y = y_window_center+row-half_y-y_center
        for column in range(result.shape[1]):
            x = x_window_center+column-half_x-x_center
            area = circle_pixel_area(x-.5, y-.5, x+.5, y+.5, radius)
            result[row, column] = outer_intensity+(inner_intensity-outer_intensity)*area
    return result


@njit(cache=True)
def circle_horizontal_window(x_window_center, y_window_center, x_center, y_center,
                             radius, inner_intensity, outer_intensity, grid_resolution):
    # grid_resolution is retained for compatibility; no numerical grid is used.
    return _circle_window(x_window_center, y_window_center, x_center, y_center,
                          radius, inner_intensity, outer_intensity, 4, 1)


@njit(cache=True)
def circle_vertical_window(x_window_center, y_window_center, x_center, y_center,
                           radius, inner_intensity, outer_intensity, grid_resolution):
    return _circle_window(x_window_center, y_window_center, x_center, y_center,
                          radius, inner_intensity, outer_intensity, 1, 4)
