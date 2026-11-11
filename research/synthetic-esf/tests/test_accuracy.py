import numpy as np
from telecentric import Camera, Primitive, Scene
from telecentric.accuracy import integrated_gaussian, crossing_1d, fit_native_gaussian_edge, shape_dimensions, stable_mask, analytic_controls


def test_linear_threshold_phase_bias_exists_without_distortion():
    x = np.arange(-8, 10)+0.5
    errors = [crossing_1d(x, integrated_gaussian(x, phase, 0.6))-phase for phase in np.arange(101)/101]
    assert 0.03 < max(np.abs(errors)) < 0.04
    assert abs(np.mean(errors)) < 1e-10


def test_native_pixel_fit_recovers_independent_exact_gaussian():
    x = np.arange(32)+0.5
    image = np.tile(integrated_gaussian(x, 16.37, 0.6), (32, 1))
    fit = fit_native_gaussian_edge(image, {'roi_xywh': [0, 0, 32, 32]}, np.array([16., 16.]), np.array([1., 0.]), 0.4)
    assert fit is not None
    assert abs(fit['shift_px']-0.37) < 1e-8
    assert abs(fit['sigma_px']-0.6) < 1e-8


def test_common_edge_translation_cancels_from_dimensions():
    camera = Camera(width=40, height=40, pixel_pitch_um=10, magnification=1)
    for shape in [Primitive('box', size_mm=(0.16, 0.18, 0.01), angle_deg=17), Primitive('sphere', radius_mm=0.08)]:
        label = Scene([shape]).labels(camera, count=128)[0]
        xy = np.asarray(label['ideal_xy_px'])
        dimensions = shape_dimensions(label, xy+[0.2, -0.1], np.ones(len(xy), bool), camera)
        assert dimensions
        for value, truth in dimensions.values():
            assert abs(value-truth) < 1e-9


def test_radial_edge_bias_doubles_in_diameter():
    camera = Camera(width=40, height=40, pixel_pitch_um=10, magnification=1)
    label = Scene([Primitive('sphere', radius_mm=0.08)]).labels(camera, count=128)[0]
    points = np.asarray(label['ideal_xy_px'])+0.02*np.asarray(label['normal_world_xy'])
    value, truth = shape_dimensions(label, points, np.ones(len(points), bool), camera)['diameter']
    assert abs((value-truth)-0.04) < 1e-9


def test_corner_exclusion_is_geometry_based():
    camera = Camera(width=40, height=40, pixel_pitch_um=10, magnification=1)
    label = Scene([Primitive('box', size_mm=(0.16, 0.16, 0.01))]).labels(camera, count=128)[0]
    keep = stable_mask(label, camera, margin_px=3)
    assert keep.sum() == 80
