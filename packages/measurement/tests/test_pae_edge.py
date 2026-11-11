"""PAE edges seeded by the coarse edge model (the GUI's default edge method)."""
from pathlib import Path

import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon

from inspection_measurement import contours_from_image, inspect_image
from inspection_measurement.adapter import ContourExtractionError, load_pae_edge_model
from inspection_measurement.pae_edge import _resolve_local_crossings
from inspection_shapes.geometry import generate
from inspection_shapes.render import Camera, render

ROOT = Path(__file__).resolve().parents[2]
SDF_MODEL = ROOT / 'edge-model' / 'models' / 'sdf_v5-onnx'


@pytest.fixture(scope='module')
def model():
    return load_pae_edge_model(SDF_MODEL)


@pytest.fixture(scope='module')
def washer():
    part = generate('washer', 561, severity=0)
    ppm = 150/np.ptp(part.rings[0].actual, axis=0).max()
    camera = Camera(width=240, height=220, supersample=4, pixels_per_mm=ppm,
                    translation_x=120, translation_y=110, sigma_px=.8,
                    read_noise_dn=0, prnu=0, electrons_per_dn=100000)
    return render(part, camera, seed=14)[0]['image']


def test_shares_the_coarse_model_of_the_edge_model_export(model):
    assert model.describe()['format'] == 'edgesr-coarse-pae-v1'
    assert Path(model.describe()['directory']) == (ROOT / 'edge-model' / 'models' / 'coarse_v2-onnx').resolve()
    assert load_pae_edge_model(ROOT / 'edge-model' / 'models' / 'coarse_v2-onnx') is model


def test_washer_rings_and_diameter(model, washer):
    contours, info = contours_from_image(washer, model=model)
    assert [r['is_hole'] for r in info['rings']] == [False, True]
    assert info['method'] == 'coarse_model_pae_rings'
    assert info['interpolated_point_count'] == 0
    assert info['pae']['used_observations'] > 0.9 * info['pae']['observations']
    outer = next(r['path'] for r in info['rings'] if not r['is_hole'])
    report, _, _ = inspect_image(washer, {'features': [{'id': 'outer', 'type': 'circle', 'selection': {'path': outer}}]},
                                 contours=contours)
    diameter = next(m['value'] for m in report['measurements'] if m['operator'] == 'least_squares_diameter')
    assert abs(diameter-150) < .15
    # Same input, same observations: JSON edge selections stay stable.
    again, _ = contours_from_image(washer, model=model)
    assert again.coordinate_hash == contours.coordinate_hash


def test_polarity_and_bit_depth_do_not_move_edges(model, washer):
    eight, _ = contours_from_image(washer, model=model)
    inverted, _ = contours_from_image(255-washer, polarity='light-on-dark', model=model)
    sixteen, info = contours_from_image(washer.astype(np.uint16) * 257, model=model)
    assert info['input_bit_depth'] == 16
    np.testing.assert_allclose(inverted.xy, eight.xy, atol=1e-9)
    np.testing.assert_allclose(sixteen.xy, eight.xy, atol=1e-6)


def test_straight_edges_are_subpixel_accurate(model):
    import shapely
    # Area-sampled tilted quadrilateral with a round hole (pixel centers at integers).
    corners = np.array([[50.3, 42.7], [210.6, 50.1], [205.2, 160.4], [44.9, 153.0]])
    center, radius, scale = np.array([130.4, 100.2]), 25., 16
    y, x = (np.mgrid[:200 * scale, :260 * scale] + .5) / scale - .5
    inside = np.ones(x.shape, bool)
    for a, b in zip(corners, np.roll(corners, -1, axis=0)):
        inside &= (b[0]-a[0])*(y-a[1]) - (b[1]-a[1])*(x-a[0]) >= 0
    inside &= (x-center[0])**2 + (y-center[1])**2 > radius**2
    coverage = inside.reshape(200, scale, 260, scale).mean(axis=(1, 3))
    image = np.rint(230 - 215 * coverage).astype(np.uint8)
    contours, info = contours_from_image(image, model=model)
    assert sorted(r['is_hole'] for r in info['rings']) == [False, True]
    for ring in info['rings']:
        q = contours.path(ring['path'])
        truth = shapely.Point(center).buffer(radius, 1024).exterior if ring['is_hole'] else Polygon(corners).exterior
        distance = shapely.distance(shapely.points(q), truth)
        # PAE's local edge model does not cover sharp corners.
        away = np.min(np.linalg.norm(q[:, None] - corners[None], axis=2), axis=1) > 4
        assert np.median(distance[away]) < .01 and np.percentile(distance[away], 99) < .02
        # A lone observation where a small circle switches between row and column fits.
        assert distance[away].max() < .2


def test_part_cut_by_the_frame_and_roi_closes(model):
    image = np.full((220, 240), 235, np.uint8)
    cv2.rectangle(image, (0, 45), (239, 175), 12, -1)
    cv2.circle(image, (120, 110), 22, 235, -1)
    contours, info = contours_from_image(image, model=model)
    assert sorted(r['is_hole'] for r in info['rings']) == [False, True]
    outer = contours.path(next(r['path'] for r in info['rings'] if not r['is_hole']))
    assert outer[:, 0].min() <= 0 and outer[:, 0].max() >= image.shape[1] - 1

    image = np.full((240, 240), 235, np.uint8)
    cv2.rectangle(image, (0, 80), (239, 160), 12, -1)
    yy, xx = np.ogrid[:240, :240]
    valid = (xx + .5 - 120) ** 2 + (yy + .5 - 120) ** 2 <= 90 ** 2
    contours, _ = contours_from_image(image, valid_mask=valid, model=model)
    assert contours.closed.tolist() == [True]
    assert np.ptp(contours.path(0), axis=0).min() > 80


def test_several_objects_and_minimum_areas(model):
    image = np.full((200, 200), 230, np.uint8)
    image[60:140, 60:140] = 15
    image[90:110, 90:110] = 230                 # hole
    image[20:24, 20:24] = 15                    # 16 px^2 speck
    image[20:32, 160:172] = 15                  # 144 px^2 part
    contours, info = contours_from_image(image, model=model)
    assert sorted((r['is_hole'], r['depth']) for r in info['rings']) == [(False, 0), (False, 0), (True, 1)]
    contours, _ = contours_from_image(image, min_object_area=200, model=model)
    assert len(contours.closed) == 2


def test_missing_edge_support_is_a_located_failure(model):
    # The right edge fades over ~40 px: the coarse model sees the part, PAE no edge.
    xx = np.arange(240)
    ramp = np.clip((xx - 150) / 40, 0, 1)
    image = np.full((200, 240), 230.)
    image[50:150, 40:] = 15 + 215 * ramp[40:]
    with pytest.raises(ContourExtractionError, match='PAE edge extraction failed.*Deep Learning') as failure:
        contours_from_image(np.rint(image).astype(np.uint8), model=model)
    issue = failure.value.diagnostic['issue']
    assert issue['label'] == 'Missing PAE edge support'
    assert issue['bounds'][0] > 100                 # at the faded edge, not the whole frame
    assert failure.value.diagnostic['edges']        # the coarse ring for the overlay


def test_local_corner_reordering_never_moves_points():
    square = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [3, 1], [3, 2], [3, 3], [2, 3], [1, 3], [0, 3], [0, 2], [0, 1]], float)
    tangled = square.copy()
    tangled[[2, 3, 4]] = tangled[[4, 3, 2]]     # a three-point reversal at a corner
    assert not Polygon(tangled).is_valid
    fixed, adjusted = _resolve_local_crossings(tangled)
    assert adjusted == 1 and Polygon(fixed).is_valid
    np.testing.assert_array_equal(np.sort(fixed, axis=0), np.sort(square, axis=0))
