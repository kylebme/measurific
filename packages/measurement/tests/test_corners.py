import copy
from pathlib import Path

import numpy as np
import pytest

from inspection_measurement import Contours, evaluate, prepare_interactive_report
from inspection_measurement.corners import corner_exclusions


def rectangle(step=0.5):
    corners = np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 40.0], [0.0, 40.0]])
    return np.concatenate([
        a + np.arange(0, np.linalg.norm(b - a), step)[:, None] * (b - a) / np.linalg.norm(b - a)
        for a, b in zip(corners, np.roll(corners, -1, axis=0))
    ])


def test_default_mask_keeps_raw_contours_and_uses_one_pixel_radius():
    q = rectangle()
    contours = Contours(q, [0, len(q)])
    arrays, digest = contours.model_arrays(), contours.coordinate_hash
    request = {'features': [{'id': 'profile', 'type': 'contour', 'selection': {'path': 0}}]}
    original_request = copy.deepcopy(request)

    report = evaluate(contours, request)
    record = report['features'][0]

    assert report['corner_exclusion']['radius_px'] == 1
    assert len(report['corner_exclusion']['corners']) == 4
    assert len(record['support']) == 4 and record['closed'] == [False] * 4
    assert report['coordinate_hash'] == digest == contours.coordinate_hash
    np.testing.assert_array_equal(contours.model_arrays()['actual_xy_mm'], arrays['actual_xy_mm'])
    assert request == original_request
    excluded = set(report['corner_exclusion']['excluded_vertex_indices'])
    for edge in record['edge_indices']:
        assert edge not in excluded and (edge + 1) % len(q) not in excluded


@pytest.mark.parametrize('radius', [1.0, 3.0, 5.0])
def test_smooth_circles_are_not_corners(radius):
    t = np.linspace(0, 2 * np.pi, 512, endpoint=False)
    q = radius * np.c_[np.cos(t), np.sin(t)]
    keep, info = corner_exclusions(Contours(q, [0, len(q)]), radius)
    assert keep.all() and info['corners'] == []


def test_zero_radius_disables_corner_exclusion():
    q = rectangle()
    report = evaluate(Contours(q, [0, len(q)]), {
        'corner_exclusion_px': 0,
        'features': [{'id': 'polygon', 'type': 'polygon', 'selection': {'path': 0}}],
    })
    assert report['features'][0]['status'] == 'fitted'
    assert report['corner_exclusion']['excluded_vertex_count'] == 0
    assert report['measurements'][0]['value'] == 2000


def test_interactive_report_retains_selected_radius():
    q = rectangle()
    report, session = prepare_interactive_report(Contours(q, [0, len(q)]), {
        'corner_exclusion_px': 4,
        'features': [{'id': 'top', 'type': 'line', 'selection': {'spans': [
            {'ring_index': 0, 'start_edge': 0, 'edge_count': 100},
        ]}}],
    })
    assert session['corner_exclusion_px'] == 4
    assert report['corner_exclusion']['radius_px'] == 4


@pytest.mark.parametrize('value', [-1, True, '3', None, float('nan'), float('inf')])
def test_invalid_radius_is_rejected(value):
    q = rectangle()
    with pytest.raises(ValueError, match='corner_exclusion_px'):
        evaluate(Contours(q, [0, len(q)]), {'corner_exclusion_px': value})


@pytest.mark.parametrize('path', [3, 4, 5])
def test_extracted_circles_survive_false_corner_exclusions(path):
    # Unmodified PAE contours from app/demo-data/extras/000000006.image.png.
    with np.load(Path(__file__).parent / 'fixtures/corner_exclusion_circles.npz') as data:
        q = data[f'path_{path}']
    contours = Contours(q, [0, len(q)])
    request = {'features': [{'id': 'hole', 'type': 'circle', 'selection': {'path': 0}}]}
    baseline = evaluate(contours, {**request, 'corner_exclusion_px': 0})['features'][0]
    report = evaluate(contours, request)
    record = report['features'][0]
    assert record['status'] == 'fitted'
    assert len(record['excluded_edge_indices']) in (2, 3)
    assert record['closed'] == [False]
    assert record['geometry']['retained_angular_support'] > 340
    assert record['geometry']['radius'] == pytest.approx(baseline['geometry']['radius'], abs=.02)
    assert all(m['status'] == 'measured' for m in report['measurements'])
    # Excluded observations stay excluded; no synthetic closing edge is added.
    excluded = set(report['corner_exclusion']['excluded_vertex_indices'])
    for edge in record['edge_indices']:
        assert edge not in excluded and (edge + 1) % len(q) not in excluded


@pytest.mark.parametrize('excluded_ranges, succeeds', [
    ([(0, 8), (64, 72)], True),  # Multiple disconnected retained arcs.
    ([(0, 100)], False),  # Less than half the circle survives.
])
def test_circle_exclusion_coverage(monkeypatch, excluded_ranges, succeeds):
    import inspection_measurement.engine as engine

    t = np.linspace(0, 2*np.pi, 128, endpoint=False)
    q = 20*np.c_[np.cos(t), np.sin(t)]
    eligible = np.ones(len(q), dtype=bool)
    for lo, hi in excluded_ranges:
        eligible[lo:hi] = False
    monkeypatch.setattr(engine, 'corner_exclusions', lambda *_: (eligible, {}))
    report = evaluate(Contours(q, [0, len(q)]), {
        'features': [{'id': 'hole', 'type': 'circle', 'selection': {'path': 0}}],
    })
    record = report['features'][0]
    if succeeds:
        assert record['status'] == 'fitted'
        assert record['closed'] == [False, False]
        assert record['geometry']['radius'] == pytest.approx(20)
        np.testing.assert_allclose(record['geometry']['center'], [0, 0], atol=1e-10)
    else:
        assert record['status'] == 'fit_failed'
        assert 'insufficient circle angular support' in record['reason']


@pytest.mark.parametrize('radius', [0, 1])
def test_open_circle_selection_still_rejected(radius):
    t = np.linspace(0, 2*np.pi, 128, endpoint=False)
    q = 20*np.c_[np.cos(t), np.sin(t)]
    report = evaluate(Contours(q, [0, len(q)], closed=[False]), {
        'corner_exclusion_px': radius,
        'features': [{'id': 'hole', 'type': 'circle', 'selection': {'path': 0}}],
    })
    assert report['features'][0]['status'] == 'fit_failed'
    assert report['features'][0]['reason'] == 'Operator requires one closed contour'
