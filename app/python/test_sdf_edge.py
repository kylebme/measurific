"""Verify SDF coordinates/topology and both shipped inference paths."""
from pathlib import Path

import numpy as np
import pytest
from worker import InspectionWorker, configure_imports

ROOT = Path(__file__).resolve().parents[2]
configure_imports(ROOT)


def test_bilinear_sdf_rings_preserve_coordinates_and_holes():
    from inspection_measurement.sdf_edge import field_rings
    from edgesr.runtime import nest_rings
    yy, xx = np.mgrid[:65, :75].astype(float) + .5
    # Rectangular part with a rectangular hole: bilinear interpolation has exact
    # straight edges, so this detects a half-pixel or subpixel origin error.
    outer = np.minimum.reduce([xx - 10.25, 60.25 - xx, yy - 8.25, 55.25 - yy])
    hole = np.minimum.reduce([xx - 25.25, 40.25 - xx, yy - 20.25, 35.25 - yy])
    sdf = np.minimum(outer, -hole)
    traced = field_rings(sdf)
    nested = nest_rings(traced['rings'], traced['is_hole'], (0, 0))
    assert nested['is_hole'] == [False, True]
    assert nested['parent'] == [-1, 0]
    for q, bounds in zip(nested['rings'], [(10.25, 8.25, 60.25, 55.25), (25.25, 20.25, 40.25, 35.25)]):
        np.testing.assert_allclose(np.r_[q.min(0), q.max(0)], bounds, atol=1e-6)
    lattice = (np.concatenate(nested['rings']) - .5) * 16
    assert (np.min(np.abs(lattice - np.round(lattice)), axis=1) < 1e-5).all()


def test_sdf_worker_closes_circular_camera_roi():
    from inspection_measurement import contours_from_image
    worker = InspectionWorker(ROOT, backend='geometry', edge_method='deep-learning')
    yy, xx = np.ogrid[:160, :180]
    image = np.full((160, 180), 235, np.uint8)
    image[60:100, :] = 12
    valid = (xx + .5 - 90) ** 2 + (yy + .5 - 80) ** 2 <= 60 ** 2
    contours, info = contours_from_image(image, valid_mask=valid, model=worker.edge_model)
    assert contours.closed.tolist() == [True]
    assert info['model']['format'] == 'edgesr-sdf-onnx-v1'
    assert info['boundary_closure_padding_px'] == 16
    q = contours.path(0)
    assert q[:, 0].min() > 25 and q[:, 0].max() < 155


def test_sdf_torch_checkpoint_matches_onnx_contours():
    pytest.importorskip('torch')  # Packaged runtime intentionally ships only ONNX.
    from inspection_measurement.adapter import load_edge_model
    from inspection_measurement import contours_from_image
    from scipy.spatial import cKDTree
    image = np.full((96, 112), 235, np.uint8)
    image[25:75, 30:85] = 12
    model = load_edge_model(ROOT / 'packages/edge-model/models/sdf_v5.pt')
    worker = InspectionWorker(ROOT, backend='geometry', edge_method='deep-learning')
    torch_contours, _ = contours_from_image(image, model=model)
    onnx_contours, _ = contours_from_image(image, model=worker.edge_model)
    assert model.describe()['format'] == 'edgesr-sdf-torch-v1'
    assert torch_contours.closed.tolist() == onnx_contours.closed.tolist() == [True]
    assert cKDTree(torch_contours.xy).query(onnx_contours.xy)[0].max() < .01
