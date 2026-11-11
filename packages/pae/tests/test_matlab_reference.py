"""Golden tests retained from gravi-toni/subpixel-edges (MIT, commit a4f8f1a)."""

from base64 import b64decode
from io import BytesIO
from pathlib import Path

import numpy as np
from scipy import io

from pae import detect


DATA = Path(__file__).with_name("data")
FIELDS = ("x", "y", "nx", "ny", "curv", "i0", "i1")


def load_mat(name):
    encoded = (DATA / f"{name}.mat.b64").read_bytes()
    return io.loadmat(BytesIO(b64decode(encoded)))


def assert_reference(
    actual,
    reference,
    *,
    excluded=(),
    geometry_atol=1e-12,
    intensity_atol=1e-12,
):
    mask = np.ones(len(actual.position), dtype=bool)
    mask[list(excluded)] = False
    assert np.array_equal(actual.position[mask], reference["position"].ravel("F")[mask] - 1)
    for field in FIELDS:
        offset = 1 if field in ("x", "y") else 0
        expected = reference[field].ravel("F") - offset
        tolerance = intensity_atol if field in ("i0", "i1") else geometry_atol
        assert np.allclose(
            getattr(actual, field)[mask], expected[mask], rtol=0.0, atol=tolerance
        )


def test_smoothing_zero_matches_matlab_golden_data():
    image = load_mat("ring")["image"].astype(np.float64)
    assert_reference(detect(image, 25, smoothing=0, order=2), load_mat("ring_0"))


def test_smoothing_one_matches_reference_geometry():
    image = load_mat("ring")["image"].astype(np.float64)
    # One upstream NMS tie uses a different candidate index (258); the
    # corrected local estimator now agrees on its geometry as well.
    assert_reference(
        detect(image, 25, smoothing=1, order=2),
        load_mat("ring_1"),
        excluded=(258,),
        intensity_atol=2e-6,
    )


def test_area_exact_reconstruction_stays_near_sampled_matlab_fixture():
    image = load_mat("ring")["image"].astype(np.float64)
    assert_reference(
        detect(image, 25, smoothing=2, order=2),
        load_mat("ring_2"),
        # Analytic area integration intentionally removes the reference
        # reconstruction raster bias (up to .0092 px in this fixture).
        # Independent analytic area/edge tests enforce the stronger accuracy.
        geometry_atol=.01,
        intensity_atol=1e-3,
    )


def test_previously_excluded_candidate_has_correct_geometry():
    actual = detect(load_mat("ring")["image"], 25, smoothing=1)
    reference = load_mat("ring_1")
    for name in ("x", "y", "nx", "ny", "curv"):
        expected = reference[name].ravel('F')[258]-(1 if name in ('x', 'y') else 0)
        assert abs(getattr(actual, name)[258]-expected) < 1e-12
