from base64 import b64decode
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy import io

from pae import SilhouetteWorkspace, detect, detect_silhouette


FIELDS = ("position", "x", "y", "nx", "ny", "curv", "i0", "i1")


def rectangle(dtype=np.uint8):
    image = np.full((96, 128), 220, dtype=dtype)
    image[20:75, 25:105] = 20
    return image


def assert_identical(actual, expected):
    for field in FIELDS:
        assert np.array_equal(getattr(actual, field), getattr(expected, field)), field


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize("order", [1, 2])
def test_seeded_rectangle_is_bit_exact_with_full_detector(dtype, order):
    image = rectangle(dtype)
    expected = detect(image, 25, order=order)
    actual = detect_silhouette(
        image,
        25,
        silhouette_threshold=120,
        polarity="dark-on-light",
        order=order,
    )
    assert_identical(actual, expected)


@pytest.mark.parametrize("sigma", [0.5, 1.5, 3.0, 4.0])
def test_seeded_blurred_silhouette_is_bit_exact_with_full_detector(sigma):
    image = cv2.GaussianBlur(rectangle(), (0, 0), sigmaX=sigma, sigmaY=sigma)
    expected = detect(image, 10)
    actual = detect_silhouette(
        image,
        10,
        silhouette_threshold=120,
        search_radius=3,
    )
    assert_identical(actual, expected)


def test_min_area_filters_small_foreground_contours():
    clean = rectangle()
    with_dust = clean.copy()
    with_dust[10:14, 10:14] = 20
    expected = detect(clean, 25)
    actual = detect_silhouette(
        with_dust,
        25,
        silhouette_threshold=120,
        min_area=100,
    )
    assert_identical(actual, expected)


def test_exterior_and_hole_are_both_retained():
    image = rectangle()
    image[35:60, 50:80] = 220
    expected = detect(image, 25)
    actual = detect_silhouette(image, 25, silhouette_threshold=120)
    assert_identical(actual, expected)


def test_light_on_dark_polarity():
    image = 240 - rectangle()
    expected = detect(image, 25)
    actual = detect_silhouette(
        image,
        25,
        silhouette_threshold=120,
        polarity="light-on-dark",
    )
    assert_identical(actual, expected)


def test_reusable_workspace_preserves_results():
    image = rectangle()
    workspace = SilhouetteWorkspace(image.shape)
    first = detect_silhouette(
        image, 25, silhouette_threshold=120, workspace=workspace
    )
    second = detect_silhouette(
        image, 25, silhouette_threshold=120, workspace=workspace
    )
    assert_identical(first, second)


def test_empty_threshold_mask_returns_typed_empty_edges():
    image = np.full((32, 32), 220, dtype=np.uint8)
    edges = detect_silhouette(image, 25, silhouette_threshold=120)
    assert len(edges) == 0
    assert edges.position.dtype == np.int64
    for field in FIELDS[1:]:
        assert getattr(edges, field).dtype == np.float64


def test_matlab_ring_candidates_and_features_are_identical_to_full_path():
    data = Path(__file__).with_name("data") / "ring.mat.b64"
    image = io.loadmat(BytesIO(b64decode(data.read_bytes())))["image"]
    expected = detect(image, 25)
    actual = detect_silhouette(
        image,
        25,
        silhouette_threshold=150,
        search_radius=1,
    )
    assert_identical(actual, expected)


def test_workspace_shape_must_match():
    with pytest.raises(ValueError, match="does not match"):
        detect_silhouette(
            rectangle(),
            25,
            silhouette_threshold=120,
            workspace=SilhouetteWorkspace((32, 32)),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"polarity": "unknown"},
        {"search_radius": -1},
        {"search_radius": 33},
        {"min_area": -1},
        {"min_area": 10, "max_area": 5},
    ],
)
def test_invalid_silhouette_configuration_is_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        detect_silhouette(
            rectangle(), 25, silhouette_threshold=120, **kwargs
        )


@pytest.mark.parametrize('smoothing', [1, 2])
def test_seeded_noise_modes_use_corrected_reference_equations(smoothing):
    image = rectangle()
    image = cv2.GaussianBlur(image, (0, 0), .8)
    # Small crop keeps iterative restoration inexpensive.
    image = cv2.resize(image, (64, 48), interpolation=cv2.INTER_AREA)
    expected = detect(image, 25, smoothing=smoothing)
    actual = detect_silhouette(image, 25, silhouette_threshold=120, smoothing=smoothing)
    assert_identical(actual, expected)


def test_seeded_coordinates_are_validated():
    from pae import detect_seeded
    for seeds in [[[-1, 10]], [[128, 20]], [[10, 96]]]:
        with pytest.raises(ValueError, match='inside'):
            detect_seeded(rectangle(), 25, seeds=np.asarray(seeds))
