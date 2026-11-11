import numpy as np
import pytest

from pae import detect, subpixel_edges


def partial_area_vertical(
    shape=(48, 64), *, edge_x=28.2, low=10.0, high=210.0
):
    x = np.arange(shape[1], dtype=np.float64)
    bright_fraction = np.clip(x + 0.5 - edge_x, 0.0, 1.0)
    return np.broadcast_to(low + (high - low) * bright_fraction, shape).copy()


def test_vertical_partial_area_step_is_located_analytically():
    edge_x = 28.2
    edges = detect(partial_area_vertical(edge_x=edge_x), 25, order=2)

    assert len(edges) > 20
    assert np.allclose(edges.x, edge_x, atol=1e-12)
    assert np.allclose(edges.nx, 1.0, atol=1e-12)
    assert np.allclose(edges.ny, 0.0, atol=1e-12)
    assert np.allclose(edges.curv, 0.0, atol=1e-12)
    assert np.allclose(edges.i0, 10.0, atol=1e-12)
    assert np.allclose(edges.i1, 210.0, atol=1e-12)


def test_horizontal_partial_area_step_is_located_analytically():
    edge_y = 22.7
    image = partial_area_vertical((64, 48), edge_x=edge_y).T.copy()
    edges = detect(image, 25, order=1)

    assert len(edges) > 20
    assert np.allclose(edges.y, edge_y, atol=1e-12)
    assert np.allclose(edges.nx, 0.0, atol=1e-12)
    assert np.allclose(edges.ny, 1.0, atol=1e-12)


def test_outputs_are_aligned_one_dimensional_arrays():
    edges = detect(partial_area_vertical(), 25)
    fields = (
        edges.position,
        edges.x,
        edges.y,
        edges.nx,
        edges.ny,
        edges.curv,
        edges.i0,
        edges.i1,
    )
    assert all(value.ndim == 1 for value in fields)
    assert len({len(value) for value in fields}) == 1
    assert np.allclose(np.hypot(edges.nx, edges.ny), 1.0)
    assert np.all(edges.i0 <= edges.i1)


def test_upstream_function_name_is_compatible():
    image = partial_area_vertical()
    expected = detect(image, 25, smoothing=0, order=2)
    actual = subpixel_edges(image, 25, 0, 2)
    assert np.array_equal(actual.position, expected.position)
    assert np.array_equal(actual.x, expected.x)


@pytest.mark.parametrize(
    ("image", "error"),
    [
        (np.zeros((20, 20, 1)), ValueError),
        (np.zeros((10, 20)), ValueError),
        (np.full((20, 20), "x"), TypeError),
        (np.full((20, 20), np.nan), ValueError),
    ],
)
def test_rejects_invalid_images(image, error):
    with pytest.raises(error):
        detect(image, 1)


@pytest.mark.parametrize("threshold", [-1, np.inf, True, "1"])
def test_rejects_invalid_threshold(threshold):
    with pytest.raises((TypeError, ValueError)):
        detect(np.zeros((20, 20)), threshold)


@pytest.mark.parametrize("smoothing", [-1, 0.5, True])
def test_rejects_invalid_smoothing(smoothing):
    with pytest.raises((TypeError, ValueError)):
        detect(np.zeros((20, 20)), 1, smoothing=smoothing)


@pytest.mark.parametrize("order", [0, 3, 1.5, True])
def test_rejects_invalid_order(order):
    with pytest.raises((TypeError, ValueError)):
        detect(np.zeros((20, 20)), 1, order=order)
