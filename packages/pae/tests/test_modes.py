import numpy as np
import pytest

from pae import detect


def step_image(size=24):
    image = np.zeros((size, size), dtype=np.float64)
    image[:, size // 2 :] = 255.0
    return image


@pytest.mark.parametrize("smoothing", [0, 1, 2])
@pytest.mark.parametrize("order", [1, 2])
def test_all_reference_modes_return_finite_aligned_features(smoothing, order):
    edges = detect(step_image(), 25, smoothing=smoothing, order=order)
    assert len(edges) > 0
    for name in ("x", "y", "nx", "ny", "curv", "i0", "i1"):
        values = getattr(edges, name)
        assert values.shape == (len(edges),)
        assert np.all(np.isfinite(values))


@pytest.mark.parametrize('smoothing', [1, 2, 10])
def test_brightness_offset_does_not_change_edge_geometry(smoothing):
    from scipy.ndimage import gaussian_filter
    image = np.zeros((24, 32))
    image[:, 16:] = 100
    image = gaussian_filter(image, .5)+np.random.default_rng(555).normal(0, .3, image.shape)
    first = detect(image, 10, smoothing=smoothing)
    shifted = detect(image+100, 10, smoothing=smoothing)
    # Exclude image margins, where the reference convolution zero-pads.
    a = (first.y > 8) & (first.y < 16)
    b = (shifted.y > 8) & (shifted.y < 16)
    np.testing.assert_array_equal(first.position[a], shifted.position[b])
    for field in ('x', 'y', 'nx', 'ny', 'curv'):
        np.testing.assert_allclose(getattr(first, field)[a], getattr(shifted, field)[b], atol=2e-10, rtol=0)


def test_low_noise_close_edges_are_transpose_equivariant():
    import cv2
    image = np.full((36, 44), 220.)
    image[:, 15:20] = 20
    image = cv2.GaussianBlur(image, (0, 0), .7)
    image += np.random.default_rng(33).normal(0, 1, image.shape)
    a = detect(image, 10, smoothing=1)
    b = detect(image.T, 10, smoothing=1)
    transformed = (b.position % image.shape[1])*image.shape[0]+b.position//image.shape[1]
    ia, ib = np.argsort(a.position), np.argsort(transformed)
    np.testing.assert_array_equal(a.position[ia], transformed[ib])
    np.testing.assert_allclose(a.x[ia], b.y[ib], atol=1e-11)
    np.testing.assert_allclose(a.y[ia], b.x[ib], atol=1e-11)
