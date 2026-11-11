import numpy as np
import pytest
from inspection_measurement import contours_from_image, inspect_image, Contours
from inspection_shapes.geometry import generate
from inspection_shapes.render import Camera, render


@pytest.fixture(scope='module')
def washer():
    part = generate('washer', 561, severity=0)
    ppm = 150/np.ptp(part.rings[0].actual, axis=0).max()
    camera = Camera(width=240, height=220, supersample=4, pixels_per_mm=ppm,
                    translation_x=120, translation_y=110, sigma_px=.8,
                    read_noise_dn=0, prnu=0, electrons_per_dn=100000)
    image = render(part, camera, seed=14)[0]['image']
    return image, part, camera


def on_lattice(xy):
    """Edge-model vertices are crossings of the boundary with the 1/16 px lattice lines."""
    lattice = (xy + .5) * 16 - .5
    return np.minimum(*np.abs(lattice - np.round(lattice)).T) < 1e-6


def test_renderer_edge_model_roundtrip_and_measurement(washer):
    image, part, camera = washer
    c, diagnostics = contours_from_image(image)
    assert len(c.closed) == 2
    assert [r['is_hole'] for r in diagnostics['rings']] == [False, True]
    assert [r['depth'] for r in diagnostics['rings']] == [0, 1]
    assert diagnostics['method'] == 'edge_model_rings'
    assert diagnostics['interpolated_point_count'] == 0
    # Every vertex is an unmoved lattice crossing, about 1.1 px apart.
    assert on_lattice(c.xy).all()
    steps = np.linalg.norm(np.diff(c.path(0), axis=0), axis=1)
    assert 1.05 < np.median(steps) < 1.2 and steps.max() < 1.6
    outer = next(r['path'] for r in diagnostics['rings'] if not r['is_hole'])
    req = {'features': [{'id': 'outer', 'type': 'circle', 'selection': {'path': outer}}]}
    report, annotated, _ = inspect_image(image, req, contours=c)
    assert report['summary']['failed_or_unavailable'] == 0
    diameter = next(m['value'] for m in report['measurements'] if m['operator'] == 'least_squares_diameter')
    # sigma 0.8 px render blur; the model sits slightly outward on blurred edges.
    assert abs(diameter-150) < .3
    assert annotated.width > image.shape[1]
    assert annotated.height >= image.shape[0]
    np.testing.assert_array_equal(Contours.from_dict(c.to_dict()).xy, c.xy)
    # Re-extraction is deterministic: JSON edge selections remain stable.
    again, _ = contours_from_image(image)
    assert again.coordinate_hash == c.coordinate_hash


def test_point_spacing(washer):
    image, _, _ = washer
    dense, info = contours_from_image(image, point_spacing=0)
    assert len(dense.xy) == sum(r['lattice_crossings'] for r in info['rings'])
    assert np.linalg.norm(np.diff(dense.path(0), axis=0), axis=1).max() < .1
    coarse, _ = contours_from_image(image, point_spacing=4)
    thinned, _ = contours_from_image(image)
    assert len(coarse.xy) < len(thinned.xy) < len(dense.xy)
    from scipy.spatial import cKDTree
    assert cKDTree(dense.xy).query(coarse.xy)[0].max() < 1e-12


def test_polarity(washer):
    image, _, _ = washer
    c, _ = contours_from_image(image)
    inverted, info = contours_from_image(255-image, polarity='light-on-dark')
    assert info['polarity'] == 'light-on-dark'
    np.testing.assert_array_equal(inverted.xy, c.xy)


def test_part_may_extend_beyond_image_and_close_at_crop_boundary():
    import cv2
    image = np.full((220, 240), 235, np.uint8)
    cv2.rectangle(image, (0, 45), (239, 175), 12, -1)
    cv2.circle(image, (120, 110), 22, 235, -1)
    contours, info = contours_from_image(image)
    assert contours.closed.tolist() == [True, True]
    assert sorted(ring['is_hole'] for ring in info['rings']) == [False, True]
    outer = contours.path(next(ring['path'] for ring in info['rings'] if not ring['is_hole']))
    assert outer[:, 0].min() <= 0 and outer[:, 0].max() >= image.shape[1] - 1
    assert all(r['closed'] == bool(contours.closed[r['path']]) for r in info['rings'])


def test_diagonal_silhouette_may_cross_image_boundary():
    yy, xx = np.ogrid[:220, :240]
    image = np.where(xx + yy < 230, 15, 235).astype(np.uint8)
    contours, info = contours_from_image(image)
    assert contours.closed.tolist() == [True]
    assert contours.path(0)[:, 0].min() <= 0
    assert contours.path(0)[:, 1].min() <= 0
    assert len(contours.path(0)) > 100


def test_circular_roi_boundary_closes_the_silhouette():
    import cv2
    image = np.full((240, 240), 235, np.uint8)
    cv2.rectangle(image, (0, 80), (239, 160), 12, -1)
    yy, xx = np.ogrid[:240, :240]
    valid = (xx + .5 - 120) ** 2 + (yy + .5 - 120) ** 2 <= 90 ** 2
    contours, info = contours_from_image(image, valid_mask=valid)
    assert contours.closed.tolist() == [True]
    # The ROI arc joins the two real edges into one ordinary measured profile.
    assert np.ptp(contours.path(0), axis=0).min() > 80


def test_pixels_outside_roi_are_ignored():
    import cv2
    image = np.full((240, 240), 200, np.uint8)
    cv2.circle(image, (120, 120), 60, 20, -1)
    cv2.circle(image, (120, 120), 25, 200, -1)
    yy, xx = np.ogrid[:240, :240]
    valid = (xx + .5 - 120) ** 2 + (yy + .5 - 120) ** 2 <= 100 ** 2
    first, info = contours_from_image(image, valid_mask=valid)
    assert [ring['is_hole'] for ring in info['rings']] == [False, True]
    image[~valid] = np.random.default_rng(4).integers(0, 255, (~valid).sum(), dtype=np.uint8)
    second, _ = contours_from_image(image, valid_mask=valid)
    np.testing.assert_array_equal(first.xy, second.xy)


def test_minimum_areas_drop_specks_and_what_they_enclose():
    image = np.full((200, 200), 230, np.uint8)
    image[60:140, 60:140] = 15
    image[20:24, 20:24] = 15                    # 16 px^2 speck
    image[20:32, 160:172] = 15                  # 144 px^2 part
    contours, info = contours_from_image(image)
    assert len(contours.closed) == 2 and info['dropped_rings'] >= 1
    contours, _ = contours_from_image(image, min_object_area=200)
    assert len(contours.closed) == 1


def test_bit_depths_share_one_scale(washer):
    image, _, _ = washer
    eight, info8 = contours_from_image(image)
    sixteen, info16 = contours_from_image(image.astype(np.uint16) * 257)
    twelve, _ = contours_from_image((image.astype(np.uint32) * 4095 // 255).astype(np.uint16), max_value=4095)
    assert (info8['input_bit_depth'], info16['input_bit_depth']) == (8, 16)
    from scipy.spatial import cKDTree
    assert cKDTree(eight.xy).query(sixteen.xy)[0].max() < 1e-3
    # 12-bit quantization differs slightly; compare with the dense 8-bit boundary.
    dense, _ = contours_from_image(image, point_spacing=0)
    assert cKDTree(dense.xy).query(twelve.xy)[0].max() < .05


def test_saturated_backlight_part_touching_the_frame():
    import cv2
    image = np.full((160, 200), 300., np.float32)
    image[70:, 40:160] = 15
    cv2.circle(image, (100, 60), 30, 15, -1)
    image = cv2.GaussianBlur(image, (0, 0), 1.) + np.random.default_rng(1).normal(0, 1.5, image.shape)
    contours, info = contours_from_image(np.clip(np.round(image), 0, 255).astype(np.uint8))
    assert len(contours.closed) == 1 and info['backlight_gain'] == pytest.approx(.95)
    assert contours.xy[:, 1].max() >= 159


def test_blank_image_reports_a_diagnostic():
    from inspection_measurement.adapter import ContourExtractionError
    with pytest.raises(ContourExtractionError, match='No silhouette contours') as failure:
        contours_from_image(np.full((64, 64), 220, np.uint8))
    assert failure.value.diagnostic['issue']['bounds']


def test_adapter_rejects_signed_or_color_images():
    with pytest.raises(ValueError, match='8-bit or 16-bit'):
        contours_from_image(np.zeros((20, 20), dtype=np.int16))
    with pytest.raises(ValueError, match='8-bit or 16-bit'):
        contours_from_image(np.zeros((20, 20, 3), dtype=np.uint8))


def test_cli_saved_contours(tmp_path):
    from PIL import Image
    from inspection_measurement.cli import main
    import json
    image = tmp_path/'input.png'
    Image.new('L', (100, 100), 220).save(image)
    c = Contours([[10, 10], [50, 10]], [0, 2], [False])
    contour_file = tmp_path/'c.json'
    contour_file.write_text(json.dumps(c.to_dict()))
    req = tmp_path/'request.json'
    req.write_text(json.dumps({'features': [{'id': 'line', 'type': 'line', 'selection': {'path': 0}}]}))
    output = tmp_path/'output'
    assert main([str(image), str(req), str(output), '--contours', str(contour_file)]) == 0
    report = json.loads((output/'results.json').read_text())
    assert report['summary']['measured'] == 4
    assert (output/'annotated.png').exists()


def test_smooth_background_and_blurred_edges_are_localized():
    from scipy.ndimage import gaussian_filter
    yy, xx = np.mgrid[:220, :360]
    material = (yy >= 90) & (yy <= 130)
    for left in [60, 130, 200, 270]:
        material |= (xx >= left) & (xx <= left+14) & (yy >= 40) & (yy <= 185)
    ideal = gaussian_filter(np.where(material, 15., 230.), 2.)
    illumination = .60 + .35 * xx / 359 + .03 * (yy / 219)**2
    rng = np.random.default_rng(94)
    image = np.clip(np.rint((ideal*illumination+rng.normal(0, .35, ideal.shape))*257), 0, 65535).astype(np.uint16)
    contours, info = contours_from_image(image)
    assert len(contours.closed) == 1
    assert on_lattice(contours.xy).all()
    # Known straight sides, away from junctions/tips, remain localized despite
    # the 35%-of-full-scale illumination ramp and ~5 px edge spread. The model
    # sits up to ~0.2 px outward on strongly blurred edges (edge-model README).
    q = contours.xy
    for left in [60, 130, 200, 270]:
        side = q[(abs(q[:, 0]-(left-.5)) < 2) & (q[:, 1] > 55) & (q[:, 1] < 75)]
        assert len(side) >= 10
        assert np.max(abs(side[:, 0]-(left-.5))) < .3
