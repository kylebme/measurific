from dataclasses import replace
import json
import numpy as np
import pytest
from scipy.special import ndtr
from telecentric import Camera, Illumination, Optics, Sensor, Sampling, RenderConfig, Primitive, Mesh, Scene, render, SensorModel
from telecentric.optics import accepted_directions, psf_kernel
from telecentric.dataset import generate
from telecentric.validation import convergence


def config(**kwargs):
    base = RenderConfig(camera=Camera(width=32, height=24, pixel_pitch_um=10, magnification=1), illumination=Illumination(kind='collimated'), optics=Optics(psf='gaussian', sigma_px=0.65, support_px=4), sampling=Sampling(supersample=8, tile_px=12))
    return replace(base, **kwargs)


def test_camera_conventions():
    c = Camera(width=1920, height=1080, pixel_pitch_um=3.45, magnification=0.05)
    assert c.object_pitch_mm == pytest.approx(0.069)
    np.testing.assert_allclose(c.pixel_to_world([960, 540]), [0, 0])
    xy = np.array([[0.5, 0.5], [123.27, 911.64]])
    np.testing.assert_allclose(c.world_to_pixel(c.pixel_to_world(xy)), xy)


@pytest.mark.parametrize('kind,kwargs', [('sphere', {}), ('cylinder', {'axis': 'y'}), ('cylinder', {'axis': 'z'}), ('box', {})])
def test_analytic_occupancy(kind, kwargs):
    p = Primitive(kind, radius_mm=1, size_mm=(2, 2, 2), length_mm=2, **kwargs)
    origins = np.array([[0, 0, 7], [0.99, 0, 0], [1.01, 0, 0], [0, 1.01, 0]], float)
    expected = [True, True, False, False]
    np.testing.assert_array_equal(p.blocked(origins, np.array([0., 0., 1.])), expected)


def test_cylinder_angled_support():
    p = Primitive('cylinder', radius_mm=1, length_mm=10, axis='y')
    slope = 0.3
    support = np.sqrt(1+slope*slope)
    o = np.array([[support-1e-7, 0, 0], [support+1e-7, 0, 0]])
    np.testing.assert_array_equal(p.blocked(o, np.array([slope, 0, 1.])), [True, False])


def test_angular_acceptance_and_diffuse_radiance():
    opt = Optics(object_na=0.08)
    source = Illumination(kind='diffuse', samples=1024)
    d, w = accepted_directions(source, opt)
    direction_cosines = d/np.linalg.norm(d, axis=1)[:, None]
    r2 = np.sum(direction_cosines[:, :2]**2, axis=1)
    assert r2.max() <= opt.object_na**2
    assert w.sum() == pytest.approx(1)
    assert (r2@w) == pytest.approx(opt.object_na**2/2, rel=1e-3)
    d2, w2 = accepted_directions(replace(source, kind='cone', half_angle_deg=60), opt)
    np.testing.assert_array_equal(d, d2)
    np.testing.assert_array_equal(w, w2)


def test_background_energy_and_off_frame_geometry():
    c = config()
    blank = render(Scene([]), c)
    np.testing.assert_allclose(blank.irradiance, 1, atol=1e-7)
    assert not blank.center_mask.any()
    solid = render(Scene([Primitive('box', size_mm=(10, 10, 1))]), c)
    np.testing.assert_allclose(solid.irradiance, 0, atol=1e-7)
    assert solid.center_mask.all()


def test_tile_and_roi_equivalence():
    c = config(illumination=Illumination(kind='diffuse', samples=8), optics=Optics(psf='mixture', support_px=4))
    scene = Scene([Primitive('sphere', radius_mm=0.082, center_mm=(0.011, 0, 0.1))])
    full = render(scene, c)
    other = render(scene, replace(c, sampling=replace(c.sampling, tile_px=7)))
    patch = render(scene, c, (7, 5, 12, 9))
    np.testing.assert_allclose(full.irradiance, other.irradiance, atol=1e-7)
    np.testing.assert_allclose(full.irradiance[5:14, 7:19], patch.irradiance, atol=1e-7)
    np.testing.assert_array_equal(full.geometric_coverage[5:14, 7:19], patch.geometric_coverage)


def test_fixed_focus_defocus_and_translation_invariance():
    c = config(illumination=Illumination(kind='diffuse', samples=64), optics=Optics(psf='none', object_na=0.2))
    a = Primitive('box', size_mm=(0.16, 2, 0.00001))
    b = replace(a, center_mm=(0, 0, 0.15))
    focused = render(Scene([a]), c)
    defocused = render(Scene([b]), c)
    restored = render(Scene([b]), replace(c, camera=replace(c.camera, focus_z_mm=0.15)))
    assert np.max(np.abs(focused.irradiance-defocused.irradiance)) > 0.1
    np.testing.assert_allclose(focused.irradiance, restored.irradiance, atol=1e-7)
    np.testing.assert_array_equal(focused.center_mask, defocused.center_mask)


def test_straight_edge_against_exact_pixel_integrated_gaussian():
    # Independent closed-form reference: Gaussian CDF integrated over pixel
    # boundaries. Checks phase, PSF scale, and area integration together.
    c = config(sampling=Sampling(supersample=128, tile_px=8))
    phase_errors = []
    for phase in (0.071, 0.293, 0.619, 0.887):
        edge_px = 16 + phase
        edge_mm = (edge_px-c.camera.width/2)*c.camera.object_pitch_mm
        scene = Scene([Primitive('box', center_mm=(edge_mm-0.5, 0, 0), size_mm=(1, 2, 0.001))])
        actual = render(scene, c, (12, 10, 9, 2)).irradiance[0]
        sigma = c.optics.sigma_px

        def anti(x):
            t = (x-edge_px)/sigma
            return (x-edge_px)*ndtr(t)+sigma*np.exp(-t*t/2)/np.sqrt(2*np.pi)

        boundaries = np.arange(12, 22)
        exact = np.diff(anti(boundaries))
        assert np.max(np.abs(actual-exact)) < 0.003
        slope = ndtr((boundaries[1:]-edge_px)/sigma)-ndtr((boundaries[:-1]-edge_px)/sigma)
        phase_errors.append(np.max(np.abs(actual-exact)[slope > 0.15]/slope[slope > 0.15]))
    assert max(phase_errors) < 0.005


def test_geometry_labels_independent_of_optics():
    scene = Scene([Primitive('sphere', radius_mm=0.08, center_mm=(0.001, 0.002, 0.1))])
    labels = scene.labels(config().camera)
    pts = np.asarray(labels[0]['xy_mm'])
    np.testing.assert_allclose(np.linalg.norm(pts-[0.001, 0.002], axis=1), 0.08, atol=1e-14)
    np.testing.assert_allclose(labels[0]['curvature_per_mm'], 12.5)


def test_coincident_outer_boundaries_and_internal_seams():
    a = Primitive('box', size_mm=(0.1, 0.1, 0.01))
    identical = Scene([a, replace(a, name='copy')]).labels(config().camera)
    assert all(identical[0]['visible_in_union'])
    touching = Scene([a, replace(a, center_mm=(0.1, 0, 0))]).labels(config().camera)
    normals = np.asarray(touching[0]['normal_xy'])
    right_side = normals[:, 0] == 1
    assert not np.asarray(touching[0]['visible_in_union'])[right_side].any()


def test_edge_on_mesh_has_empty_projected_contour(tmp_path):
    import trimesh
    path = tmp_path/'sheet.ply'
    trimesh.Trimesh(vertices=[[0, 0, 0], [0, 1, 0], [0, 0, 1]], faces=[[0, 1, 2]]).export(path)
    labels = Scene([Mesh(path)]).labels(config().camera)
    assert labels[0]['xy_mm'] == []


@pytest.mark.parametrize('backend', ['triangle', 'embree'])
def test_mesh_matches_analytic_box_and_hole_projection(tmp_path, backend):
    import trimesh
    if backend == 'embree':
        pytest.importorskip('embreex')
    path = tmp_path/'box.stl'
    trimesh.creation.box(extents=[2, 3, 1]).export(path)
    mesh = Mesh(path, backend=backend)
    analytic = Primitive('box', size_mm=(2, 3, 1))
    rng = np.random.default_rng(1)
    origins = rng.uniform(-3, 3, (1000, 3))
    for direction in (np.array([0., 0., 1.]), np.array([0.1, -0.2, 1.])):
        np.testing.assert_array_equal(mesh.blocked(origins, direction), analytic.blocked(origins, direction))
    contour = mesh.contour()
    xy = contour['xy_mm']
    assert np.all(np.isclose(np.abs(xy[:, 0]), 1) | np.isclose(np.abs(xy[:, 1]), 1.5))
    ringpath = tmp_path/'ring.stl'
    trimesh.creation.annulus(r_min=0.5, r_max=1.0, height=0.2).export(ringpath)
    ring = Mesh(ringpath, backend=backend)
    assert len(np.unique(ring.contour()['ring_id'])) == 2
    np.testing.assert_array_equal(ring.blocked(np.array([[0, 0, 0], [0.7, 0, 0]]), np.array([0., 0., 1.])), [False, True])


@pytest.mark.parametrize('kind', ['gaussian', 'mixture', 'airy', 'measured'])
def test_psf_normalization_and_measured_sampling(tmp_path, kind):
    args = {}
    if kind == 'measured':
        grid = np.arange(-8, 9)
        x, y = np.meshgrid(grid, grid)
        path = tmp_path/'psf.npy'
        np.save(path, np.exp(-(x*x+y*y)/8))
        args = dict(measured_path=str(path), measured_pitch_um=2.0)
    kernel, info = psf_kernel(Optics(psf=kind, **args), config().camera, 8)
    assert kernel.sum() == pytest.approx(1)
    assert np.all(kernel >= 0)
    if kind != 'mixture':
        np.testing.assert_allclose(info['centroid_px'], 0, atol=1e-12)
    else:
        assert info['centroid_px'][0] > 0


def test_sensor_photon_statistics_and_repeatability():
    c = Sensor(bright_e_per_ms=1000, dark_e_per_ms=0, read_noise_e=3, full_well_e=1e7, electrons_per_adu=1, black_level_adu=100, bits=16, prnu_std=0, dsnu_e_per_ms=0)
    sensor = SensorModel(c, (400, 400))
    image = np.ones((400, 400))
    a = sensor.capture(image, seed=7)
    b = sensor.capture(image, seed=7)
    np.testing.assert_array_equal(a['raw'], b['raw'])
    assert a['raw'].mean() == pytest.approx(1100, abs=0.4)
    assert a['raw'].var() == pytest.approx(1009, rel=0.015)
    assert not np.array_equal(a['raw'], sensor.capture(image, seed=8)['raw'])


def test_sensor_persistent_patterns_roi_and_saturation():
    c = Sensor(noise=False, full_well_e=100, electrons_per_adu=1, black_level_adu=0, bits=12, prnu_std=0.03)
    model = SensorModel(c, (20, 30))
    frame = model.capture(np.ones((20, 30)), exposure_ms=1)
    assert frame['saturated'].all()
    assert np.all(frame['raw'] == 100)
    assert np.all(frame['expected_e'] > c.full_well_e)
    a = model.capture(np.full((20, 30), 0.001))
    b = model.capture(np.full((4, 5), 0.001), roi=(3, 2, 5, 4))
    np.testing.assert_array_equal(a['expected_e'][2:6, 3:8], b['expected_e'])


def test_dataset_roundtrip_seeds_and_overwrite_protection(tmp_path):
    job = dict(config=config().to_dict(), shapes=[dict(kind='sphere', radius_mm=0.05)], roi=[8, 8, 8, 8], dataset=dict(count=1, exposures_ms=[0.5, 1], repeat_captures=2))
    path = tmp_path/'job.json'
    path.write_text(json.dumps(job))
    for name in ('a', 'b'):
        manifest = generate(path, tmp_path/name)
        assert manifest['status'] == 'complete'
    with np.load(tmp_path/'a/sample_000000.npz') as a, np.load(tmp_path/'b/sample_000000.npz') as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
        assert a['raw_03'].dtype == np.uint16
    meta = json.loads((tmp_path/'a/sample_000000.json').read_text())
    assert len(meta['captures']) == 4
    assert len(meta['labels']) == 1
    with pytest.raises(ValueError, match='empty'):
        generate(path, tmp_path/'a')


def test_convergence_report_and_empty_scene_not_false_pass():
    c = config()
    scene = Scene([Primitive('box', size_mm=(0.1523, 2, 0.01))])
    report = convergence(scene, c, levels=[16, 32], tolerance_px=0.1)
    assert report['studies']['supersample'][-1]['paired_crossings'] > 0
    empty = convergence(Scene([]), c, levels=[2, 4])
    assert empty['last_comparisons_within_tolerance'] is False


@pytest.mark.parametrize('factory,kwargs', [(Camera, {'magnification': 0}), (Illumination, {'samples': 7}), (Optics, {'object_na': 1}), (Sampling, {'supersample': 0}), (Sensor, {'bits': 17}), (Optics, {'psf': 'measured'}), (Camera, {'width': 3.4})])
def test_invalid_configuration(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)
