"""Exercise the worker with the bundled learned model (ONNX)."""
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from worker import InspectionWorker

ROOT = Path(__file__).resolve().parents[2]


def test_camera_png_grayscale_depth_is_preserved():
    pixels = np.array([[0, 1024], [4095, 65535]], dtype=np.uint16)
    assert np.array_equal(InspectionWorker.grayscale_pixels(Image.fromarray(pixels)), pixels)
    color = Image.new('RGB', (2, 2), (255, 0, 0))
    converted = InspectionWorker.grayscale_pixels(color)
    assert converted.dtype == np.uint8 and converted.shape == (2, 2)


EDGE_FORMATS = {'pae': 'edgesr-coarse-pae-v1', 'deep-learning': 'edgesr-sdf-onnx-v1'}
EXTRACTION_METHODS = {'pae': 'coarse_model_pae_rings', 'deep-learning': 'edge_model_rings'}


@pytest.fixture(scope='module', params=['pae', 'deep-learning'])
def worker(request):
    return InspectionWorker(ROOT, backend='learned', edge_method=request.param)


def test_worker_defaults_to_pae_edges():
    assert InspectionWorker.__init__.__kwdefaults__['edge_method'] == 'pae'
    with pytest.raises(ValueError, match='Edge method'):
        InspectionWorker(ROOT, backend='geometry', edge_method='threshold')


def test_worker_reports_the_onnx_runtime_it_uses(worker):
    health = worker.health()
    assert health['model']['format'] == 'contour-transformer-onnx-v1'
    assert health['model']['providers'][0] == 'CPUExecutionProvider'
    assert 'CPUExecutionProvider' in health['availableProviders']
    assert 'onnxruntime' in health['packages']
    assert health['edgeMethod'] == worker.edge_method
    assert health['edgeModel']['format'] == EDGE_FORMATS[worker.edge_method]
    assert health['edgeModel']['providers'][0] == 'CPUExecutionProvider'


def test_worker_closes_a_border_silhouette_for_normal_recognition(worker, tmp_path):
    image = np.full((180, 200), 235, dtype=np.uint8)
    image[35:, 65:135] = 12
    path = tmp_path / 'border-cutoff.png'
    Image.fromarray(image).save(path)
    result = worker.analyze(str(path))
    assert result['report']['extraction']['method'] == EXTRACTION_METHODS[worker.edge_method]
    assert [ring['closed'] for ring in result['report']['extraction']['rings']] == [True]
    fitted = [feature for feature in result['report']['features'] if feature['status'] == 'fitted']
    assert any(feature['type'] == 'line' and feature.get('source') == 'model' for feature in fitted)
    assert not any(feature.get('source') == 'adapter' for feature in result['report']['features'])


def test_worker_treats_clipped_part_with_holes_as_normal_closed_topology(worker, tmp_path):
    image = np.full((220, 300), 235, dtype=np.uint8)
    image[40:180] = 12
    yy, xx = np.ogrid[:220, :300]
    for cx in (90, 210):
        image[(xx - cx) ** 2 + (yy - 110) ** 2 <= 22 ** 2] = 235
    path = tmp_path / 'clipped-with-holes.png'
    Image.fromarray(image).save(path)
    result = worker.analyze(str(path))
    rings = result['report']['extraction']['rings']
    assert [ring['closed'] for ring in rings] == [True, True, True]
    assert sum(ring['is_hole'] for ring in rings) == 2
    features = result['prediction']['features'] + result['prediction']['compound_features']
    assert len({feature['id'] for feature in features}) == len(features)
    assert not any(feature.get('source') == 'adapter' for feature in result['report']['features'])


def test_worker_measures_several_objects(worker, tmp_path):
    # A plate with a hole plus a separate block (or a large piece of dust).
    image = np.full((220, 320), 235, dtype=np.uint8)
    image[40:180, 30:190] = 12
    yy, xx = np.ogrid[:220, :320]
    image[(xx - 110) ** 2 + (yy - 110) ** 2 <= 30 ** 2] = 235
    image[80:140, 230:290] = 12
    path = tmp_path / 'two-objects.png'
    Image.fromarray(image).save(path)
    result = worker.analyze(str(path))
    rings = result['report']['extraction']['rings']
    assert [(ring['is_hole'], ring['parent']) for ring in rings] == [(False, -1), (False, -1), (True, 0)]
    objects = result['prediction']['objects']
    assert [o['ring_indices'] for o in objects] == [[0, 2], [1]]
    features = result['prediction']['features'] + result['prediction']['compound_features']
    assert len({f['id'] for f in features}) == len(features)
    assert {f['parent'] for f in features} == {'object_1', 'object_2'}
    lines = {o['id']: next(f for f in result['report']['features'] if f['type'] == 'line' and f['status'] == 'fitted'
                           and f.get('parent') == o['id']) for o in objects}
    distance = worker.relationship({'operator': 'distance_construction', 'params': {'construction': 'closest_clearance'},
                                    'featureRefs': [lines['object_1']['id'], lines['object_2']['id']]})
    assert distance['value'] > 30


@pytest.mark.parametrize('name', ['drilled_plate', 'stepped_shaft', 'grooved_pin', 'slotted_bracket'])
def test_release_demo_handoff(worker, name):
    result = worker.analyze(str(ROOT / 'app' / 'demo-data' / (name + '.png')))
    prediction, report = result['prediction'], result['report']
    assert prediction['label_version'] == 'primitives-compounds-v2'
    assert report['relations'] == prediction['relations']
    predicted_count = len(prediction['features']) + len(prediction['compound_features'])
    assert len(report['features']) >= predicted_count
    vertices = [f for f in report['features'] if f['type'] == 'point']
    records = {f['id']: f for f in report['features']}
    for vertex in vertices:
        # A junction joins segments; at a corner-excluded gap each segment keeps
        # its own vertex on its retained endpoint.
        if len(vertex['adjacent_feature_ids']) == 1:
            [member] = vertex['adjacent_feature_ids']
            ends = [run[i] for run in records[member]['support'] for i in (0, -1)]
            assert any(np.allclose(vertex['geometry']['center'], end) for end in ends)
        else:
            assert len(vertex['adjacent_feature_ids']) >= 2
    assert report['interaction']['measurement_specs']['symmetry']['count'] == 3
    assert report['interaction']['deformation']['max_factor'] >= 1
    assert report['summary']['measured'] > 0
    for f in report['features']:
        assert f['label_level'] in ('base', 'compound')
        if f['label_level'] == 'compound':
            assert f['base_instance_ids']
            assert 'union' not in f.get('reason', '')
    if name == 'slotted_bracket':
        group = next(f for f in report['features'] if f['type'] == 'slot' and f['status'] == 'fitted')
        base = next(f for f in report['features'] if f['id'] in group['base_instance_ids'] and f['status'] == 'fitted')
        distance = worker.relationship({'featureA': group['id'], 'featureB': base['id'], 'operator': 'distance_construction', 'params': {'construction': 'closest_clearance'}})
        assert distance['value'] == pytest.approx(0., abs=1e-8)
        with pytest.raises(ValueError, match='Unsupported interactive measurement operator'):
            worker.relationship({'featureRefs': [group['id'], base['id']], 'operator': 'distance_statistics'})
        with pytest.raises(ValueError):
            worker.relationship({'featureA': group['id'], 'featureB': base['id'], 'operator': 'angle'})


def test_worker_planar_gdt_measurements(worker):
    result = worker.analyze(str(ROOT / 'app/demo-data/drilled_plate.png'))
    lines = [f for f in result['report']['features'] if f['type'] == 'line' and f['status'] == 'fitted']
    circles = [f for f in result['report']['features'] if f['type'] == 'circle' and f['status'] == 'fitted']
    datum = worker.relationship({'operator': 'datum', 'featureRefs': [lines[0]['id']]})
    assert datum['status'] == 'measured'
    assert datum['datum']['direction']
    concentricity = worker.relationship({'operator': 'concentricity',
                                         'featureRefs': [circles[0]['id'], circles[1]['id']]})
    assert concentricity['status'] == 'measured'
    assert concentricity['value'] >= 0


def test_worker_updates_corner_exclusion_without_rerunning_recognition(worker):
    result = worker.analyze(str(ROOT / 'app/demo-data/drilled_plate.png'))
    assert result['report']['corner_exclusion']['radius_px'] == 1
    updated = worker.set_corner_exclusion(5)
    assert updated['prediction']['corner_exclusion_px'] == 5
    assert updated['report']['corner_exclusion']['radius_px'] == 5
    assert worker.current_report is updated['report']
    restored = worker.set_corner_exclusion(0)
    assert restored['report']['corner_exclusion']['excluded_vertex_count'] == 0


def test_worker_geometry_override_commits_updated_report(worker):
    result = worker.analyze(str(ROOT / 'app/demo-data/drilled_plate.png'))
    circle = next(f for f in result['report']['features'] if f['type'] == 'circle' and f['status'] == 'fitted')
    point = circle['support'][0][0]
    redefined = worker.redefine_geometry({'type': 'circle', 'points': [point]})
    assert redefined['complete'] is True
    assert circle['id'] in redefined['invalidated_feature_ids']
    feature = next(f for f in redefined['report']['features'] if f['id'] == redefined['feature_id'])
    assert feature['type'] == 'circle' and feature['source'] == 'user'
    assert worker.current_report is redefined['report']


def test_worker_remove_compound_restores_primitives(worker):
    result = worker.analyze(str(ROOT / 'app/demo-data/slotted_bracket.png'))
    compound = next(f for f in result['report']['features'] if f.get('label_level') == 'compound' and f.get('support'))
    removed = worker.remove_compound({'featureId': compound['id']})
    assert removed['feature_id'] == compound['id']
    assert removed['invalidated_feature_ids'] == [compound['id']]
    ids = {f['id'] for f in removed['report']['features']}
    assert compound['id'] not in ids and set(removed['restored_feature_ids']) <= ids
    assert worker.current_report is removed['report']
    assert all(f['id'] != compound['id'] for f in worker.current_prediction.get('compound_features', []))

def test_failed_load_clears_previous_relationship_state(worker):
    worker.analyze(str(ROOT / 'app/demo-data/drilled_plate.png'))
    with pytest.raises(FileNotFoundError):
        worker.analyze(str(ROOT / 'app/demo-data/missing.png'))
    with pytest.raises(ValueError, match='Analyze an image'):
        worker.relationship({'featureA': 'base_1', 'featureB': 'base_2', 'operator': 'angle'})


def test_extraction_failure_returns_gui_diagnostic(worker, tmp_path):
    path = tmp_path / 'blank.png'
    Image.fromarray(np.full((80, 100), 128, dtype=np.uint8)).save(path)

    result = worker.analyze_response(str(path))

    assert result['error'] == 'No silhouette contours'
    assert result['width'] == 100 and result['height'] == 80
    assert result['diagnostic']['edges'] == []
    assert result['diagnostic']['issue']['bounds']
    assert worker.current_contours is None


def test_corner_exclusion_remeasures_suspends_and_restores_saved_measurements():
    # The geometric backend splits drilled_plate into short segments that a 5 px
    # exclusion consumes, and it renumbers junction vertices between radii.
    worker = InspectionWorker(ROOT, backend='geometry')
    worker.analyze(str(ROOT / 'app/demo-data/drilled_plate.png'))
    wide = {f['id'] for f in worker.set_corner_exclusion(5)['report']['features'] if f['status'] == 'fitted'}
    worker.set_corner_exclusion(0)
    fitted = {f['id']: f for f in worker.current_report['features'] if f['status'] == 'fitted'}
    # A short straight segment is entirely consumed by a 5 px exclusion.
    vanishing = next(f['id'] for f in fitted.values() if f['type'] == 'line' and f['id'] not in wide)
    vertices = [f for f in fitted.values() if f.get('semantic') == 'vertex']
    saved = [worker.relationship({'id': 'straight', 'operator': 'straightness', 'featureRefs': [vanishing]})]
    stable = next(f for f in fitted.values() if f['type'] == 'circle')
    saved.append(worker.relationship({'id': 'vertex_gap', 'operator': 'distance_construction',
                                      'featureRefs': [vertices[-1]['id'], stable['id']],
                                      'params': {'construction': 'center_distance', 'basis': 'fitted'}}))
    assert saved[1]['feature_anchors'][vertices[-1]['id']]['adjacent_feature_ids']
    requests = [{'id': m['id'], 'operator': m['operator'], 'featureRefs': m['feature_refs'],
                 'params': m.get('definition') or {}, 'featureAnchors': m['feature_anchors']} for m in saved]

    outcomes = {o['id']: o for o in worker.set_corner_exclusion(5, requests)['measurements']}
    assert outcomes['straight']['status'] == 'unavailable'
    gap = outcomes['vertex_gap']
    if gap['status'] == 'measured':
        # The vertex is renumbered, so the anchor rather than its old ID locates it.
        moved = gap['result']['feature_refs'][0]
        assert gap['feature_map'][vertices[-1]['id']] == moved
        assert gap['result']['value'] == pytest.approx(saved[1]['value'], abs=6)

    restored = {o['id']: o for o in worker.set_corner_exclusion(0, requests)['measurements']}
    assert restored['straight']['status'] == 'measured'
    assert restored['straight']['result']['value'] == pytest.approx(saved[0]['value'])
    assert restored['vertex_gap']['result']['value'] == pytest.approx(saved[1]['value'])
    assert restored['vertex_gap']['result']['feature_refs'] == saved[1]['feature_refs']
