"""Regression for the old manufacturing IDs colliding with new class IDs."""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for directory in ['packages/measurement/src', 'packages/measurement/benchmarks', 'packages/contour-model/src', 'research/synthetic-shapes/src']:
    sys.path.insert(0, str(ROOT / directory))
from inspection_measurement import Contours

# The benchmark tool runs the torch checkpoint; the app runtime has no torch.
pytest.importorskip('torch', reason='measurement benchmarks need the training environment')
from evaluate_known_truth import match_request  # noqa: E402


@pytest.mark.parametrize('kind,legacy_id,expected', [('line', 4, 'line'), ('slot', 3, 'slot'), ('circle', 2, 'circle')])
def test_match_uses_geometry_identity_and_separate_namespaces(kind, legacy_id, expected):
    c = Contours(np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float), [0, 4])
    case = {'request': {'features': [{'id': 'target', 'type': kind, 'semantic_id': legacy_id,
                                    'selection': {'edge_indices': [0, 1, 2, 3]}}]}}
    base = [{'id': name, 'semantic': name, 'semantic_id': ident, 'label_level': 'base',
             'status': 'accepted', 'confidence': .9, 'raw_edge_indices': [0, 1, 2, 3]}
            for name, ident in [('arbitrary_profile', 4), ('arc', 2), ('circle', 3), ('line', 1)]]
    compound = {'id': 'slot', 'semantic': 'full_round_slot', 'semantic_id': 1, 'label_level': 'compound',
                'status': 'accepted', 'confidence': .9, 'raw_edge_indices': [0, 1, 2, 3]}
    prediction = {'label_version': 'primitives-compounds-v1', 'features': base, 'compound_features': [compound]}
    _, matching = match_request(case, c, prediction)
    assert matching['matches']['target']['prediction_id'] == expected
    for f in base + [compound]: f['status'] = 'abstain'
    request, matching = match_request(case, c, prediction)
    assert matching['matched_target_features'] == 0
    assert 'selection' not in request['features'][0]
