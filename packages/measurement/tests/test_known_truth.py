"""Independent construction truth locks down more than selector consistency."""
import ast
from pathlib import Path
import numpy as np
import pytest
from inspection_measurement import Contours, evaluate
from known_geometry import FAMILIES, generate


@pytest.mark.parametrize('family', FAMILIES)
def test_known_construction_measurements(family):
    case = generate(family, 20260908 + FAMILIES.index(family)*10000)
    contours = Contours.from_dict(case['contours'])
    report = evaluate(contours, case['request'])
    actual = {m['id']: m for m in report['measurements']}
    for truth in case['truth']:
        measurement = actual[truth['measurement_id']]
        assert measurement['status'] == 'measured', measurement
        difference = np.abs(np.asarray(measurement[truth['field']])-truth['expected'])
        if 'period' in truth:
            difference %= truth['period']
            difference = np.minimum(difference, truth['period']-difference)
        assert np.max(difference) <= truth['absolute_tolerance'], (truth, measurement)


def test_generator_truth_is_reproducible_and_does_not_use_fitters():
    import known_geometry
    a = generate('lobed_bore', 415)
    b = generate('lobed_bore', 415)
    assert a == b
    source = ast.parse(Path(known_geometry.__file__).read_text())
    imports = []
    for node in ast.walk(source):
        if isinstance(node, ast.Import): imports.extend(x.name for x in node.names)
        if isinstance(node, ast.ImportFrom): imports.append(node.module)
    assert set(imports) <= {'hashlib', 'pathlib', 'numpy'}
    for expected in a['truth']:
        assert expected['formula'] and 'expected' in expected
    params = a['parameters']['bore']
    circularity = next(t for t in a['truth'] if t['measurement_id'] == 'bore.circularity')
    assert circularity['expected'] == 2*params['radial_amplitude'] > 0
