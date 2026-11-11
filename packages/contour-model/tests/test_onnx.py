"""The exported ONNX graphs must reproduce the torch checkpoint they came from."""
import json
from pathlib import Path

import numpy as np
import pytest

from contour_model.onnx_runtime import select_providers

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'release-v6'
DEMOS = sorted((ROOT.parents[1] / 'app' / 'demo-data').glob('*.contours.npz'))


def test_gpu_preference_falls_back_to_cpu():
    assert select_providers('cpu', ['CoreMLExecutionProvider', 'CPUExecutionProvider']) == ['CPUExecutionProvider']
    assert select_providers('gpu', ['CPUExecutionProvider']) == ['CPUExecutionProvider']
    assert select_providers('gpu', ['DmlExecutionProvider', 'CPUExecutionProvider']) == ['DmlExecutionProvider', 'CPUExecutionProvider']
    with pytest.raises(ValueError):
        select_providers('auto', ['CPUExecutionProvider'])


def test_committed_export_matches_its_source_checkpoint():
    import hashlib
    metadata = json.loads((RELEASE / 'onnx' / 'model.json').read_text())
    assert metadata['provenance']['source_checkpoint_sha256'] == hashlib.sha256((RELEASE / 'model.pt').read_bytes()).hexdigest()
    for name, digest in metadata['graphs'].items():
        assert hashlib.sha256((RELEASE / 'onnx' / f'{name}.onnx').read_bytes()).hexdigest() == digest


@pytest.mark.skipif(not DEMOS, reason='GUI demo contours are not present')
def test_committed_export_reproduces_torch_logits():
    pytest.importorskip('onnxruntime')
    from contour_model.onnx_export import parity
    from contour_model.train import load_model
    model, _ = load_model(RELEASE / 'model.pt')
    worst, providers = parity(model, RELEASE / 'onnx', DEMOS, tolerance=1e-3)
    assert providers == ['CPUExecutionProvider'] and worst < 1e-3


@pytest.mark.skipif(not DEMOS, reason='GUI demo contours are not present')
def test_onnx_and_torch_select_the_same_features():
    pytest.importorskip('onnxruntime')
    from contour_model.inference_primitives import predict
    from contour_model.onnx_export import load_contours
    from contour_model.onnx_runtime import OnnxContourModel
    from contour_model.train import load_model
    torch_model, _ = load_model(RELEASE / 'model.pt')
    onnx_model = OnnxContourModel(RELEASE / 'onnx')
    for path in DEMOS:
        contours = load_contours(path)
        expected, _ = predict(torch_model, contours, units='px')
        actual, _ = predict(onnx_model, contours, units='px')
        key = lambda f: (f['semantic'], f['status'], f['raw_edge_indices'])
        assert [key(f) for f in actual['features']] == [key(f) for f in expected['features']]
        assert np.allclose([f['confidence'] for f in actual['features']],
                           [f['confidence'] for f in expected['features']], atol=1e-4)
