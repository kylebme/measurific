"""Torch-free ContourTransformer inference with ONNX Runtime.

The transformer's ring bookkeeping (window gathers, pooling buckets and context
interpolation) is data-dependent Python in ``model.py``. Here the host computes
the same index arrays with numpy and the exported graphs only do tensor math:

- ``encoder.onnx``: embedding, local windows, pooled global encoder, decode
- ``context.onnx``: direct global-context attention for one chunk of tokens
- ``head.onnx``: gated context, refine window and every output head

Chunking the context attention on the host keeps memory linear in the number
of tokens, as ``model.py`` does, without a data-dependent loop in the graph.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

FORMAT = 'contour-transformer-onnx-v1'
CONTEXT_CHUNK = 256
OUTPUT_NAMES = ('primary', 'semantic', 'boundary', 'defect', 'compound', 'query_class', 'query_masks')
GPU_PROVIDERS = ('CUDAExecutionProvider', 'DmlExecutionProvider', 'CoreMLExecutionProvider')


def window_indices(offsets, window, shift):
    """Gather/restore indices for one LocalBlock (see ``LocalBlock.forward``)."""
    gather, restore, start = [], [], 0
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        n = hi - lo
        pad = (-n) % window
        gather.append(lo + (np.arange(n + pad) - shift) % n)
        restore.append(start + (np.arange(n) + shift) % n)
        start += n + pad
    return np.concatenate(gather).astype(np.int64), np.concatenate(restore).astype(np.int64)


def pooling_indices(offsets, token_count, pooled_tokens, adaptive_sampling):
    """Pooling buckets and coarse-context interpolation (see ``ContourTransformer.forward``)."""
    sources, buckets, lefts, rights, fractions, start = [], [], [], [], [], 0
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        n = hi - lo
        size = min(n, max(1 if adaptive_sampling else 4, round(pooled_tokens * n / token_count)))
        for j in range(size):
            indices = np.arange(j * n // size, ((j + 1) * n + size - 1) // size) + lo
            sources.append(indices)
            buckets.append(np.full(len(indices), start + j))
        pos = (np.arange(n, dtype=np.float32) + .5) * size / n - .5
        left = np.floor(pos).astype(np.int64)
        lefts.append(start + left % size)
        rights.append(start + (left + 1) % size)
        fractions.append(pos - left)
        start += size
    bucket = np.concatenate(buckets).astype(np.int64)
    return {
        'pool_source': np.concatenate(sources).astype(np.int64),
        'pool_bucket': bucket,
        'pool_count': np.bincount(bucket, minlength=start).astype(np.float32),
        'context_left': np.concatenate(lefts).astype(np.int64),
        'context_right': np.concatenate(rights).astype(np.int64),
        'context_fraction': np.concatenate(fractions).astype(np.float32)[:, None],
    }


def host_inputs(config, x, offsets):
    offsets = [int(v) for v in offsets]
    shifts = [(i % 2) * (config.window // 2) for i in range(config.local_depth + 1)]
    windows = [window_indices(offsets, config.window, shift) for shift in shifts]
    encoder = {
        'x': np.ascontiguousarray(x, dtype=np.float32),
        'local_gather': np.stack([gather for gather, _ in windows[:-1]]),
        'local_restore': np.stack([restore for _, restore in windows[:-1]]),
        **pooling_indices(offsets, len(x), config.pooled_tokens, config.adaptive_sampling),
    }
    refine_gather, refine_restore = windows[-1]
    return encoder, {'refine_gather': refine_gather, 'refine_restore': refine_restore}


def select_providers(device='cpu', available=None):
    """Execution providers for ``cpu`` or ``gpu``; ``gpu`` falls back to CPU when none is available."""
    if device not in ('cpu', 'gpu'):
        raise ValueError('Compute device must be cpu or gpu')
    if available is None:
        import onnxruntime
        available = onnxruntime.get_available_providers()
    gpu = [name for name in GPU_PROVIDERS if name in available]
    if device == 'cpu' or not gpu:
        return ['CPUExecutionProvider']
    return [gpu[0], 'CPUExecutionProvider']


def is_onnx_model(path) -> bool:
    path = Path(path)
    return path.is_dir() and (path / 'model.json').is_file()


class OnnxContourModel:
    """Duck-types the pieces of ``ContourTransformer`` used by ``inference_primitives``."""

    def __init__(self, directory, device='cpu', threads=None):
        import onnxruntime

        self.directory = Path(directory)
        metadata = json.loads((self.directory / 'model.json').read_text())
        if metadata.get('format') != FORMAT:
            raise ValueError(f'Unsupported ONNX model format: {metadata.get("format")!r}')
        self.metadata = metadata
        self.config = SimpleNamespace(**metadata['model_config'])
        self.checkpoint = metadata['checkpoint']
        options = onnxruntime.SessionOptions()
        if threads:
            options.intra_op_num_threads = int(threads)
        requested = select_providers(device)
        self.sessions = {name: onnxruntime.InferenceSession(str(self.directory / f'{name}.onnx'), options, providers=requested)
                         for name in ('encoder', 'context', 'head')}
        self.providers = self.sessions['head'].get_providers()

    def run_prepared(self, prepared, instances=True):
        encoder_inputs, head_inputs = host_inputs(self.config, prepared.x, prepared.offsets)
        z, memory = self.sessions['encoder'].run(['z', 'memory'], encoder_inputs)
        context = self.sessions['context']
        attended = np.concatenate([context.run(['attended'], {'tokens': z[i:i + CONTEXT_CHUNK], 'memory': memory})[0]
                                   for i in range(0, len(z), CONTEXT_CHUNK)])
        outputs = self.sessions['head'].run(list(OUTPUT_NAMES), {'z': z, 'attended': attended, **head_inputs})
        result = dict(zip(OUTPUT_NAMES, outputs))
        if not instances:
            result.pop('query_class')
            result.pop('query_masks')
        return result

    def describe(self):
        return {
            'format': FORMAT,
            'directory': str(self.directory),
            'providers': self.providers,
            'label_version': self.checkpoint.get('label_version'),
            'source_checkpoint_sha256': self.metadata.get('provenance', {}).get('source_checkpoint_sha256'),
        }


def load_recognition_model(path, device='cpu', threads=None):
    """Load an exported ONNX directory, or a torch checkpoint for development."""
    if is_onnx_model(path):
        model = OnnxContourModel(path, device=device, threads=threads)
        return model, model.checkpoint
    from .train import load_model
    return load_model(path, 'cpu')
