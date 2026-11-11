"""Export an inference checkpoint to the three-graph ONNX layout in ``onnx_runtime``.

    python -m contour_model.onnx_export CHECKPOINT OUTPUT_DIR [--parity CONTOURS.npz ...]

``--parity`` runs the torch checkpoint and the exported graphs on the given
contour files (``actual_xy_px`` or ``actual_xy_mm`` plus ``ring_offsets``) and
fails if any token logit differs by more than ``--tolerance``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn

from .onnx_runtime import FORMAT, OUTPUT_NAMES, OnnxContourModel, host_inputs
from .sampling import prepare_adaptive
from .train import load_model

OPSET = 18


def run_window(block, x, gather, restore):
    z = x[gather].reshape(-1, block.window, x.shape[-1])
    norm = block.norm(z)
    z = z + block.attn(norm, norm, norm, need_weights=False)[0]
    z = z + block.ff(z)
    return z.reshape(-1, x.shape[-1])[restore]


class Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, local_gather, local_restore, pool_source, pool_bucket, pool_count,
                context_left, context_right, context_fraction):
        m = self.model
        z = m.embed(x)
        for i, block in enumerate(m.local):
            z = run_window(block, z, local_gather[i], local_restore[i])
        pooled = z.new_zeros((pool_count.shape[0], z.shape[-1])).index_add(0, pool_bucket, z[pool_source])
        pooled = pooled / pool_count[:, None]
        coarse = m.global_encoder(m.up(pooled)[None])[0]
        context = coarse[context_left] * (1 - context_fraction) + coarse[context_right] * context_fraction
        memory = m.up(z) + context if m.config.full_context else coarse
        return m.decode(torch.cat([z, context], dim=-1)), memory


class Context(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, memory):
        m = self.model
        return m.context_attention(m.context_norm(tokens)[None], memory[None], memory[None], need_weights=False)[0][0]


class Head(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, z, attended, refine_gather, refine_restore):
        m = self.model
        z = z + m.context_gate.tanh() * attended
        z = run_window(m.refine, z, refine_gather, refine_restore)
        primary = m.primary(z)
        compound = m.compound(torch.cat([z, primary.softmax(-1)], dim=-1))
        q = m.query_decoder(m.queries[None], z[None])[0]
        query_masks = m.query_mask(q) @ m.mask_embed(z).T / (m.config.width ** .5)
        return (primary, m.semantic(z), m.boundary(z).squeeze(-1), m.defect(z), compound,
                m.query_class(q), query_masks)


def check_supported(config):
    problems = []
    if config.backbone != 'transformer':
        problems.append(f'backbone={config.backbone!r}')
    if config.topology_instances:
        problems.append('topology_instances=True')
    if not config.direct_global_context:
        problems.append('direct_global_context=False')
    if config.label_schema not in ('primitives-compounds-v1', 'primitives-compounds-v2'):
        problems.append(f'label_schema={config.label_schema!r}')
    if problems:
        raise ValueError('ONNX export does not support ' + ', '.join(problems))


def load_contours(path):
    with np.load(path, allow_pickle=False) as values:
        xy = values['actual_xy_px'] if 'actual_xy_px' in values else values['actual_xy_mm']
        return {'actual_xy_mm': xy.astype(np.float64), 'ring_offsets': values['ring_offsets'].astype(np.int64)}


def sample_inputs(model):
    """A small synthetic two-ring part, only used to trace the graphs."""
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    s = np.linspace(0, 2 * np.pi, 150, endpoint=False)
    outer = np.c_[200 * np.cos(t), 120 * np.sin(t)]
    inner = np.c_[30 * np.cos(-s), 30 * np.sin(-s)]
    contours = {'actual_xy_mm': np.vstack([outer, inner]), 'ring_offsets': np.array([0, 400, 550])}
    prepared = prepare_adaptive(contours, version=model.config.sampling_version)
    return host_inputs(model.config, prepared.x, prepared.offsets)


def export(module, args, names, outputs, dynamic, path):
    # Parameters keep requires_grad so the exporter traces the composite
    # attention/encoder layers instead of their inference-only fused kernels.
    program = torch.onnx.export(module, tuple(torch.as_tensor(a) for a in args), dynamo=True, opset_version=OPSET,
                                input_names=names, output_names=outputs, dynamic_shapes=dynamic,
                                external_data=False, optimize=True)
    program.save(str(path))


def export_onnx(checkpoint, output, overwrite=False):
    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        if not overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    model, ckpt = load_model(checkpoint, 'cpu')
    check_supported(model.config)
    staging = output.with_name(output.name + '.partial')
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    encoder_inputs, head_inputs = sample_inputs(model)
    tokens, gathered, pooled, sourced, window_tokens = (torch.export.Dim(name, min=2) for name in
                                                        ('tokens', 'gathered', 'pooled', 'sourced', 'window_tokens'))
    depth = model.config.local_depth
    encoder_names = list(encoder_inputs)
    export(Encoder(model), encoder_inputs.values(), encoder_names, ['z', 'memory'], {
        'x': {0: tokens}, 'local_gather': {1: gathered}, 'local_restore': {1: tokens},
        'pool_source': {0: sourced}, 'pool_bucket': {0: sourced}, 'pool_count': {0: pooled},
        'context_left': {0: tokens}, 'context_right': {0: tokens}, 'context_fraction': {0: tokens},
    }, staging / 'encoder.onnx')
    with torch.no_grad():
        z, memory = Encoder(model)(*(torch.as_tensor(v) for v in encoder_inputs.values()))
    memory_tokens = torch.export.Dim('memory_tokens', min=2)
    export(Context(model), (z[:17].numpy(), memory.numpy()), ['tokens', 'memory'], ['attended'], {
        'tokens': {0: torch.export.Dim('chunk', min=1)}, 'memory': {0: memory_tokens},
    }, staging / 'context.onnx')
    export(Head(model), (z.numpy(), z.numpy(), *head_inputs.values()), ['z', 'attended', *head_inputs],
           list(OUTPUT_NAMES), {
               'z': {0: tokens}, 'attended': {0: tokens},
               'refine_gather': {0: window_tokens}, 'refine_restore': {0: tokens},
           }, staging / 'head.onnx')
    assert depth == encoder_inputs['local_gather'].shape[0]
    metadata = {
        'format': FORMAT,
        'model_config': model.metadata(),
        'checkpoint': {key: ckpt[key] for key in ('label_version', 'instances_trained', 'input_contract', 'settings', 'step')
                       if key in ckpt},
        'provenance': {
            'source_checkpoint': checkpoint.name,
            'source_checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'torch_version': torch.__version__,
            'opset': OPSET,
        },
        'graphs': {name: hashlib.sha256((staging / f'{name}.onnx').read_bytes()).hexdigest()
                   for name in ('encoder', 'context', 'head')},
    }
    # Training-only settings (cache paths, validation ids) are not part of the contract.
    metadata['checkpoint'].pop('settings', None)
    (staging / 'model.json').write_text(json.dumps(metadata, indent=2) + '\n')
    staging.replace(output)
    return output, model


def parity(model, onnx_dir, contour_paths, tolerance, device='cpu'):
    """Largest absolute logit difference between the torch model and exported graphs."""
    runtime = OnnxContourModel(onnx_dir, device=device)
    worst = 0.
    for path in contour_paths:
        prepared = prepare_adaptive(load_contours(path), version=model.config.sampling_version)
        with torch.no_grad():
            expected = model(torch.as_tensor(prepared.x), prepared.offsets, instances=True)
        actual = runtime.run_prepared(prepared, instances=True)
        for name in OUTPUT_NAMES:
            difference = float(np.abs(expected[name].numpy() - actual[name]).max())
            worst = max(worst, difference)
            if difference > tolerance:
                raise AssertionError(f'{Path(path).name}: {name} differs by {difference:.3g} (> {tolerance:g}) on {runtime.providers[0]}')
    return worst, runtime.providers


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--parity', type=Path, nargs='*', default=[])
    parser.add_argument('--tolerance', type=float, default=1e-3)
    parser.add_argument('--device', choices=('cpu', 'gpu'), default='cpu', help='execution provider for --parity')
    args = parser.parse_args(argv)
    output, model = export_onnx(args.checkpoint, args.output, args.overwrite)
    print(f'Wrote {output}')
    if args.parity:
        worst, providers = parity(model, output, args.parity, args.tolerance, args.device)
        print(f'Parity on {providers[0]}: max |logit difference| {worst:.3g} over {len(args.parity)} files')


if __name__ == '__main__':
    main()
