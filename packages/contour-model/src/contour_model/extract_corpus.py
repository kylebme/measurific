"""Render imperfect contours, extract with the production edge model, transfer labels.

Successor to ``pae_corpus`` after the GUI switched from PAE to the learned edge
model (``inspection_measurement.contours_from_image`` now runs edge-model ONNX).
Reuses its renderer and strict label transfer (topology/order/1.5 px checks).
Resumable: each attempt writes a receipt; rerunning skips completed attempts.

  python -m contour_model.extract_corpus SOURCE OUTPUT --splits test,val --renders 1 --workers 12
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import numpy as np
from PIL import Image
from .data import load_cached
from .contact_sheets import source_paths
from .sampling import prepare_adaptive
from .corpus_primitives import targets
from .pae_corpus import render, transfer, FIELDS

OFFSET = 200000000  # sample-id namespace for edge-model renders (PAE used 100,000,000)
_MODEL = None


def extractor():
    global _MODEL
    if _MODEL is None:
        from inspection_measurement.adapter import load_edge_model
        _MODEL = load_edge_model(threads=int(os.environ.get('EXTRACT_THREADS', '1')))
    return _MODEL


def write_one(job):
    entry, source, output, render_index, sides, max_tokens, save_png = job
    root = Path(output); ident = str(OFFSET + 10*int(entry['id']) + render_index)
    receipt = root/'receipts'/(ident+'.json')
    if receipt.exists(): return json.loads(receipt.read_text())
    meta_path, raw_path = source_paths(source, entry)
    raw = load_cached(raw_path); meta = json.loads(meta_path.read_text())
    seed = int(entry['id'])*7 + 9173*render_index + 55001
    side = int(np.random.default_rng(seed).choice(sides))
    image, truth, settings = render(raw, seed, side)
    audit = dict(id=entry['id'], family=entry['family'], split=entry['split'], render=render_index, **settings)
    try:
        from inspection_measurement import contours_from_image
        extracted, diagnostics = contours_from_image(image, model=extractor())
        # Extractor coordinates are y-down pixel centers; truth uses the same raster frame.
        observed, evidence = transfer(raw, truth, extracted.xy, extracted.offsets)
        observed['actual_xy_mm'] = (observed['actual_xy_mm']-truth[0])*[1/settings['ppm'], -1/settings['ppm']]+raw['actual_xy_mm'][0]
        annotations = {k: observed[k] for k in FIELDS if k.startswith('edge_base') or k.startswith('edge_compound')}
        prepared = prepare_adaptive(observed)
        if max_tokens and len(prepared.x) > max_tokens: raise ValueError(f'Extracted contour exceeds {max_tokens} tokens')
        t, _, _ = targets(meta, observed, prepared, annotation_override=(annotations, meta['compound_instances']))
        folder = root/'sources'
        np.savez_compressed(root/(ident+'.npz'), x=prepared.x, offsets=prepared.offsets, weights=prepared.weights, **t)
        np.savez_compressed(folder/(ident+'.contours.npz'), **observed)
        if save_png: Image.fromarray(image).save(folder/(ident+'.png'))
        meta.update(sample_id=ident, parent_sample_id=entry['id'], render=settings, transfer=evidence,
                    source_contours=str(raw_path), observation_pipeline='render-to-edge-model',
                    extraction_method=diagnostics.get('method'))
        (folder/(ident+'.json')).write_text(json.dumps(meta))
        row = dict(entry, id=ident, parent_id=entry['id'], file=ident+'.npz', tokens=len(prepared.x))
        for k in ('source_contours', 'source_metadata'): row.pop(k, None)
        result = dict(row=row, audit=dict(audit, accepted=True, **evidence))
    except ValueError as exc:
        result = dict(row=None, audit=dict(audit, accepted=False, reason=str(exc)[:200]))
    tmp = receipt.with_suffix('.tmp'); tmp.write_text(json.dumps(result)); tmp.replace(receipt)
    return result


def select(source, splits, limit, seed):
    entries = [e for e in json.loads((Path(source)/'index.json').read_text()) if e['split'] in splits]
    # Balanced round-robin over family/split cells, randomized within each cell.
    rng = np.random.default_rng(seed); rng.shuffle(entries); buckets = {}
    for e in entries: buckets.setdefault((e['family'], e['split']), []).append(e)
    chosen = []
    while (not limit or len(chosen) < limit) and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and (not limit or len(chosen) < limit): chosen.append(buckets[key].pop())
    return chosen


def build(source, output, splits='test,val', limit=None, renders=1, workers=12, sides='640,960,1280,1920',
          max_tokens=16384, seed=1881, save_png=False):
    source = Path(source); output = Path(output)
    (output/'sources').mkdir(parents=True, exist_ok=True); (output/'receipts').mkdir(exist_ok=True)
    entries = select(source, splits.split(','), limit, seed); sides = [int(v) for v in sides.split(',')]
    config = json.loads((source/'config.json').read_text())
    config.update(source='render-edge-model-v1', parent_source=str(source), splits=splits, renders=renders, sides=sides,
                  max_tokens=max_tokens, attempted=len(entries)*renders,
                  source_hash=hashlib.sha256(Path(__file__).read_bytes()+(Path(__file__).parent/'pae_corpus.py').read_bytes()).hexdigest())
    (output/'config.json').write_text(json.dumps(config, indent=2))
    jobs = [(e, str(source), str(output), r, sides, max_tokens, save_png) for r in range(renders) for e in entries]
    rows = []; audit = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for result in pool.map(write_one, jobs, chunksize=4):
            if result['row']: rows.append(result['row'])
            audit.append(result['audit'])
            if len(audit) % 100 == 0:
                print(f'extract {len(audit)}/{len(jobs)} attempted, {len(rows)} accepted', flush=True)
                (output/'index.json').write_text(json.dumps(rows))
    (output/'index.json').write_text(json.dumps(rows, indent=1)); (output/'audit.json').write_text(json.dumps(audit))
    print(dict(accepted=len(rows), attempted=len(jobs), splits=dict(Counter(r['split'] for r in rows)),
               rejections=dict(Counter(a.get('reason') for a in audit if not a['accepted']).most_common(8))), flush=True)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('source'); p.add_argument('output')
    p.add_argument('--splits', default='test,val'); p.add_argument('--limit', type=int)
    p.add_argument('--renders', type=int, default=1); p.add_argument('--workers', type=int, default=12)
    p.add_argument('--sides', default='640,960,1280,1920'); p.add_argument('--max-tokens', type=int, default=16384)
    p.add_argument('--seed', type=int, default=1881); p.add_argument('--save-png', action='store_true')
    build(**vars(p.parse_args()))
