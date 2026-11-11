"""Render v3 designs, extract with the GUI's PAE and/or edge model, transfer labels.

Each selected design gets one render (``render_v3``). The same image is passed to
each assigned extractor, so PAE and edge-model observations of one render are
paired. Labels are first recomputed at the render's pixel scale
(``labels_v3.observable_labels``), then transferred to the extracted contour.

Unlike the strict PAE-era transfer, a render is not discarded because a tiny
feature vanished or a fibre touched the outline. Instead, extracted edges that are
far from the source contour, that backtrack, or that lie around a lost source
instance get zero loss weight. Only a lost outer ring, too much ignored outer
length, or an extractor failure rejects the observation.

  python -m contour_model.extract_v3 CORPUS OUTPUT --pae-limit 300 --edge-limit 150 --workers 4
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
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
from .data import load_cached, ring_geometry
from .labels_v3 import observable_labels, weighted_targets, FIELDS as LABEL_FIELDS
from .pae_corpus import project
from .render_v3 import render, to_mm
from .sampling import prepare_adaptive

EDGE_MODEL = Path(__file__).resolve().parents[3]/'edge-model'/'models'
OFFSETS = {'pae': 300000000, 'edge': 400000000}
CARRY = LABEL_FIELDS+('edge_feature_id', 'edge_anomaly_class', 'edge_label_weight')
_MODELS = {}


def extractor(kind):
    if kind not in _MODELS:
        from inspection_measurement.adapter import load_edge_model, load_pae_edge_model
        threads = int(os.environ.get('EXTRACT_THREADS', '2'))
        if kind == 'pae':
            _MODELS[kind] = load_pae_edge_model(EDGE_MODEL/'sdf_v5-onnx', threads=threads)
            _MODELS[kind].warmup()
        else:
            # Same weights as the GUI's sdf_v5 ONNX export; torch on the GPU is ~10x faster.
            device = os.environ.get('EDGE_DEVICE', 'gpu')
            _MODELS[kind] = load_edge_model(EDGE_MODEL/('sdf_v5.pt' if device == 'gpu' else 'sdf_v5-onnx'),
                                            device=device, threads=threads)
    return _MODELS[kind]


def _area(q):
    return np.sum(q[:, 0]*np.roll(q[:, 1], -1)-q[:, 1]*np.roll(q[:, 0], -1))/2


def _runs(ids):
    starts = np.flatnonzero(ids != np.roll(ids, 1))
    return ids[starts] if len(starts) else ids[:1]


def transfer_keep(raw, truth, observed, obs_offsets, max_distance=1.5, guard_px=2., dilate=1):
    """Label every extracted edge from the nearest source edge; zero-weight doubtful ones."""
    offsets = np.asarray(raw['ring_offsets'])
    truth_rings = [truth[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    obs_rings = [observed[a:b] for a, b in zip(obs_offsets[:-1], obs_offsets[1:])]
    t_area = np.array([abs(_area(r)) for r in truth_rings]); o_area = np.array([abs(_area(r)) for r in obs_rings])
    t_outer, o_outer = int(np.argmax(t_area)), int(np.argmax(o_area))
    cost = np.full((len(truth_rings), len(obs_rings)), 1e9)
    for i, a in enumerate(truth_rings):
        for j, b in enumerate(obs_rings):
            if (i == t_outer) != (j == o_outer):
                continue
            gate = max(4., .25*np.sqrt(t_area[i]))
            d = np.linalg.norm(a.mean(0)-b.mean(0))
            if d < gate and .5 < o_area[j]/max(t_area[i], 1e-9) < 2.:
                cost[i, j] = d
    rows, cols = linear_sum_assignment(cost)
    pairs = {int(c): int(r) for r, c in zip(rows, cols) if cost[r, c] < 1e9}
    if o_outer not in pairs:
        raise ValueError('Outer ring lost or changed')
    out = {k: [] for k in CARRY}; coords = []; new_offsets = [0]
    outer_poly = Polygon(obs_rings[o_outer]).buffer(0)
    stats = Counter()
    for j, q in enumerate(obs_rings):
        q = q.copy()
        if j not in pairs:
            # An extra opening (for example a hole split by a fibre) is kept as
            # zero-weight context. Separate objects (dust beside the part) and
            # islands inside openings are not part of the silhouette: drop them.
            inside = j != o_outer and outer_poly.contains(Polygon(q)) and not any(
                Polygon(obs_rings[k]).contains(Polygon(q)) for k in range(len(obs_rings)) if k not in (j, o_outer))
            if inside:
                if _area(q) > 0:
                    q = q[::-1].copy()
                mid = (q+np.roll(q, -1, axis=0))/2
                ref = truth_rings[t_outer]
                idx, _, _, _ = project(mid, ref)
                src = offsets[t_outer]+idx
                for k in CARRY:
                    out[k].append(np.asarray(raw[k])[src])
                out['edge_label_weight'][-1] = np.zeros(len(q), np.float32)
                coords.append(q); new_offsets.append(new_offsets[-1]+len(q)); stats['extra_rings'] += 1
            else:
                stats['dropped_rings'] += 1
            continue
        i = pairs[j]; ref = truth_rings[i]
        if np.sign(_area(ref)) != np.sign(_area(q)):
            q = q[::-1].copy()
        n = len(q)
        _, s, d, perimeter = project(q, ref)
        mid = (q+np.roll(q, -1, axis=0))/2
        idx, smid, dmid, _ = project(mid, ref)
        src = offsets[i]+idx
        weight = np.asarray(raw['edge_label_weight'], np.float32)[src].copy()
        bad = (dmid > max_distance) | (d > max_distance) | (np.roll(d, -1) > max_distance)
        ds = (np.roll(s, -1)-s+perimeter/2) % perimeter-perimeter/2
        bad |= ds < -.25
        # Source instances that vanished entirely: ignore extracted edges around them.
        ids = np.asarray(raw['edge_base_instance'])
        ring_ids = ids[offsets[i]:offsets[i+1]]
        lost = set(np.unique(ring_ids))-set(np.unique(ids[src]))
        if lost:
            seg = np.linalg.norm(np.roll(ref, -1, axis=0)-ref, axis=1); arc = np.r_[0, np.cumsum(seg)]
            m = len(ring_ids)
            for ident in lost:
                mask = ring_ids == ident
                for start in np.flatnonzero(mask & ~np.roll(mask, 1)) if not mask.all() else [0]:
                    k, length = start, 0.
                    while mask[k % m] and k-start < m:
                        length += seg[k % m]; k += 1
                    rel = (smid-(arc[start]-guard_px)) % perimeter
                    bad |= rel <= length+2*guard_px
            stats['lost_instances'] += len(lost)
        for _ in range(dilate):
            bad = bad | np.roll(bad, 1) | np.roll(bad, -1)
        weight[bad] = 0.
        stats['edges'] += n; stats['ignored_edges'] += int(bad.sum())
        if i == t_outer:
            seg = np.linalg.norm(np.roll(q, -1, axis=0)-q, axis=1)
            stats['outer_ignored_fraction'] = float(seg[weight == 0].sum()/seg.sum())
        for k in CARRY:
            out[k].append(np.asarray(raw[k])[src] if k != 'edge_label_weight' else weight)
        coords.append(q); new_offsets.append(new_offsets[-1]+n)
    stats['lost_rings'] = len(truth_rings)-len(pairs)
    result = {k: np.concatenate(v) for k, v in out.items()}
    result.update(actual_xy_px=np.concatenate(coords), ring_offsets=np.array(new_offsets))
    return result, dict(stats)


def write_one(job):
    entry, source, output, kinds, save_png, max_outer_ignored = job
    root = Path(output); folder = root/'sources'
    meta_path = Path(source)/Path(entry['file']).parent/'sources'/(entry['id']+'.json')
    raw_path = meta_path.with_name(entry['id']+'.contours.npz')
    receipts = [root/'receipts'/(str(OFFSETS[k]+int(entry['id']))+'.json') for k in kinds]
    if all(r.exists() for r in receipts):
        return [json.loads(r.read_text()) for r in receipts]
    raw = load_cached(raw_path); meta = json.loads(meta_path.read_text())
    seed = int(entry['id'])*13+770001
    image, truth, settings = render(raw, seed)
    annotations, groups, weight = observable_labels(meta, raw, settings['ppm'])
    labelled = dict(raw, **annotations, edge_label_weight=weight)
    results = []
    if save_png:
        png = folder/f'{entry["id"]}.render.png'
        Image.fromarray(image).save(png)
    for kind in kinds:
        ident = str(OFFSETS[kind]+int(entry['id']))
        receipt = root/'receipts'/(ident+'.json')
        if receipt.exists():
            results.append(json.loads(receipt.read_text())); continue
        audit = dict(id=entry['id'], family=entry['family'], group=entry.get('group'), split=entry['split'], extractor=kind,
                     **{k: v for k, v in settings.items() if k not in ('to_mm', 'contamination')},
                     contamination=len(settings['contamination']))
        try:
            from inspection_measurement import contours_from_image
            extracted, diagnostics = contours_from_image(image, model=extractor(kind), max_value=settings['max_value'])
            observed, evidence = transfer_keep(labelled, truth, extracted.xy, extracted.offsets)
            if evidence.get('outer_ignored_fraction', 0) > max_outer_ignored:
                raise ValueError('Too much of the outer contour is unlabelled')
            observed['actual_xy_mm'] = to_mm(observed.pop('actual_xy_px'), settings)
            ring_geometry(observed)
            prepared = prepare_adaptive(observed)
            annotations_obs = {k: observed[k] for k in LABEL_FIELDS}
            t, token_weights = weighted_targets(meta, observed, prepared, annotations_obs, groups, observed['edge_label_weight'])
            np.savez_compressed(root/(ident+'.npz'), x=prepared.x, offsets=prepared.offsets, weights=token_weights, **t)
            np.savez_compressed(folder/(ident+'.contours.npz'), **observed)
            meta_out = dict(meta, sample_id=ident, parent_sample_id=entry['id'], render=settings, transfer=evidence,
                            source_contours=str(raw_path), observation_pipeline=f'render-v3-to-{kind}',
                            extraction_method=diagnostics.get('method'), compound_instances=groups)
            (folder/(ident+'.json')).write_text(json.dumps(meta_out))
            row = dict(entry, id=ident, parent_id=entry['id'], extractor=kind, file=ident+'.npz', tokens=len(prepared.x),
                       ignored_fraction=float(1-token_weights.sum()/prepared.weights.sum()), extent_px=settings['extent_px'])
            result = dict(row=row, audit=dict(audit, accepted=True, **evidence))
        except ValueError as exc:
            result = dict(row=None, audit=dict(audit, accepted=False, reason=str(exc)[:200]))
        tmp = receipt.with_suffix('.tmp'); tmp.write_text(json.dumps(result)); tmp.replace(receipt)
        results.append(result)
    return results


def select(index, splits, limit, seed):
    """Group-balanced selection: round-robin over (group, split) cells."""
    entries = [e for e in index if e['split'] in splits]
    rng = np.random.default_rng(seed); rng.shuffle(entries); buckets = {}
    for e in entries:
        buckets.setdefault((e.get('group'), e['split']), []).append(e)
    chosen = []
    while (not limit or len(chosen) < limit) and any(buckets.values()):
        for key in sorted(buckets, key=str):
            if buckets[key] and (not limit or len(chosen) < limit):
                chosen.append(buckets[key].pop())
    return chosen


def parse_plan(plan):
    """'train:4000:2000,val:500:500' -> {split: (pae designs, edge designs)}."""
    out = {}
    for item in plan.split(','):
        split, pae, edge = item.split(':'); out[split] = (int(pae), int(edge))
    return out


def build(source, output, pae_limit=300, edge_limit=150, splits='train,val,test,holdout,stress', workers=4,
          seed=1881, save_png=True, max_outer_ignored=.25, plan=None):
    source = Path(source); output = Path(output)
    (output/'sources').mkdir(parents=True, exist_ok=True); (output/'receipts').mkdir(exist_ok=True)
    index = json.loads((source/'index.json').read_text())
    # Edge-model observations are a subset of the PAE renders: paired observations.
    plan = parse_plan(plan) if plan else {tuple(splits.split(',')): (pae_limit, edge_limit)}
    jobs = []
    for split, (pae_n, edge_n) in plan.items():
        chosen = select(index, list(split) if isinstance(split, tuple) else [split], max(pae_n, edge_n), seed)
        jobs += [(e, str(source), str(output), (['pae'] if i < pae_n else [])+(['edge'] if i < edge_n else []),
                  save_png, max_outer_ignored) for i, e in enumerate(chosen)]
    config = json.loads((source/'config.json').read_text())
    config.update(source='render-v3-extract', parent_source=str(source),
                  plan={(','.join(k) if isinstance(k, tuple) else k): v for k, v in plan.items()}, renderer='render_v3', transfer='keep-with-ignore-v1',
                  edge_model='sdf_v5 (torch gpu, ONNX-equivalent)', pae_model='sdf_v5-onnx coarse + PAE',
                  source_hash=hashlib.sha256(b''.join((Path(__file__).parent/n).read_bytes()
                                                      for n in ('extract_v3.py', 'render_v3.py', 'labels_v3.py'))).hexdigest())
    (output/'config.json').write_text(json.dumps(config, indent=2))
    rows, audit = [], []

    def collect(results):
        for r in results:
            if r['row']:
                rows.append(r['row'])
            audit.append(r['audit'])
        if len(audit) % 50 < len(results):
            print(f'extract {len(audit)} attempted, {len(rows)} accepted', flush=True)
            (output/'index.json').write_text(json.dumps(rows))
    # PAE runs on CPU workers. The edge model then runs in this process only, on one
    # GPU context; renders are deterministic per seed, so both see the same image.
    pae_jobs = [j[:3]+(['pae'],)+j[4:] for j in jobs if 'pae' in j[3]]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for results in pool.map(write_one, pae_jobs, chunksize=1):
            collect(results)
    for job in jobs:
        if 'edge' in job[3]:
            collect(write_one(job[:3]+(['edge'], False)+job[5:]))
    (output/'index.json').write_text(json.dumps(rows, indent=1)); (output/'audit.json').write_text(json.dumps(audit))
    summary = dict(accepted=len(rows), attempted=len(audit),
                   by_extractor={k: dict(Counter(a['accepted'] for a in audit if a['extractor'] == k)) for k in ('pae', 'edge')},
                   rejections=dict(Counter(f"{a['extractor']}: {a.get('reason')}" for a in audit if not a['accepted']).most_common(10)))
    (output/'summary.json').write_text(json.dumps(summary, indent=2)); print(json.dumps(summary, indent=1), flush=True)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('source'); p.add_argument('output')
    p.add_argument('--pae-limit', type=int, default=300); p.add_argument('--edge-limit', type=int, default=150)
    p.add_argument('--splits', default='train,val,test,holdout,stress'); p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=1881); p.add_argument('--no-png', dest='save_png', action='store_false')
    p.add_argument('--plan', help="Per-split design counts 'split:pae:edge,...' (overrides limits/splits)")
    build(**vars(p.parse_args()))
