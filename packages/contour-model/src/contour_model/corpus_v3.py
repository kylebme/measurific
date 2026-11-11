"""Corpus v3: realistic corners, grammar-free parts, observable labels, rebalanced mix.

Differences from the primitives/intent corpus (``corpus_primitives``):

* Families are drawn from weighted groups instead of cycling 46 families evenly.
  About a third of designs are grammar-free CSG parts, and dense showcase families
  use the ``moderate`` complexity caps, so typical parts carry tens rather than
  hundreds of instances. Mirrored-thread and freeform cam/pulley grammars are
  replaced by ``turned_part`` and tangent-arc curved profiles.
* Every nominal part gets a sampled manufacturing corner style (tool radii,
  fillets, edge breaks, rounded thread roots) before imperfections.
* Labels follow ``labels_v3.observable_labels`` at a reference resolution, with
  zero-weight edges excluded from token weights.
* Splits: train/val/test/stress by design hash, plus ``holdout``: every design of
  the HOLDOUT families (and CSG ``radial`` style) is excluded from training.

  python -m contour_model.corpus_v3 OUTPUT --count 600 --workers 8
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
import multiprocessing
import os
import numpy as np
from inspection_shapes import geometry
from inspection_shapes.expanded import expanded_nominal
from inspection_shapes.realism import add_corner_details
from inspection_shapes.turned import turned_part
from inspection_shapes.csg import csg_part, CURVED
from inspection_shapes.render import feature_bits
from inspection_shapes.primitive_labels import LABEL_VERSION, BASE_NAMES, COMPOUND_NAMES, labels
from .corpus import composed
from .corpus_v2 import DefectBudget
from .labels_v3 import observable_labels, weighted_targets, reference_px_per_mm, POLICY
from .sampling import prepare_adaptive, VERSION

SOURCE = 'corpus-v3'
GROUPS = {
    'csg': (.30, {'csg_part': 1.}),
    'common': (.25, {'mixed_plate': 1., 'perforated_disk': .6, 'drilled_plate': 1., 'slotted_bracket': 1.,
                     'l_bracket': 1., 't_bracket': .8, 'c_frame': .8, 'u_channel_section': .8,
                     'clevis_link': 1., 'dogbone_link': .8, 'd_profile_spacer': .8, 'washer': .5,
                     'flange': 1., 'polygon_spacer': .6, 'mixed_aperture_panel': .6}),
    'turned': (.20, {'turned_part': 4., 'stepped_shaft': .4, 'grooved_pin': .4, 'tapered_pin': .3,
                     'multi_step_shaft': .3, 'groove_stack_pin': .3, 'stepped_taper_arbor': .3}),
    'repeat': (.18, {'toothed_disk': 1.5, 'rack': 1., 'sawtooth_rack': .8, 'internal_spline_ring': 1.,
                     'ratchet_wheel': .8, 'serrated_washer': .6, 'perforated_grid_plate': 1.,
                     'staggered_filter_plate': .7, 'honeycomb_screen': .7, 'polygon_broach_plate': .6,
                     'slotted_grille': .8, 'multi_bolt_circle_flange': .8, 'index_plate': .6,
                     'spoke_wheel': .8, 'comb_plate': .7, 'connector_strip': .7, 'heat_sink_section': .7,
                     'labyrinth_plate': .3}),
    'curved': (.07, {'lobed_profile': 1., 'tangent_cam': 1., 'round_groove_pulley': 1.}),
}
HOLDOUT_FAMILIES = ('t_bracket', 'dogbone_link', 'ratchet_wheel', 'stepped_taper_arbor')
HOLDOUT_CSG_STYLE = 'radial'
LEGACY_SIMPLE = ('drilled_plate', 'slotted_bracket', 'flange', 'washer', 'polygon_spacer',
                 'stepped_shaft', 'grooved_pin', 'tapered_pin', 'toothed_disk', 'rack')


def choose(rng):
    names = list(GROUPS)
    group = names[int(rng.choice(len(names), p=np.array([GROUPS[g][0] for g in names])/sum(GROUPS[g][0] for g in names)))]
    families = GROUPS[group][1]; keys = list(families)
    weights = np.array([families[k] for k in keys]); family = keys[int(rng.choice(len(keys), p=weights/weights.sum()))]
    style = str(rng.choice(['rectilinear', 'union', 'link', 'radial'], p=[.35, .35, .15, .15])) if family == 'csg_part' else None
    return group, family, style


def nominal(family, rng, style=None):
    if family == 'csg_part':
        return csg_part(rng, style)
    if family in CURVED:
        return CURVED[family](rng)
    if family == 'turned_part':
        return turned_part(rng)
    if family in ('mixed_plate', 'perforated_disk'):
        return composed(family, rng)
    if family in LEGACY_SIMPLE and rng.random() < .5:
        return geometry.make_nominal(family, rng)
    return expanded_nominal(family, rng, 'moderate')


def split_for(seed, index, family, style):
    if family in HOLDOUT_FAMILIES or style == HOLDOUT_CSG_STYLE:
        return 'holdout'
    h = int.from_bytes(hashlib.blake2b(f'{seed}:{index}:split-v3'.encode(), digest_size=8).digest(), 'big') % 100
    return 'train' if h < 80 else 'val' if h < 88 else 'test' if h < 96 else 'stress'


def _contours(p):
    return {'nominal_xy_mm': np.concatenate([r.nominal for r in p.rings]),
            'ring_offsets': np.r_[0, np.cumsum([len(r.nominal) for r in p.rings])].astype(np.int64),
            'edge_feature_id': np.concatenate([r.feature for r in p.rings]),
            'edge_primitive_id': np.concatenate([r.primitive for r in p.rings])}


def create_record(index, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, 31337]))
    group, family, style = choose(rng)
    split = split_for(seed, index, family, style)
    stress = split == 'stress' or (split == 'train' and rng.random() < .03)
    severity = float(rng.uniform(.8, 2.) if stress else rng.choice([0., .03, .08, .16, .30], p=[.05, .20, .40, .25, .10]))
    for attempt in range(60):
        try:
            p = nominal(family, rng, style)
        except (ValueError, RuntimeError):
            continue
        if not geometry.valid(p, False):
            continue
        p, corners = add_corner_details(p, rng)
        extent = max(np.ptp(np.concatenate([r.nominal for r in p.rings]), axis=0))
        effective = severity*min(1., 80/extent)
        defects = int(rng.poisson(2. if stress else .35)) if effective else 0
        geometry.imperfect(p, DefectBudget(rng, len(p.rings), defects), effective)
        if not geometry.valid(p):
            continue
        try:
            # Rejects rare constructions where one straight span joins two compounds.
            labels(dict(features=p.features, primitives=p.primitives), _contours(p))
        except ValueError:
            continue
        break
    else:
        raise ValueError(f'Generation failed: {family}, {index}')
    c = {'actual_xy_mm': np.concatenate([r.actual for r in p.rings]),
         'nominal_xy_mm': np.concatenate([r.nominal for r in p.rings]),
         'ring_offsets': np.r_[0, np.cumsum([len(r.actual) for r in p.rings])].astype(np.int64),
         'edge_feature_id': np.concatenate([r.feature for r in p.rings]),
         'edge_primitive_id': np.concatenate([r.primitive for r in p.rings]),
         'edge_anomaly_class': np.concatenate([r.defect for r in p.rings])}
    c['edge_semantic_bits'] = feature_bits(p)[c['edge_feature_id']]
    delta = np.linalg.norm(c['actual_xy_mm']-c['nominal_xy_mm'], axis=1)
    corner_summary = {k: v for k, v in corners.items() if k != 'details'}
    corner_summary['convex'] = sum(d['convex'] for d in corners.get('details', []))
    meta = {'sample_id': f'{index:09d}', 'design_id': index, 'corpus_id': f'{SOURCE}-{seed}', 'family': family,
            'group': group, 'csg_style': style, 'split': split, 'features': p.features, 'primitives': p.primitives,
            'patterns': p.patterns, 'relationships': p.relationships, 'seed': seed, 'severity': severity,
            'effective_severity': effective, 'stress': stress, 'complexity': 'moderate', 'attempts': attempt,
            'corner_style': corner_summary, 'form_displacement_mm': {'p95': float(np.quantile(delta, .95)), 'max': float(delta.max())},
            'anomalies': p.anomalies, 'imperfections': p.imperfections}
    return meta, c


def write_one(job):
    index, seed, destination = job
    root = Path(destination); ident = f'{index:09d}'
    folder = root/'shards'/f'{index//500:05d}'; folder.mkdir(parents=True, exist_ok=True)
    receipt = folder/(ident+'.record.json')
    if receipt.exists():
        return json.loads(receipt.read_text())
    metadata, contours = create_record(index, seed)
    ppm = reference_px_per_mm(contours)
    annotations, groups, weight = observable_labels(metadata, contours, ppm)
    prepared = prepare_adaptive(contours)
    t, token_weights = weighted_targets(metadata, contours, prepared, annotations, groups, weight)
    contours.update(annotations); contours['edge_label_weight'] = weight
    metadata.update(label_version=LABEL_VERSION, label_policy=POLICY, label_px_per_mm=ppm, base_names=BASE_NAMES,
                    compound_names=COMPOUND_NAMES, compound_instances=groups)
    source = folder/'sources'; source.mkdir(exist_ok=True)
    np.savez_compressed(source/(ident+'.contours.npz'), **contours)
    (source/(ident+'.json')).write_text(json.dumps(metadata))
    filename = folder/(ident+'.npz')
    np.savez_compressed(filename, x=prepared.x, offsets=prepared.offsets, weights=token_weights, **t)
    ids = annotations['edge_base_instance']
    row = dict(id=ident, file=str(filename.relative_to(root)), family=metadata['family'], group=metadata['group'],
               csg_style=metadata['csg_style'], split=metadata['split'], design_id=index, tokens=len(prepared.x),
               rings=len(prepared.offsets)-1, instances=int(len(np.unique(ids))), compounds=len(groups),
               stress=metadata['stress'], ignored_fraction=float(1-token_weights.sum()/prepared.weights.sum()))
    tmp = receipt.with_suffix('.tmp'); tmp.write_text(json.dumps(row)); tmp.replace(receipt)
    return row


def fingerprint():
    from inspection_shapes import geometry as g, expanded, primitive_labels, teeth, realism, turned, csg
    from . import corpus, corpus_v2, corpus_primitives, sampling, labels_v3, recache_intent, geometric
    modules = [g, expanded, primitive_labels, teeth, realism, turned, csg, corpus, corpus_v2, corpus_primitives,
               sampling, labels_v3, recache_intent, geometric]
    return hashlib.sha256(b''.join(Path(m.__file__).read_bytes() for m in modules)+Path(__file__).read_bytes()).hexdigest()


def build(output, count=600, seed=20261018, workers=8):
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    config = dict(source=SOURCE, seed=seed, source_hash=fingerprint(), label_version=LABEL_VERSION,
                  label_policy=POLICY, sampling=VERSION, tokens='adaptive', base_names=BASE_NAMES,
                  compound_names=COMPOUND_NAMES, groups={g: dict(weight=w, families=f) for g, (w, f) in GROUPS.items()},
                  holdout_families=list(HOLDOUT_FAMILIES), holdout_csg_style=HOLDOUT_CSG_STYLE)
    config_path = root/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Corpus configuration changed; use a fresh directory')
    lock = root/'.writer.lock'; fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        config_path.write_text(json.dumps(config, indent=2))
        rows = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for i, row in enumerate(pool.map(write_one, ((i, seed, str(root)) for i in range(count)), chunksize=2)):
                rows.append(row)
                if (i+1) % 100 == 0:
                    print(f'Generated {i+1}/{count}', flush=True)
        tmp = root/'index.json.partial'; tmp.write_text(json.dumps(rows, indent=1)); tmp.replace(root/'index.json')
    finally:
        os.close(fd); lock.unlink()
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('output'); p.add_argument('--count', type=int, default=600)
    p.add_argument('--seed', type=int, default=20261018); p.add_argument('--workers', type=int, default=8)
    build(**vars(p.parse_args()))
