"""Fixed autoresearch benchmark: raw-edge instance quality on extracted and original test contours.

Primary score: base instance F1 (IoU>=.5) on every accepted edge-model test render
(`edge`). `edge_val` is used for tuning decisions; `orig` is clean generator contours.
The `pae*` sets are the superseded PAE extractor, kept for reference.
Diagnostics split false-positive instances into fragments (mostly inside one
same-class reference instance: over-segmentation) and mislabels (mostly inside a
reference instance of another class), and report segments per reference instance.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import torch
from contour_model.train import load_model
from contour_model.data import remap_path
from contour_model.inference_primitives import predict, base_names_for_schema, COMPOUND_NAMES
from contour_model.validate_primitives import instance_counts, summarize

DISK = Path('/media/kyle/T7 Shield/inspection-contour-training')
SETS = {
    'edge': (DISK/'primitives-edge-v1-bench', 'test', None),
    'edge_val': (DISK/'primitives-edge-v1-bench', 'val', None),
    'orig': (DISK/'primitives-intent-v1-20260920', 'test', 300),
    'hires': (DISK/'primitives-edge-v1-bench-hires', 'test', None),
    'hires_val': (DISK/'primitives-edge-v1-bench-hires', 'val', None),
    'pae': (DISK/'primitives-pae-v1-20260921', 'test', None),
    'pae_val': (DISK/'primitives-pae-v1-20260921', 'val', None),
    'orig_pilot': (DISK/'experiments-pae-v1-20260921/original-eval-cache', 'test', None),
}
# Corpus v3 (DATASET_V3.md): observable labels, zero-weight edges excluded from scoring.
# A fourth element filters extraction rows by extractor (GUI default: pae).
V3 = DISK/'v3-20261018'
for _split, _suffix in (('test', ''), ('val', '_val'), ('holdout', '_holdout'), ('stress', '_stress')):
    for _kind in ('pae', 'edge'):
        SETS[f'v3_{_kind}{_suffix}'] = (V3/'extract', _split, None, _kind)
    SETS[f'v3_clean{_suffix}'] = (V3/'corpus', _split, 300, None)


# Families whose reference decomposition is conventional (smooth freeform curves cut
# into arcs at arbitrary tangent-continuous breakpoints) or sub-resolution teeth.
AMBIGUOUS = {'timing_pulley', 'lobed_coupling', 'cam', 'eccentric_cam'}


def edge_weights(c):
    return np.concatenate([np.linalg.norm(np.roll(c['actual_xy_mm'][lo:hi], -1, axis=0)-c['actual_xy_mm'][lo:hi], axis=1)
                           for lo, hi in zip(c['ring_offsets'][:-1], c['ring_offsets'][1:])])


def resolvable_counts(truth_ids, truth_classes, features, w, names, min_len, ignored=None):
    """Instance counts ignoring references shorter than min_len (sub-resolution
    details) and edges in ``ignored``; predictions lying mostly on ignored edges are
    ignored too."""
    lengths = np.bincount(truth_ids, weights=w)
    tiny = np.flatnonzero((lengths < min_len) & (np.arange(len(lengths)) != 0))
    ignored = np.isin(truth_ids, tiny) | (False if ignored is None else ignored)
    ids = truth_ids.copy(); ids[ignored] = 0
    kept = [f for f in features if w[f['raw_edge_indices']][~ignored[f['raw_edge_indices']]].sum() >= .5*w[f['raw_edge_indices']].sum()]
    ww = np.where(ignored, 0., w)
    return instance_counts(ids, truth_classes, kept, ww, names)


def diagnose(truth_ids, truth_classes, features, w):
    """Classify each predicted instance by where its length lies in the reference."""
    out = dict(fp_fragment=0, fp_mislabel=0, fp_merge=0, predicted=len(features), edge_correct=0., edge_total=float(w.sum()))
    pred_class = np.zeros(len(truth_ids), int)
    for f in features: pred_class[f['raw_edge_indices']] = f['semantic_id']
    out['edge_correct'] = float(w[pred_class == truth_classes].sum())
    pieces = {}
    for f in features:
        ids = np.asarray(f['raw_edge_indices']); fw = w[ids]; t = truth_ids[ids]
        if fw.sum() <= 0:
            out['predicted'] -= 1; continue  # entirely on zero-weight (ignored) edges
        share = np.bincount(t, weights=fw); major = int(share.argmax()); frac = share[major]/fw.sum()
        tmask = truth_ids == major; tlen = w[tmask].sum()
        inter = share[major]; iou = inter/(fw.sum()+tlen-inter)
        same = truth_classes[np.flatnonzero(tmask)[0]] == f['semantic_id']
        if frac >= .5: pieces.setdefault(major, []).append(f)
        if iou >= .5 and same: continue
        if frac >= .5 and same: out['fp_fragment'] += 1
        elif frac >= .5: out['fp_mislabel'] += 1
        else: out['fp_merge'] += 1
    refs = [v for v in np.unique(truth_ids) if v != 0]
    out['references'] = len(refs)
    out['segments_per_reference'] = float(np.mean([len(pieces.get(v, [])) for v in refs])) if refs else 0.
    out['over_split_references'] = int(sum(len(pieces.get(v, [])) > 1 for v in refs))
    return out


def run(checkpoint, name, limit=None, device='cuda', predict_kwargs=None):
    root, split, cap, *extractor = SETS[name]; limit = limit or cap
    extractor = extractor[0] if extractor else None
    # Extraction corpora rewrite index.json while building; audit.json marks completion.
    while (root/'receipts').exists() and not (root/'audit.json').exists():
        print(f'waiting for {root.name} to finish building', flush=True); time.sleep(120)
    geometric = str(checkpoint) == 'geometric'
    if geometric:
        from contour_model.geometric import predict as geometric_predict
        names = base_names_for_schema('primitives-compounds-v2')
    elif Path(checkpoint).is_dir():
        # Exported ONNX directory: exactly the GUI's torch-free recognition path.
        from contour_model.onnx_runtime import load_recognition_model
        model = load_recognition_model(Path(checkpoint)); model = model[0] if isinstance(model, tuple) else model
        names = base_names_for_schema(model.config.label_schema)
    else:
        model, ckpt = load_model(checkpoint, device)
        names = base_names_for_schema(model.config.label_schema)
    entries = [e for e in json.loads((root/'index.json').read_text())
               if e['split'] == split and (extractor is None or e.get('extractor') == extractor)]
    entries.sort(key=lambda e: e['id'])
    if limit: entries = entries[:limit]
    base = np.zeros((len(names), 3), int); comp = np.zeros((5, 3), int); exact = np.zeros((len(names), 3), int)
    resolv = np.zeros((len(names), 3), int)
    diag = dict(fp_fragment=0, fp_mislabel=0, fp_merge=0, predicted=0, references=0, over_split_references=0,
                edge_correct=0., edge_total=0.)
    seg = []; lat = []; rows = []
    for e in entries:
        path = remap_path(Path(e['file']) if Path(e['file']).is_absolute() else root/e['file'])
        with np.load(path.parent/'sources'/(e['id']+'.contours.npz')) as z: c = {k: z[k] for k in z.files}
        raw = {k: c[k] for k in ('actual_xy_mm', 'ring_offsets')}
        t0 = time.perf_counter()
        if geometric:
            meta = path.parent/'sources'/(e['id']+'.json')
            ppm = json.loads(meta.read_text()).get('pae_render', {}).get('ppm') if meta.exists() else None
            if ppm: result, _ = geometric_predict(dict(raw, actual_xy_mm=raw['actual_xy_mm']*ppm), units='px')
            else: result, _ = geometric_predict(raw, units='mm')
        else: result, _ = predict(model, raw, instances=True, **(predict_kwargs or {}))
        lat.append(time.perf_counter()-t0)
        w = edge_weights(c)
        ignored = c['edge_label_weight'] == 0 if 'edge_label_weight' in c else None
        if ignored is not None and ignored.any():
            bc = resolvable_counts(c['edge_base_instance'], c['edge_base_class'], result['features'], w, names, 0., ignored)
            cc = resolvable_counts(c['edge_compound_instance'], c['edge_compound_class'], result['compound_features'], w,
                                   COMPOUND_NAMES, 0., ignored)
            w = np.where(ignored, 0., w)
        else:
            bc = instance_counts(c['edge_base_instance'], c['edge_base_class'], result['features'], w, names)
            cc = instance_counts(c['edge_compound_instance'], c['edge_compound_class'], result['compound_features'], w, COMPOUND_NAMES)
        base += bc; comp += cc
        if e['family'] not in AMBIGUOUS: exact += bc
        meta = path.parent/'sources'/(e['id']+'.json')
        ppm = json.loads(meta.read_text()).get('render', {}).get('ppm') if meta.exists() else None
        # Resolvable score: exact families, references >= 4 px long in the rendered image.
        if ppm and e['family'] not in AMBIGUOUS:
            resolv += resolvable_counts(c['edge_base_instance'], c['edge_base_class'], result['features'], w, names, 4./ppm, ignored)
        d = diagnose(c['edge_base_instance'], c['edge_base_class'], result['features'], w)
        for k in diag: diag[k] += d[k]
        seg.append(d['segments_per_reference'])
        rows.append(dict(id=e['id'], family=e['family'], f1=summarize(bc, names)['overall']['f1'], **{k: d[k] for k in ('fp_fragment', 'fp_mislabel', 'fp_merge')}))
    b = summarize(base, names); cm = summarize(comp, COMPOUND_NAMES)
    fam = {}
    for r in rows: fam.setdefault(r['family'], []).append(r['f1'])
    return dict(set=name, samples=len(entries), base=b, compound=cm, exact=summarize(exact, names),
                resolvable=summarize(resolv, names) if resolv.any() else None,
                edge_accuracy=diag['edge_correct']/diag['edge_total'],
                fp_fragment=diag['fp_fragment'], fp_mislabel=diag['fp_mislabel'], fp_merge=diag['fp_merge'],
                over_split_reference_fraction=diag['over_split_references']/max(diag['references'], 1),
                latency_median=float(np.median(lat)), latency_p95=float(np.quantile(lat, .95)),
                family_f1={k: float(np.mean(v)) for k, v in sorted(fam.items())}, rows=rows)


def headline(r):
    b = r['base']; cl = b['classes']
    return dict(f1=round(b['overall']['f1'], 4), P=round(b['overall']['precision'], 4), R=round(b['overall']['recall'], 4),
                macro=round(b['macro_f1'], 4), arcP=round(cl['arc']['precision'], 3), arcR=round(cl['arc']['recall'], 3),
                lineP=round(cl['line']['precision'], 3), lineR=round(cl['line']['recall'], 3),
                edge_acc=round(r['edge_accuracy'], 4), frag=r['fp_fragment'], mislabel=r['fp_mislabel'], merge=r['fp_merge'],
                comp_f1=round(r['compound']['overall']['f1'], 4),
                # Mean over families of mean per-part F1: timing_pulley teeth dominate instance counts.
                exact_f1=round(r['exact']['overall']['f1'], 4) if 'exact' in r else None,
                resolv_f1=round(r['resolvable']['overall']['f1'], 4) if r.get('resolvable') else None,
                fam_macro=round(float(np.mean(list(r['family_f1'].values()))), 4) if r.get('family_f1') else None)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint'); p.add_argument('output')
    p.add_argument('--sets', default='edge,edge_val,orig'); p.add_argument('--limit', type=int)
    p.add_argument('--device', default='cuda'); p.add_argument('--kwargs', default='{}')
    a = p.parse_args(); torch.set_num_threads(4)
    report = dict(checkpoint=str(Path(a.checkpoint).resolve()),
                  sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()[:16] if Path(a.checkpoint).is_file() else None,
                  predict_kwargs=json.loads(a.kwargs), results={})
    for name in a.sets.split(','):
        r = run(a.checkpoint, name, a.limit, a.device, json.loads(a.kwargs)); report['results'][name] = r
        print(name, json.dumps(headline(r)), flush=True)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(report, indent=1))
