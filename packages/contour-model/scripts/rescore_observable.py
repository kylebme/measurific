"""Score models on an old clean corpus with both its stored labels and v3 observable labels.

Separates genuine regressions from the labelling-policy change: the intent-v1
labels mark sub-pixel tooth crowns and similar spans as arcs, which v3 calls lines.

  python scripts/rescore_observable.py OUT.json CORPUS SPLIT LIMIT CHECKPOINT [CHECKPOINT ...]
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import edge_weights, resolvable_counts, AMBIGUOUS  # noqa: E402
from contour_model.data import remap_path  # noqa: E402
from contour_model.inference_primitives import predict, base_names_for_schema  # noqa: E402
from contour_model.labels_v3 import observable_labels  # noqa: E402
from contour_model.train import load_model  # noqa: E402
from contour_model.validate_primitives import instance_counts, summarize  # noqa: E402


def main(out, corpus, split, limit, checkpoints):
    root = Path(corpus)
    entries = sorted([e for e in json.loads((root/'index.json').read_text()) if e['split'] == split], key=lambda e: e['id'])[:limit]
    models = {c: load_model(c, 'cuda')[0] for c in checkpoints}
    names = base_names_for_schema('primitives-compounds-v2')
    totals = {c: {k: np.zeros((len(names), 3), int) for k in ('stored', 'observable', 'stored_exact', 'observable_exact')} for c in checkpoints}
    families = {}
    for e in entries:
        path = remap_path(Path(e['file']) if Path(e['file']).is_absolute() else root/e['file'])
        src = path.parent/'sources'
        meta = json.loads((src/(e['id']+'.json')).read_text())
        with np.load(src/(e['id']+'.contours.npz')) as z:
            c = {k: z[k] for k in z.files}
        if 'nominal_xy_mm' not in c:
            continue
        ann, _, weight = observable_labels(meta, c)
        ignored = weight == 0
        w = edge_weights(c)
        raw = {k: c[k] for k in ('actual_xy_mm', 'ring_offsets')}
        for ckpt, model in models.items():
            result, _ = predict(model, raw, instances=True)
            stored = instance_counts(c['edge_base_instance'], c['edge_base_class'], result['features'], w, names)
            obs = resolvable_counts(ann['edge_base_instance'], ann['edge_base_class'], result['features'], w, names, 0., ignored)
            t = totals[ckpt]; t['stored'] += stored; t['observable'] += obs
            if e['family'] not in AMBIGUOUS:
                t['stored_exact'] += stored; t['observable_exact'] += obs
            f = families.setdefault(e['family'], {}).setdefault(ckpt, np.zeros((2, len(names), 3), int))
            f[0] += stored; f[1] += obs
    report = {}
    for ckpt, t in totals.items():
        report[ckpt] = {k: {'f1': round(summarize(v, names)['overall']['f1'], 4),
                            'arc_recall': round(summarize(v, names)['classes']['arc']['recall'], 3),
                            'arc_precision': round(summarize(v, names)['classes']['arc']['precision'], 3)} for k, v in t.items()}
    report['families'] = {fam: {ckpt: [round(summarize(v[0], names)['overall']['f1'], 3), round(summarize(v[1], names)['overall']['f1'], 3)]
                                for ckpt, v in d.items()} for fam, d in sorted(families.items())}
    Path(out).write_text(json.dumps(report, indent=1))
    for ckpt in checkpoints:
        print(Path(ckpt).parent.name or ckpt, json.dumps(report[ckpt]))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5:])
