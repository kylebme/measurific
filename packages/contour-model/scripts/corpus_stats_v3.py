"""Compare label/geometry statistics of contour corpora (e.g. intent-v1 vs v3 pilot).

  python scripts/corpus_stats_v3.py OUT.json NAME=CORPUS[:N] [NAME=CORPUS[:N] ...]

Statistics use each corpus's stored source contours and reference labels on up
to N training designs: instances/rings per part, base-class length fractions,
arc sweep (turning within each arc instance), tangent junctions, short
instances, compound presence and zero-weight (ignored) length.
"""
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from contour_model.data import remap_path  # noqa: E402


def source_paths(root, entry):
    f = remap_path(Path(entry['file']) if Path(entry['file']).is_absolute() else root/entry['file'])
    folder = f.parent/'sources'
    return folder/(entry['id']+'.json'), folder/(entry['id']+'.contours.npz')


def turning(q):
    a = q[1:-1]-q[:-2]; b = q[2:]-q[1:-1]
    return float(np.abs(np.arctan2(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0], (a*b).sum(1))).sum())


def stats(root, limit):
    root = Path(root)
    index = [e for e in json.loads((root/'index.json').read_text()) if e['split'] == 'train']
    rng = np.random.default_rng(0)
    pick = rng.choice(len(index), size=min(limit, len(index)), replace=False)
    S = {k: [] for k in ('instances', 'rings', 'sweep', 'junction_turn', 'short_frac', 'ignored', 'compound', 'extent')}
    frac = np.zeros(4); families = Counter()
    for k in pick:
        e = index[k]; meta_path, path = source_paths(root, e)
        meta = json.loads(meta_path.read_text()); families[meta['family']] += 1
        with np.load(path) as z:
            a = {n: z[n] for n in z.files}
        q_all = a['nominal_xy_mm']; off = a['ring_offsets']; cls = a['edge_base_class']; inst = a['edge_base_instance']
        weight = a.get('edge_label_weight', np.ones(len(cls)))
        extent = float(np.ptp(q_all, axis=0).max()); S['extent'].append(extent)
        seg = np.zeros(len(cls))
        for lo, hi in zip(off[:-1], off[1:]):
            q = q_all[lo:hi]; seg[lo:hi] = np.linalg.norm(np.roll(q, -1, axis=0)-q, axis=1)
            ids = inst[lo:hi]; starts = np.flatnonzero(ids != np.roll(ids, 1)); n = hi-lo
            for j, s in enumerate(starts):
                e_ = starts[(j+1) % len(starts)]
                idx = np.arange(s, e_ if e_ > s else e_+n) % n
                if cls[lo+s] == 2:
                    S['sweep'].append(np.degrees(turning(q[np.r_[idx, (idx[-1]+1) % n]])))
                a_ = q[s]-q[s-1]; b_ = q[(s+1) % n]-q[s]
                S['junction_turn'].append(np.degrees(abs(np.arctan2(a_[0]*b_[1]-a_[1]*b_[0], a_@b_))))
        lengths = np.bincount(np.unique(inst, return_inverse=True)[1], weights=seg)
        S['short_frac'].append(float((lengths < extent/400).mean()))
        S['instances'].append(len(lengths)); S['rings'].append(len(off)-1)
        frac += np.bincount(cls, weights=seg, minlength=4)/seg.sum()
        S['ignored'].append(float(seg[weight == 0].sum()/seg.sum()))
        S['compound'].append(int((a['edge_compound_class'] > 0).any()))
    pct = lambda v: {p: round(float(np.percentile(v, p)), 3) for p in (5, 25, 50, 75, 95)} if len(v) else {}
    sweep = np.asarray(S['sweep']); turn = np.asarray(S['junction_turn'])
    return {'parts': len(pick), 'families': len(families), 'instances_per_part': pct(S['instances']),
            'rings_per_part': pct(S['rings']), 'extent_mm': pct(S['extent']),
            'length_fraction_line_arc_circle': (frac[1:]/len(pick)).round(3).tolist(),
            'arc_sweep_deg': pct(sweep), 'arcs_under_5deg': round(float((sweep < 5).mean()), 3) if len(sweep) else None,
            'arc_count': int(len(sweep)), 'junction_turn_deg': pct(turn),
            'tangent_junctions_under_3deg': round(float((turn < 3).mean()), 3),
            'short_instances_under_0.25pct_extent': round(float(np.mean(S['short_frac'])), 4),
            'parts_with_compound': round(float(np.mean(S['compound'])), 3),
            'ignored_length_fraction_mean': round(float(np.mean(S['ignored'])), 4),
            'top_families': families.most_common(12)}


if __name__ == '__main__':
    out = Path(sys.argv[1]); result = {}
    for spec in sys.argv[2:]:
        name, _, rest = spec.partition('=')
        path, _, n = rest.rpartition(':') if rest.rsplit(':', 1)[-1].isdigit() else (rest, '', '')
        result[name] = stats(path, int(n or 500))
        print(name, json.dumps(result[name], indent=1), flush=True)
    out.write_text(json.dumps(result, indent=2))
