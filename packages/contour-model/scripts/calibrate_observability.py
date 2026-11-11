"""Measure what PAE and the edge model actually resolve, to set labels_v3 thresholds.

For every accepted extracted observation of a v3 extraction corpus, the source
contour's *construction* labels (before observability rules) are mapped onto the
extracted points (rendered pixel frame). Two detectability tests:

* Arc curvature: points inside a construction arc (ends trimmed) are fitted by a
  line and by a circle. The arc is visible if the line's RMS residual exceeds both
  the circle's RMS by ``ratio`` and the extraction noise floor.
* Short spans: for a short construction instance between two straight neighbours,
  the neighbours' lines are fitted and intersected; the span is visible if the
  extracted points near the corner deviate from that sharp-corner model by more
  than the same deviation at genuinely sharp corners of similar angle (95th pct).

  python scripts/calibrate_observability.py EXTRACT_DIR OUT.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from inspection_shapes.primitive_labels import labels  # noqa: E402
from contour_model.pae_corpus import project  # noqa: E402


def px(mm, settings):
    m = np.asarray(settings['to_mm']['matrix']); o = np.asarray(settings['to_mm']['offset'])
    return np.asarray(mm)@np.linalg.inv(m).T+o


def line_rms(p):
    c = p-p.mean(0); s = np.linalg.svd(c, compute_uv=False)
    return float(s[-1]/np.sqrt(len(p)))


def circle_rms(p):
    A = np.c_[2*p, np.ones(len(p))]; b = (p*p).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:2]; r = np.sqrt(max(sol[2]+c@c, 1e-12))
    return float(np.sqrt(np.mean((np.linalg.norm(p-c, axis=1)-r)**2)))


def fit_line(p):
    c = p.mean(0); _, _, vt = np.linalg.svd(p-c); return c, vt[0]


def intersect(c1, d1, c2, d2):
    den = d1[0]*d2[1]-d1[1]*d2[0]
    if abs(den) < 1e-9:
        return None
    t = ((c2-c1)[0]*d2[1]-(c2-c1)[1]*d2[0])/den
    return c1+t*d1


def corner_deviation(q, apex, d1, d2):
    """Max distance of points q from the two rays meeting at apex (d1 incoming, d2 outgoing)."""
    def seg_dist(points, a, d):
        t = np.clip((points-a)@d, -1e9, 0) if d is d1 else np.clip((points-a)@d, 0, 1e9)
        return np.linalg.norm(points-(a+t[:, None]*d), axis=1)
    return float(np.minimum(seg_dist(q, apex, d1), seg_dist(q, apex, d2)).max())


def analyze(root):
    root = Path(root)
    index = json.loads((root/'index.json').read_text())
    arcs = defaultdict(list); shorts = defaultdict(list); sharp = defaultdict(list)
    for e in index:
        meta = json.loads((root/'sources'/(e['id']+'.json')).read_text()); settings = meta['render']
        with np.load(root/'sources'/(e['id']+'.contours.npz')) as z:
            obs = {k: z[k] for k in z.files}
        with np.load(meta['source_contours']) as z:
            raw = {k: z[k] for k in z.files}
        ann, _ = labels(meta, raw)
        cls, inst = ann['edge_base_class'], ann['edge_base_instance']
        prim = {p['id']: p for p in meta['primitives']}
        truth = px(raw['actual_xy_mm'], settings); ext = px(obs['actual_xy_mm'], settings)
        ro, oo = raw['ring_offsets'], obs['ring_offsets']
        centers = [truth[a:b].mean(0) for a, b in zip(ro[:-1], ro[1:])]
        kind = e['extractor']
        for a, b in zip(oo[:-1], oo[1:]):
            q = ext[a:b]
            i = int(np.argmin([np.linalg.norm(q.mean(0)-c) for c in centers]))
            ref = truth[ro[i]:ro[i+1]]
            idx, s, d, perim = project(q, ref)
            if d.max() > 1.5:
                continue
            src = ro[i]+idx
            ids = inst[src]
            ring_inst = inst[ro[i]:ro[i+1]]
            seg = np.linalg.norm(np.roll(ref, -1, axis=0)-ref, axis=1)
            lengths = {k: seg[ring_inst == k].sum() for k in np.unique(ring_inst)}
            for k in np.unique(ring_inst):
                sel = np.flatnonzero(ids == k)
                if cls[ro[i]+np.flatnonzero(ring_inst == k)[0]] != 2 or len(sel) < 8:
                    continue
                pid = int(raw['edge_primitive_id'][ro[i]+np.flatnonzero(ring_inst == k)[0]])
                p = prim[pid]
                if p['kind'] != 'circular_arc':
                    continue
                sweep = abs(p['nominal_parameters']['end_angle']-p['nominal_parameters']['start_angle'])
                if sweep >= np.pi:
                    continue
                sag_px = p['nominal_parameters']['radius']*(1-np.cos(sweep/2))*settings['ppm']
                trim = sel[len(sel)//8: len(sel)-len(sel)//8]
                if len(trim) < 6:
                    continue
                pts = q[trim]
                arcs[kind].append((sag_px, line_rms(pts), circle_rms(pts), lengths[k]))
            # Short spans and sharp corners between straight neighbours.
            runs = np.flatnonzero(ring_inst != np.roll(ring_inst, 1))
            n = len(ring_inst)
            order = [ring_inst[r] for r in runs]
            for j, k in enumerate(order):
                prev, nxt = order[j-1], order[(j+1) % len(order)]
                first = lambda x: cls[ro[i]+np.flatnonzero(ring_inst == x)[0]]
                if len(order) < 3:
                    continue
                if first(prev) == 1 and first(nxt) == 1 and lengths[k]*settings['ppm'] < 8 and lengths[prev]*settings['ppm'] > 12 and lengths[nxt]*settings['ppm'] > 12:
                    pa, pb = q[ids == prev], q[ids == nxt]
                    if len(pa) < 6 or len(pb) < 6:
                        continue
                    c1, d1 = fit_line(pa); c2, d2 = fit_line(pb)
                    apex = intersect(c1, d1, c2, d2)
                    if apex is None:
                        continue
                    if d1@(apex-c1) < 0: d1 = -d1
                    if d2@(c2-apex) < 0: d2 = -d2
                    near = q[np.linalg.norm(q-apex, axis=1) < max(3., lengths[k]*settings['ppm'])]
                    if len(near) < 2:
                        continue
                    turn = np.degrees(np.arccos(np.clip(d1@d2, -1, 1)))
                    shorts[kind].append((lengths[k]*settings['ppm'], corner_deviation(near, apex, d1, d2), turn))
                if first(k) == 1 and first(nxt) == 1 and lengths[k]*settings['ppm'] > 12 and lengths[nxt]*settings['ppm'] > 12:
                    pa, pb = q[ids == k], q[ids == nxt]
                    if len(pa) < 6 or len(pb) < 6:
                        continue
                    c1, d1 = fit_line(pa); c2, d2 = fit_line(pb)
                    apex = intersect(c1, d1, c2, d2)
                    if apex is None:
                        continue
                    if d1@(apex-c1) < 0: d1 = -d1
                    if d2@(c2-apex) < 0: d2 = -d2
                    near = q[np.linalg.norm(q-apex, axis=1) < 3.]
                    if len(near) >= 2:
                        turn = np.degrees(np.arccos(np.clip(d1@d2, -1, 1)))
                        sharp[kind].append((corner_deviation(near, apex, d1, d2), turn))
    return arcs, shorts, sharp


def summarize(arcs, shorts, sharp, ratio=1.5):
    out = {}
    for kind in sorted(arcs):
        a = np.array(arcs[kind]); noise = float(np.median(a[:, 2]))
        visible = (a[:, 1] > ratio*a[:, 2]) & (a[:, 1] > 2*noise)
        bins = [0, .05, .1, .15, .2, .25, .35, .5, .75, 1, 1.5, 2, 4, 1e9]
        rate = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (a[:, 0] >= lo) & (a[:, 0] < hi)
            if m.sum():
                rate.append(dict(sagitta_px=[lo, hi], arcs=int(m.sum()), visible=round(float(visible[m].mean()), 3)))
        sh = np.array(sharp.get(kind, [])); st = np.array(shorts.get(kind, []))
        srate = []
        if len(sh) and len(st):
            base = {}
            for lo, hi in ((0, 60), (60, 120), (120, 181)):
                m = (sh[:, 1] >= lo) & (sh[:, 1] < hi)
                base[(lo, hi)] = float(np.percentile(sh[m, 0], 95)) if m.sum() > 10 else float(np.percentile(sh[:, 0], 95))
            def threshold(turn):
                return next(v for (lo, hi), v in base.items() if lo <= turn < hi)
            vis = np.array([dev > threshold(t) for _, dev, t in st])
            for lo, hi in ((0, 1), (1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, 4), (4, 5), (5, 6), (6, 8)):
                m = (st[:, 0] >= lo) & (st[:, 0] < hi)
                if m.sum():
                    srate.append(dict(length_px=[lo, hi], spans=int(m.sum()), visible=round(float(vis[m].mean()), 3)))
            out_base = {f'{lo}-{hi}deg': round(v, 3) for (lo, hi), v in base.items()}
        else:
            out_base = {}
        out[kind] = dict(extraction_noise_rms_px=round(noise, 4), arc_visibility=rate, sharp_corner_p95_deviation_px=out_base,
                         short_span_visibility=srate)
    return out


if __name__ == '__main__':
    arcs, shorts, sharp = analyze(sys.argv[1])
    result = summarize(arcs, shorts, sharp)
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=1))
