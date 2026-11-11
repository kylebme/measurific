"""Side silhouettes of turned parts with the asymmetries real profiles have.

Compared with the mirrored ``geometry``/``expanded`` shafts:

* A helical thread's two visible flanks are offset by half a pitch, so the lower
  silhouette is the upper one shifted by P/2 (with a small phase jitter), not its
  mirror image. Multi-start or annular-looking cases keep a small in-phase share.
* Threads have sampled flank angles and crest/root truncations, standard-like
  pitches, start chamfers, runout (vanishing depth) or a relief groove.
* Grooves include rectangular, V, trapezoidal and radiused (U) bottoms.
* Keyway seats and wrench flats can cut one side only; cross holes appear when
  their axis faces the camera.
* Dimensions are quantized like drawing dimensions.

Shoulder fillets, groove-corner radii, rounded thread roots and edge breaks come
from ``realism.add_corner_details`` applied afterwards.
"""
import numpy as np
from .geometry import Part, Chain, circle
from .realism import loguniform

PITCHES = [.35, .4, .45, .5, .6, .7, .75, .8, 1., 1.25, 1.5, 1.75, 2., 2.5, 3.]
TRAPEZOID_PITCHES = [1.5, 2., 3., 4., 5., 6.]


def q(value, step):
    return float(np.round(value/step)*step)


def _line(a, b, role, group=None):
    return ('line', np.asarray(a, float), np.asarray(b, float), role, group)


def _arc(c, r, a0, a1, role, group=None):
    return ('arc', np.asarray(c, float), float(r), float(a0), float(a1), role, group)


def _end(seg):
    if seg[0] == 'line':
        return seg[2]
    c, r, a1 = seg[1], seg[2], seg[4]
    return c+r*np.array([np.cos(a1), np.sin(a1)])


def thread_template(rng, style):
    """One pitch of a generic thread: x fractions and crest flags, plus depth/pitch."""
    if style == 'v':
        half_angle = np.deg2rad(rng.uniform(27.5, 30.))
        crest, root = rng.uniform(.03, .13), rng.uniform(.03, .2)
    elif style == 'trapezoid':
        half_angle = np.deg2rad(rng.uniform(7., 16.))
        crest, root = rng.uniform(.25, .4), rng.uniform(.2, .35)
    else:
        half_angle = np.deg2rad(rng.uniform(0., 2.))
        crest, root = rng.uniform(.42, .5), rng.uniform(.42, .5)
    flank = max((1-crest-root)/2, .02)
    total = crest+root+2*flank
    crest, root, flank = crest/total, root/total, flank/total
    # Depth follows the flank angle (vertical flanks get a square-thread depth).
    depth = flank/np.tan(half_angle) if half_angle > np.deg2rad(1.) else rng.uniform(.4, .55)
    u = np.array([0, root/2, root/2+flank, root/2+flank+crest, root/2+2*flank+crest, 1.])
    z = np.array([0, 0, 1, 1, 0, 0.])
    return u, z, float(min(depth, .62))


def thread_points(x0, x1, pitch, phase, u, z, r, depth, runout=0., chamfer_start=None, chamfer_end=None):
    """Upper-side thread polyline over [x0, x1]; crest at r, root at r-depth."""
    k0, k1 = int(np.floor((x0-phase)/pitch))-1, int(np.ceil((x1-phase)/pitch))+1
    fx = (phase+(np.arange(k0, k1+1)[:, None]+u[None, :-1])*pitch).ravel()
    fz = np.tile(z[:-1], k1-k0+1)
    keep = (fx > x0+1e-9) & (fx < x1-1e-9)
    xs = np.r_[x0, fx[keep], x1]
    zs = np.r_[np.interp(x0, fx, fz), fz[keep], np.interp(x1, fx, fz)]
    ramp = np.ones_like(xs)
    if runout > 0:
        ramp = np.clip((x1-xs)/runout, 0, 1)
    y = r-depth*(1-zs)*ramp
    points = np.c_[xs, y]
    for chamfer in (chamfer_start, chamfer_end):
        if chamfer is not None:
            points = _polyline_min(points, chamfer)
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-7]
    return points[keep]


def _polyline_min(points, line):
    """Pointwise minimum of an x-monotone polyline and the line y = a + b*x."""
    a, b = line
    out = []
    for p, n in zip(points[:-1], points[1:]):
        fp, fn = p[1]-(a+b*p[0]), n[1]-(a+b*n[0])
        out.append(p if fp <= 0 else np.array([p[0], a+b*p[0]]))
        if fp*fn < 0 and abs(n[0]-p[0]) > 1e-12:
            t = fp/(fp-fn); x = p[0]+t*(n[0]-p[0])
            out.append(np.array([x, a+b*x]))
    last = points[-1]
    out.append(last if last[1] <= a+b*last[0] else np.array([last[0], a+b*last[0]]))
    out = np.asarray(out)
    # Vertical flanks under a chamfer collapse to repeated points.
    return out[np.r_[True, np.linalg.norm(np.diff(out, axis=0), axis=1) > 1e-9]]


def plan(rng):
    """Shared dimensions and features of one turned part (both silhouette sides)."""
    R = q(loguniform(rng, 2.5, 16.), .05)
    count = int(rng.choice([1, 2, 3, 4, 5, 6, 8], p=[.08, .2, .25, .2, .12, .1, .05]))
    segments = []
    radius = R
    for i in range(count):
        if i:
            radius = q(np.clip(radius*np.exp(rng.normal(0, .3)), .35*R, R), .05)
        content = str(rng.choice(['plain', 'thread', 'grooves', 'keyway', 'flat'], p=[.36, .35, .15, .07, .07]))
        if content == 'thread' and segments and segments[-1]['content'] == 'thread':
            content = 'plain'  # adjacent zones would share one straight crest land
        length = q(radius*loguniform(rng, .6, 6.), .1)
        segment = dict(radius=radius, length=max(length, .6), content=content)
        if content == 'thread':
            style = str(rng.choice(['v', 'trapezoid', 'square'], p=[.5, .32, .18]))
            u, z, depth_ratio = thread_template(rng, style)
            choices = [p for p in (TRAPEZOID_PITCHES if style != 'v' else PITCHES) if p*depth_ratio < .22*radius]
            if not choices:
                segment['content'] = content = 'plain'
            else:
                # ISO-like: coarse pitch ~0.14 d (trapezoidal ~0.2 d), fine series ~0.6 of that.
                ideal = 2*radius*(.14 if style == 'v' else .2)*(.6 if rng.random() < .3 else 1.)*float(np.exp(rng.normal(0, .15)))
                pitch = float(min(choices, key=lambda p: abs(np.log(p/ideal))))
                in_phase = rng.random() < .1
                segment.update(style=style, u=u, z=z, pitch=pitch, depth=pitch*depth_ratio,
                               phase=float(rng.uniform(0, pitch)),
                               bottom_shift=float(0. if in_phase else pitch/2+rng.normal(0, .03*pitch)),
                               end=str(rng.choice(['runout', 'relief', 'plain'], p=[.45, .4, .15])),
                               runout=float(pitch*rng.uniform(.8, 2.5)), turns=int(rng.integers(5, 40)))
                segment['length'] = max(segment['length'], q(pitch*segment['turns']+2*pitch, .1))
        if content == 'grooves':
            k = int(rng.integers(1, 4))
            segment['grooves'] = [dict(width=q(radius*loguniform(rng, .12, .5), .05), depth=q(radius*rng.uniform(.06, .28), .05),
                                       style=str(rng.choice(['rect', 'v', 'trap', 'u'], p=[.4, .15, .15, .3]))) for _ in range(k)]
            need = sum(g['width'] for g in segment['grooves'])*2.2+radius*.4
            segment['length'] = max(segment['length'], q(need, .1))
        if content in ('keyway', 'flat'):
            segment.update(cut_depth=q(radius*rng.uniform(.08, .3), .05), cut_length=q(segment['length']*rng.uniform(.3, .75), .1),
                           cut_style=str(rng.choice(['square', 'sled'], p=[.6, .4])), cutter=radius*rng.uniform(1.5, 4.),
                           both=bool(content == 'flat' and rng.random() < .5))
        segments.append(segment)
    transitions = [str(rng.choice(['shoulder', 'taper'], p=[.75, .25])) for _ in range(count-1)]
    taper_angles = [float(np.deg2rad(rng.choice([5., 7.5, 10., 15., 20., 30., 45., 60.]))) for _ in range(count-1)]
    ends = []
    for _ in range(2):
        kind = str(rng.choice(['chamfer', 'radius', 'sharp'], p=[.65, .12, .23]))
        ends.append(dict(kind=kind, size=q(loguniform(rng, .2, 1.5), .05), angle=float(np.deg2rad(rng.choice([45., 30., 60.])))))
    return dict(segments=segments, transitions=transitions, taper_angles=taper_angles, ends=ends,
                cross_hole=bool(rng.random() < .12), cross_fraction=float(rng.uniform(.2, .8)))


def _cut(segments, xa, xb, r, seg, role, group):
    """Keyway/flat notch spanning the middle of a land, from (xa, r) to (xb, r)."""
    d, length, style = seg['cut_depth'], seg['cut_length'], seg['cut_style']
    xc = (xa+xb)/2; left, right = xc-length/2, xc+length/2
    if style == 'square':
        segments += [_line((xa, r), (left, r), 'flat'), _line((left, r), (left, r-d), role, group),
                     _line((left, r-d), (right, r-d), role, group), _line((right, r-d), (right, r), role, group),
                     _line((right, r), (xb, r), 'flat')]
        return
    R = max(seg['cutter'], d*1.05); cy = r-d+R; w = np.sqrt(R*R-(R-d)**2)
    left, right = max(left, xa+w+.05), min(right, xb-w-.05)
    if right <= left:
        segments.append(_line((xa, r), (xb, r), 'flat')); return
    t1 = np.arctan2(r-cy, -w)
    segments += [_line((xa, r), (left-w, r), 'flat'), _arc((left, cy), R, t1, -np.pi/2, role, group),
                 _line((left, r-d), (right, r-d), role, group), _arc((right, cy), R, -np.pi/2, -np.pi-t1, role, group),
                 _line((right+w, r), (xb, r), 'flat')]


def _groove(segments, xc, r, g, group):
    w, d, style = g['width'], g['depth'], g['style']
    a, b = xc-w/2, xc+w/2
    if style == 'rect':
        segments += [_line((a, r), (a, r-d), 'groove', group), _line((a, r-d), (b, r-d), 'groove', group),
                     _line((b, r-d), (b, r), 'groove', group)]
    elif style == 'v':
        segments += [_line((a, r), (xc, r-d), 'groove', group), _line((xc, r-d), (b, r), 'groove', group)]
    elif style == 'trap':
        h = w*.225
        segments += [_line((a, r), (xc-h, r-d), 'groove', group), _line((xc-h, r-d), (xc+h, r-d), 'groove', group),
                     _line((xc+h, r-d), (b, r), 'groove', group)]
    else:
        rho = w/2
        if d >= rho:
            cy = r-d+rho
            segments += [_line((a, r), (a, cy), 'groove', group) if d > rho+1e-9 else None,
                         _arc((xc, cy), rho, np.pi, 2*np.pi, 'groove', group),
                         _line((b, cy), (b, r), 'groove', group) if d > rho+1e-9 else None]
            segments[:] = [s for s in segments if s is not None]
        else:
            # Shallow radiused groove: a circular arc below the land.
            R = (rho*rho+d*d)/(2*d); cy = r-d+R
            t = np.arctan2(r-cy, -rho)
            segments.append(_arc((xc, cy), R, t, -np.pi-t, 'groove', group))


def side_profile(spec, side):
    """Upper-frame profile (y > 0) of one silhouette side, left to right."""
    segs, out = spec['segments'], []
    ends = spec['ends']; x = 0.
    total = sum(s['length'] for s in segs)+sum(
        abs(segs[i]['radius']-segs[i+1]['radius'])/np.tan(spec['taper_angles'][i])
        for i, t in enumerate(spec['transitions']) if t == 'taper')

    def end_drop(end, r):
        if end['kind'] == 'chamfer':
            return min(end['size'], .35*r), min(end['size'], .35*r)/np.tan(end['angle'])
        if end['kind'] == 'radius':
            return min(end['size'], .35*r), min(end['size'], .35*r)
        return 0., 0.

    first, last = segs[0], segs[-1]
    current = None
    for i, seg in enumerate(segs):
        r, xa = seg['radius'], x; xb = x+seg['length']
        start_cut = i == 0; end_cut = i == len(segs)-1
        pieces = []
        if seg['content'] == 'thread':
            depth, pitch = seg['depth'], seg['pitch']
            phase = seg['phase']+(seg['bottom_shift'] if side == 'bottom' else 0.)
            x0, x1 = xa, xb
            relief = None
            if seg['end'] == 'relief' and not end_cut:
                width = q(max(pitch*1.5, .4), .05); land = q(max(.3, pitch*.6), .05); x1 = xb-width-land
                relief = dict(width=width, depth=q(depth*1.15, .01))
            # Ends of the part get a thread start chamfer to below the root.
            cs = (r-depth*1.15-x0, 1.) if start_cut else None
            ce = (r-depth*1.15+x1, -1.) if end_cut else None
            runout = seg['runout'] if seg['end'] == 'runout' and not end_cut else 0.
            points = thread_points(x0, x1, pitch, phase % pitch, seg['u'], seg['z'], r, depth, runout, cs, ce)
            group = ('thread', i)
            for a, b in zip(points[:-1], points[1:]):
                pieces.append(_line(a, b, 'thread', group))
            if relief:
                # The thread runs straight into a rectangular relief (undercut).
                cur, d, w = points[-1], relief['depth'], relief['width']
                if cur[1] > r-d+1e-9:
                    pieces.append(_line(cur, (x1, r-d), 'groove', ('relief', i, side)))
                pieces += [_line((x1, r-d), (x1+w, r-d), 'groove', ('relief', i, side)),
                           _line((x1+w, r-d), (x1+w, r), 'groove', ('relief', i, side)),
                           _line((x1+w, r), (xb, r), 'flat')]
        else:
            e0 = end_drop(ends[0], r) if start_cut else (0., 0.)
            e1 = end_drop(ends[1], r) if end_cut else (0., 0.)
            la, lb = xa+e0[1], xb-e1[1]
            if seg['content'] == 'grooves':
                gs = seg['grooves']; span = lb-la
                centers = la+span*(np.arange(len(gs))+1)/(len(gs)+1)
                cur = la
                for k, (g, c) in enumerate(zip(gs, centers)):
                    pieces.append(_line((cur, r), (c-g['width']/2, r), 'flat'))
                    _groove(pieces, c, r, g, ('groove', i, k, side))
                    cur = c+g['width']/2
                pieces.append(_line((cur, r), (lb, r), 'flat'))
            elif seg['content'] in ('keyway', 'flat') and (side == 'top' or seg['both']):
                _cut(pieces, la, lb, r, seg, 'notch', ('notch', i, side))
            else:
                pieces.append(_line((la, r), (lb, r), 'flat'))
            if start_cut and e0[0] > 0:
                d, w = e0
                if ends[0]['kind'] == 'chamfer':
                    pieces.insert(0, _line((xa, r-d), (la, r), 'chamfer'))
                else:
                    pieces.insert(0, _arc((xa+d, r-d), d, np.pi, np.pi/2, 'fillet'))
            if end_cut and e1[0] > 0:
                d, w = e1
                if ends[1]['kind'] == 'chamfer':
                    pieces.append(_line((lb, r), (xb, r-d), 'chamfer'))
                else:
                    pieces.append(_arc((xb-d, r-d), d, np.pi/2, 0., 'fillet'))
        start = pieces[0][1] if pieces[0][0] == 'line' else pieces[0][1]+pieces[0][2]*np.array([np.cos(pieces[0][3]), np.sin(pieces[0][3])])
        if current is not None and np.linalg.norm(start-current) > 1e-9:
            out.append(_line(current, start, 'shoulder' if abs(start[0]-current[0]) < 1e-9 else 'taper'))
        out.extend(pieces)
        current = _end(pieces[-1])
        x = xb
        if i < len(segs)-1:
            nxt = segs[i+1]['radius']
            if spec['transitions'][i] == 'taper' and abs(nxt-r) > 1e-6:
                x += abs(nxt-r)/np.tan(spec['taper_angles'][i])
    return out, total


def _reverse(seg):
    if seg[0] == 'line':
        return ('line', seg[2], seg[1], seg[3], seg[4])
    return ('arc', seg[1], seg[2], seg[4], seg[3], seg[5], seg[6])


def _mirror(seg):
    if seg[0] == 'line':
        return ('line', seg[1]*[1, -1], seg[2]*[1, -1], seg[3], seg[4])
    return ('arc', seg[1]*[1, -1], seg[2], -seg[3], -seg[4], seg[5], seg[6])


def turned_part(rng, family='turned_part'):
    spec = plan(rng)
    p = Part(family)
    root = p.feature('external_profile', generator='turned-v1')
    shared = {}
    top, length = side_profile(spec, 'top')
    bottom, _ = side_profile(spec, 'bottom')
    for seg in top+bottom:
        if seg[0] == 'line' and np.linalg.norm(seg[2]-seg[1]) < 1e-7:
            raise ValueError('Degenerate turned profile segment')
    shift = -_end(top[-1])[0]/2
    chain = Chain(p)

    def feature(role, group):
        if group is None:
            return p.feature(role, root)
        if group not in shared:
            params = {}
            if role == 'thread':
                s = spec['segments'][group[1]]
                params = dict(pitch=s['pitch'], depth=s['depth'], style=s['style'], domain='linear',
                              half_pitch_offset=bool(abs(s['bottom_shift']) > 1e-9), standard_match=None)
            shared[group] = p.feature(role, root, **params)
        return shared[group]

    def emit(seg):
        fid = feature(seg[-2], seg[-1])
        if seg[0] == 'line':
            a, b = seg[1]+[shift, 0], seg[2]+[shift, 0]
            if chain.points:
                if np.linalg.norm(chain.points[-1][-1]-a) > 1e-6:
                    raise ValueError('Disconnected turned profile')
                a = chain.points[-1][-1]
            chain.line(a, b, fid)
        else:
            chain.arc(seg[1]+[shift, 0], seg[2], seg[3], seg[4], fid)
            if len(chain.points) > 1:
                chain.points[-1][0] = chain.points[-2][-1]
    for seg in [_reverse(s) for s in reversed(top)]:
        emit(seg)
    start_bottom = (bottom[0][1] if bottom[0][0] == 'line' else bottom[0][1]+bottom[0][2]*np.array([np.cos(bottom[0][3]), np.sin(bottom[0][3])]))*[1, -1]+[shift, 0]
    chain.line(chain.points[-1][-1], start_bottom, p.feature('flat', root))
    for seg in [_mirror(s) for s in bottom]:
        emit(seg)
    top_start = chain.points[0][0]
    chain.line(chain.points[-1][-1], top_start, p.feature('flat', root))
    chain.points[-1][-1] = chain.points[0][0].copy()
    chain.close()
    if spec['cross_hole']:
        plain = [i for i, s in enumerate(spec['segments']) if s['content'] == 'plain']
        if plain:
            i = plain[0]; x = sum(s['length'] for s in spec['segments'][:i])
            seg = spec['segments'][i]
            circle(p, [x+seg['length']*spec['cross_fraction']+shift, 0], q(seg['radius']*rng.uniform(.12, .3), .05), 'hole', True)
    p.patterns.append(dict(type='turned', segments=len(spec['segments']),
                           threads=sum(s['content'] == 'thread' for s in spec['segments'])))
    return p
