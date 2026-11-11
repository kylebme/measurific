"""Independent analytic fixture construction: no inspection_measurement imports.

Truth comes from construction parameters and closed-form expressions, never fits
of generated points. Each record distinguishes finite-polyline quantities from
associated smooth geometry. All coordinates and dimensions are pixels.
"""
import hashlib
from pathlib import Path
import numpy as np

VERSION = 'analytic-pixel-parts-v1'
FAMILIES = ('rectangle_holes', 'slotted_plate', 'washer', 'rounded_plate',
            'stepped_plate', 'tapered_plate', 'polar_holes', 'linear_rack',
            'angular_teeth', 'bowed_plate', 'lobed_bore', 'ellipse_bore')


def plain(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [plain(v) for v in x]
    return x


class Builder:
    def __init__(self, family, seed):
        self.family, self.seed = family, seed
        self.rng = np.random.default_rng(seed)
        self.theta = float(self.rng.uniform(-.60, .60))
        self.rotation = np.array([[np.cos(self.theta), -np.sin(self.theta)],
                                  [np.sin(self.theta), np.cos(self.theta)]])
        self.origin = self.rng.uniform([470, 350], [550, 410])
        self.rings, self.holes, self.features, self.specs, self.truth = [], [], [], [], []
        self.parameters = {'rotation_radians': self.theta, 'origin': self.origin,
                           'generation_units': 'px', 'random_generator': 'numpy.PCG64'}
        self.offset = 0

    def world(self, q):
        return np.asarray(q)@self.rotation.T+self.origin

    def axis(self, q):
        return np.asarray(q)@self.rotation.T

    def add_feature(self, ident, kind, semantic, indices=None, **kwargs):
        feature = {'id': ident, 'type': kind, 'semantic_id': semantic,
                   'source': 'user', **kwargs}
        if indices is not None:
            feature['selection'] = {'edge_indices': list(map(int, indices))}
        self.features.append(feature)
        return ident

    def check(self, ident, operator, refs, expected, formula, params=None,
              tolerance=1e-5, field='value', category='exact', period=None, units=None):
        mid = ident
        if not any(s['id'] == mid for s in self.specs):
            self.specs.append({'id': mid, 'operator': operator, 'feature_refs': refs,
                               **({'params': params} if params else {})})
        if units is None:
            units = 'deg' if operator in ('orientation', 'angle', 'parallelism', 'perpendicularity', 'angular_span') else 'px'
        self.truth.append({'measurement_id': mid, 'operator': operator, 'field': field,
                           'expected': expected, 'absolute_tolerance': tolerance,
                           'units': units, 'formula': formula, 'truth_category': category,
                           **({'period': period} if period else {})})

    def ring(self, q, hole=False):
        q = np.asarray(q, float)
        start = self.offset
        self.rings.append(self.world(q))
        self.holes.append(hole)
        self.offset += len(q)
        return np.arange(start, self.offset)

    def polygon(self, vertices, prefix='side', semantics=None, bowed=False):
        vertices = np.asarray(vertices, float)
        self.parameters[prefix+'_vertices'] = vertices
        q, spans = [], []
        count = int(self.rng.choice([16, 32, 64]))
        for i, (a, b) in enumerate(zip(vertices, np.roll(vertices, -1, axis=0))):
            run = np.linspace(a, b, count+1)[:-1]
            spans.append((len(q), count))
            q.extend(run)
        ids = self.ring(q)
        for i, (start, n) in enumerate(spans):
            ident = f'{prefix}_{i}'
            self.add_feature(ident, 'line', semantics[i] if semantics else 4, ids[start:start+n])
            a, b = vertices[i], vertices[(i+1) % len(vertices)]
            length = float(np.hypot(*(b-a)))
            angle = float((np.degrees(np.arctan2(*(b-a)[::-1]))+np.degrees(self.theta)) % 180)
            self.check(ident+'.length', 'length', [ident], length, 'Euclidean length of prescribed segment endpoints')
            self.check(ident+'.orientation', 'orientation', [ident], angle, 'construction segment angle + rigid rotation, modulo 180', period=180)
            self.check(ident+'.straightness', 'straightness', [ident], 0., 'All prescribed points lie on one exact line')
            self.check(ident+'.rms', 'rms_residual', [ident], 0., 'All prescribed points lie on one exact line')
        return ids

    def rectangle(self, w, h):
        self.parameters.update(width=w, height=h)
        ids = self.polygon([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
        self.add_feature('outline', 'polygon', 12, ids)
        self.check('outline.area', 'area', ['outline'], w*h, 'W*H', units='px^2')
        self.check('outline.perimeter', 'perimeter', ['outline'], 2*(w+h), '2*(W+H)')
        self.check('outline.centroid', 'centroid', ['outline'], self.origin, 'Rigidly transformed rectangle center')
        self.check('parallel', 'parallelism', ['side_0', 'side_2'], 0., 'Opposite prescribed sides are parallel')
        self.check('perpendicular', 'perpendicularity', ['side_0', 'side_1'], 0., 'Adjacent prescribed sides meet at 90 degrees')
        self.check('height', 'distance', ['side_0', 'side_2'], h, 'Opposite line distance equals H')
        self.check('corner', 'intersection', ['side_0', 'side_1'], self.world([w/2, -h/2]), 'Prescribed common corner')
        self.add_feature('width_pair', 'width', 0, members=['side_0', 'side_2'])
        self.check('compound.width', 'compound', ['width_pair'], h, 'Parallel side normal width is H', params={'stations': [-w/4, 0., w/4]})
        self.check('local_size', 'local_size', ['outline'], h, 'Rectangle normal cross section is H',
                   params={'direction': self.axis([1, 0]), 'origin': self.origin, 'stations': [-w/4, 0., w/4]})

    def circle(self, ident, center, radius, hole=True, lobes=0, amplitude=0.):
        n = int(self.rng.choice([64, 128, 256])) if not lobes else 64*lobes
        t = np.arange(n)*2*np.pi/n
        radial = radius+amplitude*np.cos(lobes*t) if lobes else np.full(n, radius)
        ids = self.ring(np.c_[radial*np.cos(t), radial*np.sin(t)]+center, hole)
        self.add_feature(ident, 'circle', 2 if hole else 1, ids)
        self.parameters[ident] = {'center': center, 'radius': radius, 'samples': n,
                                  'lobes': lobes, 'radial_amplitude': amplitude}
        self.check(ident+'.center', 'center', [ident], self.world(center), 'Prescribed center; balanced k-fold symmetric samples' if lobes else 'Prescribed circle center')
        self.check(ident+'.diameter', 'least_squares_diameter', [ident], 2*radius,
                   '2*R; uniform k-fold radial modulation has mean radius R' if lobes else '2*R')
        self.check(ident+'.circularity', 'circularity_ls', [ident], 2*amplitude,
                   'Sampled radial maximum minus minimum = 2*A, both extrema explicitly included')
        self.check(ident+'.rms', 'rms_residual', [ident], amplitude/np.sqrt(2) if lobes else 0.,
                   'Uniform complete periods: mean(cos^2)=1/2; residual RMS=A/sqrt(2)')
        if not lobes:
            self.check(ident+'.area', 'area', [ident], n*radius**2*np.sin(2*np.pi/n)/2,
                       'Regular inscribed n-gon area = n*R^2*sin(2*pi/n)/2; finite polyline, not pi*R^2', units='px^2')
            self.check(ident+'.perimeter', 'perimeter', [ident], 2*n*radius*np.sin(np.pi/n),
                       'Regular inscribed n-gon perimeter = 2*n*R*sin(pi/n); not 2*pi*R')
        return n

    def slot(self, ident, center, half_centers, radius):
        n = int(self.rng.choice([24, 48, 96]))
        t = np.linspace(-np.pi/2, np.pi/2, n+1)
        right = np.c_[half_centers+radius*np.cos(t), radius*np.sin(t)]
        top = np.linspace([half_centers, radius], [-half_centers, radius], n+1)[1:-1]
        left = np.c_[-half_centers-radius*np.cos(t), -radius*np.sin(t)]
        bottom = np.linspace([-half_centers, -radius], [half_centers, -radius], n+1)[1:-1]
        ids = self.ring(np.vstack([right, top, left, bottom])+center, True)
        self.add_feature(ident, 'slot', 3, ids)
        self.parameters[ident] = {'center': center, 'half_center_distance': half_centers, 'radius': radius, 'semicircle_intervals': n}
        for op, expected, formula in [('width', 2*radius, '2*r'), ('overall_length', 2*(half_centers+radius), '2*(a+r)'),
                                      ('end_radius', radius, 'Prescribed semicircle radius'),
                                      ('end_center_spacing', 2*half_centers, '2*a'), ('rms_residual', 0., 'Points lie exactly on a stadium')]:
            self.check(ident+'.'+op, op, [ident], expected, formula)
        self.check(ident+'.center', 'center', [ident], self.world(center), 'Prescribed stadium center')
        self.check(ident+'.orientation', 'orientation', [ident], np.degrees(self.theta) % 180, 'Prescribed stadium axis', period=180)

    def finish(self):
        xy = np.concatenate(self.rings)
        offsets = np.r_[0, np.cumsum([len(q) for q in self.rings])]
        return plain({'id': f'{self.family}-{self.seed}', 'family': self.family, 'seed': self.seed,
                      'generator_version': VERSION, 'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'parameters': self.parameters, 'units': 'px', 'coordinate_system': 'image',
                      'contours': {'xy': xy, 'offsets': offsets, 'closed': [True]*len(self.rings), 'units': 'px'},
                      'ring_is_hole': self.holes,
                      'request': {'schema_version': 1, 'units': 'px', 'corner_exclusion_px': 0,
                                  'features': self.features, 'measurements': self.specs},
                      'truth': self.truth})


def generate(family, seed):
    b = Builder(family, seed)
    r = b.rng
    w, h = float(r.uniform(280, 460)), float(r.uniform(180, 260))
    if family in ('rectangle_holes', 'slotted_plate', 'polar_holes', 'lobed_bore', 'ellipse_bore'):
        b.rectangle(w, h)
    if family == 'rectangle_holes':
        spacing = float(r.uniform(70, 110))
        radius = float(r.uniform(12, 25))
        for i, x in enumerate([-spacing, 0., spacing]): b.circle(f'hole_{i}', [x, 0.], radius)
        b.check('hole_pitch', 'pattern', ['hole_0', 'hole_1', 'hole_2'], spacing, 'Prescribed linear center pitch', params={'direction': b.axis([1, 0]), 'nominal_pitch': spacing})
        b.check('hole_to_line', 'distance', ['side_0', 'hole_0'], h/2, 'Circle center lies at rectangle mid-height')
        b.check('center_distance', 'center_distance', ['hole_0', 'hole_2'], 2*spacing, 'Center positions differ by 2*pitch')
    elif family == 'slotted_plate':
        for i, y in enumerate([-h/4, h/4]): b.slot(f'slot_{i}', [0., y], float(r.uniform(40, 75)), float(r.uniform(10, 20)))
        b.check('slot_spacing', 'center_distance', ['slot_0', 'slot_1'], h/2, 'Prescribed slot center spacing')
    elif family == 'washer':
        radius, bore = float(r.uniform(110, 180)), float(r.uniform(30, 65))
        eccentric = r.uniform([-8., -8.], [8., 8.])
        n0 = b.circle('outer', [0., 0.], radius, False)
        n1 = b.circle('bore', eccentric, bore)
        offset = float(np.hypot(*eccentric))
        b.check('eccentricity', 'center_offset', ['outer', 'bore'], offset, 'Norm of prescribed center offset')
        bound = radius*(1-np.cos(np.pi/n0))+bore*(1-np.cos(np.pi/n1))+1e-7
        b.check('wall', 'minimum_separation', ['outer', 'bore'], radius-bore-offset,
                'Smooth nested circles: R-r-center_offset; polygon distance error bounded by sum of chord sagittae',
                tolerance=bound, category='bounded_discretization')
    elif family == 'rounded_plate':
        radius = float(r.uniform(12, 28)); n = int(r.choice([16, 32, 64]))
        centers = [[w/2-radius, -h/2+radius], [w/2-radius, h/2-radius], [-w/2+radius, h/2-radius], [-w/2+radius, -h/2+radius]]
        starts = [[-w/2+radius, -h/2], [w/2, -h/2+radius], [w/2-radius, h/2], [-w/2, h/2-radius]]
        q, selections = [], []
        for i, (c, a) in enumerate(zip(centers, starts)):
            t = np.linspace(-np.pi/2+i*np.pi/2, i*np.pi/2, n+1)
            arc = np.c_[np.cos(t), np.sin(t)]*radius+c
            line = np.linspace(a, arc[0], n+1)[:-1]
            selections.append((f'land_{i}', 'line', 4, len(q), n)); q.extend(line)
            selections.append((f'fillet_{i}', 'arc', 6, len(q), n)); q.extend(arc[:-1])
        ids = b.ring(q)
        b.add_feature('rounded', 'rounded_rectangle', 13, ids)
        b.parameters.update(width=w, height=h, corner_radius=radius, quarter_arc_intervals=n)
        for op, value, formula in [('width', h, 'H'), ('overall_length', w, 'W'), ('corner_radius', radius, 'r'), ('rms_residual', 0., 'Exact rounded rectangle')]:
            b.check('rounded.'+op, op, ['rounded'], value, formula)
        for ident, kind, semantic, start, length in selections:
            b.add_feature(ident, kind, semantic, ids[start:start+length])
            if kind == 'arc':
                for op, value, formula in [('radius', radius, 'r'), ('angular_span', 90., 'Quarter circle'), ('arc_length', np.pi*radius/2, 'pi*r/2'),
                                           ('chord_length', np.sqrt(2)*radius, 'sqrt(2)*r'), ('sagitta', radius*(1-1/np.sqrt(2)), 'r*(1-cos(pi/4))')]:
                    b.check(ident+'.'+op, op, [ident], value, formula)
        b.circle('bore', [0., 0.], 25.)
    elif family == 'stepped_plate':
        depth, notch = float(r.uniform(25, 55)), float(r.uniform(60, 100))
        vertices = [[-w/2,-h/2],[w/2,-h/2],[w/2,h/2],[notch/2,h/2],[notch/2,h/2-depth],[-notch/2,h/2-depth],[-notch/2,h/2],[-w/2,h/2]]
        ids = b.polygon(vertices, semantics=[4, 4, 4, 7, 4, 7, 4, 4])
        b.parameters.update(width=w, height=h, notch_width=notch, step_depth=depth)
        b.add_feature('step', 'step', 0, members=['side_2', 'side_4'])
        b.check('step.depth', 'compound', ['step'], depth, 'Parallel land separation equals prescribed step depth')
        b.add_feature('opening', 'line', 4, [*range(int(ids[0]+2*(len(ids)//8)), int(ids[0]+3*(len(ids)//8))),
                                                   *range(int(ids[0]+6*(len(ids)//8)), int(ids[0]+7*(len(ids)//8)))],
                      compose_selections_from=['side_2', 'side_6'])
        # Explicit groove inspection plan: opening datum is two coplanar lands.
        b.add_feature('groove', 'groove', 0, members=['opening', 'side_3', 'side_5', 'side_4'])
        params = {'roles': {'opening': 'opening', 'left': 'side_3', 'right': 'side_5', 'bottom': 'side_4'}}
        b.check('groove.dimensions', 'compound', ['groove'], depth, 'Prescribed rectangular notch depth', params=params)
        b.check('groove.dimensions', 'compound', ['groove'], notch, 'Prescribed opening width', params=params, field='opening_width')
    elif family == 'tapered_plate':
        slope = float(r.uniform(.08, .25))
        b.parameters.update(width=w, height=h, slope=slope)
        b.polygon([[-w/2,-h/2],[w/2,-h/2],[w/2,h/2+slope*w/2],[-w/2,h/2-slope*w/2]], semantics=[4,4,9,4])
        b.add_feature('taper', 'taper', 0, members=['side_0','side_2'])
        params = {'stations': [-w/4, 0., w/4]}
        b.check('taper.width', 'compound', ['taper'], h, 'Mean width at symmetric stations is H', params=params)
        b.check('taper.width', 'compound', ['taper'], [h-slope*w/4, h, h+slope*w/4], 'Widths H+s*x at prescribed stations', params=params, field='widths')
        b.check('taper.width', 'compound', ['taper'], slope, 'd(width)/dx equals prescribed slope', params=params, field='signed_width_slope', units='px/px')
    elif family == 'polar_holes':
        count = int(r.integers(5, 10)); radius = float(min(h*.32, w*.25)); bore = 9.
        b.parameters.update(pattern_count=count, pattern_radius=radius)
        for i in range(count):
            a = 2*np.pi*i/count; b.circle(f'hole_{i}', [radius*np.cos(a), radius*np.sin(a)], bore)
        refs = [f'hole_{i}' for i in range(count)]
        params = {'domain': 'polar', 'center': b.origin, 'nominal_radius': radius}
        b.check('polar.pitch', 'pattern', refs, 360/count, '360/N for a complete equally spaced bolt pattern', params=params, units='deg')
        b.check('polar.pitch', 'pattern', refs, radius, 'Prescribed pitch-circle radius', params=params, field='mean_radius')
    elif family == 'linear_rack':
        count = int(r.integers(5, 13)); pitch = float(r.uniform(18, 28)); amplitude = float(r.uniform(4, 8))
        n = count*64+1; x = np.linspace(-count*pitch/2, count*pitch/2, n)
        q = np.c_[x, 35-amplitude*np.cos(2*np.pi*(x-x[0])/pitch)]
        closure = [[x[-1],-35], [x[0],-35]]
        ids = b.ring(np.vstack([q, closure]))
        b.add_feature('rack', 'repeated_profile', 11, ids[:n-1])
        b.parameters.update(count=count, pitch=pitch, amplitude=amplitude, sampling_intervals_per_period=64)
        params = {'direction': b.axis([1,0]), 'prominence': amplitude, 'nominal_pitch': pitch}
        # The algorithm interpolates to a uniform grid. Bound one pitch by two grid steps.
        bound = 2*count*pitch/(4*n-1)+1e-7
        b.check('rack.pitch', 'repetition', ['rack'], pitch, 'Prescribed sinusoid period; finite peak-grid error <= two resampling intervals', params=params, tolerance=bound, category='bounded_discretization')
        b.check('rack.pitch', 'repetition', ['rack'], count, 'One interior maximum per complete period; endpoints are minima', params=params, field='count', tolerance=0., units='count')
        b.check('rack.pitch', 'repetition', ['rack'], 2*amplitude, 'Sampled extrema differ by 2*A', params=params, field='profile_height')
    elif family == 'angular_teeth':
        count = int(r.integers(8, 21)); radius = float(r.uniform(110, 170)); amplitude = float(r.uniform(5, 10))
        n = count*64; t = np.arange(n)*2*np.pi/n; radial = radius+amplitude*np.cos(count*t)
        ids = b.ring(np.c_[radial*np.cos(t), radial*np.sin(t)])
        b.add_feature('teeth', 'repeated_profile', 11, ids)
        b.parameters.update(count=count, radius=radius, amplitude=amplitude, samples=n)
        params = {'domain': 'angular', 'center': b.origin, 'prominence': amplitude, 'nominal_pitch': 360/count}
        b.check('teeth.pitch', 'repetition', ['teeth'], 360/count, '360/N; includes cyclic wrap', params=params, tolerance=1e-5, units='deg')
        b.check('teeth.pitch', 'repetition', ['teeth'], count, 'Exactly N periodic maxima, seam counted once', params=params, field='count', tolerance=0., units='count')
        b.check('teeth.pitch', 'repetition', ['teeth'], 2*amplitude, 'Prescribed radial max-min', params=params, field='profile_height')
        b.circle('bore', [0.,0.], 25.)
    elif family == 'bowed_plate':
        amplitude = float(r.uniform(.1, 2.)); n = int(r.choice([33,65,129])); x = np.linspace(-w/2,w/2,n)
        z = 2*x/w; y = h/2+amplitude*(1-z*z)
        q = np.c_[x,y]; ids = b.ring(np.vstack([q, [[w/2,-h/2],[-w/2,-h/2]]]))
        b.add_feature('bow', 'line', 4, ids[:n-1])
        # Closed-form moments of symmetric, equally spaced coordinates including endpoints.
        m2 = (n+1)/(3*(n-1))
        m4 = (n+1)*(3*n*n-7)/(15*(n-1)**3)
        rms = amplitude*np.sqrt(m4-m2*m2)
        b.parameters.update(width=w,height=h,bow_amplitude=amplitude,samples=n,normalized_second_moment=m2,normalized_fourth_moment=m4)
        b.check('bow.straightness','straightness',['bow'],amplitude,'Curve is in strip [0,A] and contains triangle endpoints+apex whose minimum width is A (A<W/2)')
        b.check('bow.length','length',['bow'],w,'Symmetric bow has TLS direction along x; projected endpoint span W')
        b.check('bow.rms','rms_residual',['bow'],rms,'A*sqrt(E[z^4]-E[z^2]^2), closed-form equally spaced moments')
        b.check('bow.ptv','peak_to_valley_residual',['bow'],amplitude,'Sampled apex and endpoints differ by A')
        b.check('bow.center','center',['bow'],b.world([0.,h/2+amplitude*(1-m2)]),'TLS center is prescribed discrete mean using analytic second moment')
    elif family == 'lobed_bore':
        radius=float(r.uniform(35,65)); amplitude=float(r.uniform(.05,.5)); lobes=int(r.choice([3,4,5,6]))
        b.circle('bore',[0.,0.],radius,lobes=lobes,amplitude=amplitude)
    elif family == 'ellipse_bore':
        major,minor=float(r.uniform(45,90)),float(r.uniform(18,30)); n=int(r.choice([64,96,128]));t=np.arange(n)*2*np.pi/n
        ids=b.ring(np.c_[major*np.cos(t),minor*np.sin(t)],True)
        b.add_feature('ellipse','ellipse',13,ids)
        b.parameters.update(ellipse_semimajor=major,ellipse_semiminor=minor,ellipse_samples=n)
        for op,value,formula in [('major_axis',2*major,'2*a'),('minor_axis',2*minor,'2*b'),('rms_residual',0.,'Exact ellipse samples')]:
            b.check('ellipse.'+op,op,['ellipse'],value,formula)
        b.check('ellipse.center','center',['ellipse'],b.origin,'Prescribed ellipse center')
        b.check('ellipse.orientation','orientation',['ellipse'],np.degrees(b.theta)%180,'Prescribed major axis orientation',period=180)
    else:
        raise ValueError(family)
    return b.finish()
