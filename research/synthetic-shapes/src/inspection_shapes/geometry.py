"""Nominal primitive chains -> corresponding imperfect closed contour chains.

An edge i runs from vertex i to vertex (i+1)%N. Labels belong to edges, not
vertices. Primitive and feature identity survive all deformations.
"""
from dataclasses import dataclass, field
import numpy as np
from shapely.geometry import Polygon
from .taxonomy import SEMANTICS, PRIMITIVES, PARTS, COMPOUNDS, MEASUREMENTS

@dataclass
class Ring:
    nominal: np.ndarray
    feature: np.ndarray
    primitive: np.ndarray
    hole: bool = False
    actual: np.ndarray | None = None
    defect: np.ndarray | None = None
    anomaly_instance: np.ndarray | None = None

@dataclass
class Part:
    family: str
    features: list = field(default_factory=list)
    primitives: list = field(default_factory=list)
    rings: list = field(default_factory=list)
    relationships: list = field(default_factory=list)
    patterns: list = field(default_factory=list)
    imperfections: dict = field(default_factory=dict)
    anomalies: list = field(default_factory=list)

    def feature(self, semantic, parent=0, **params):
        fid = len(self.features) + 1
        self.features.append(dict(id=fid, semantic=semantic, semantic_id=SEMANTICS[semantic],
                                  parent=parent, geometry_type=COMPOUNDS[semantic],
                                  measurement_candidates=MEASUREMENTS.get(semantic,["profile_2d"]),
                                  nominal_parameters=params))
        return fid

class Chain:
    def __init__(self, part, hole=False, spacing=0.055):
        self.part, self.hole, self.spacing = part, hole, spacing
        self.points, self.features, self.primitives = [], [], []

    def add(self, points, fid, kind, **params):
        points = np.asarray(points, dtype=float)
        pid = len(self.part.primitives) + 1
        self.part.primitives.append(dict(id=pid, kind=kind, kind_id=PRIMITIVES[kind],
                                         feature_id=fid, nominal_parameters=params))
        if self.points and not np.allclose(self.points[-1][-1], points[0], atol=1e-7):
            raise ValueError("Disconnected primitive chain")
        self.points.append(points)
        self.features.extend([fid] * (len(points)-1))
        self.primitives.extend([pid] * (len(points)-1))

    def line(self, a, b, fid):
        n = max(1, int(np.ceil(np.linalg.norm(np.asarray(b)-a)/self.spacing)))
        self.add(np.linspace(a, b, n+1), fid, "line", start=list(a), end=list(b))

    def arc(self, c, r, a, b, fid):
        n = max(4, int(np.ceil(abs(b-a)*r/self.spacing)))
        t = np.linspace(a, b, n+1)
        self.add(np.c_[c[0]+r*np.cos(t), c[1]+r*np.sin(t)], fid, "circular_arc",
                 center=list(c), radius=r, start_angle=a, end_angle=b)

    def close(self):
        if not np.allclose(self.points[-1][-1], self.points[0][0], atol=1e-7):
            raise ValueError("Open ring")
        p = np.concatenate([x[:-1] for x in self.points])
        ring = Ring(p, np.array(self.features, np.int32), np.array(self.primitives, np.int32), self.hole)
        self.part.rings.append(ring)
        return ring

def circle(part, center, radius, semantic, hole=False, parent=0):
    fid = part.feature(semantic, parent, center=list(center), radius=float(radius))
    c = Chain(part, hole)
    c.arc(center, radius, 0, -2*np.pi if hole else 2*np.pi, fid)
    c.close()
    return fid

def slot(part, center, length, width, angle=0):
    fid = part.feature("slot", length=float(length), width=float(width), center=list(center), angle=angle)
    x, y = center
    a, r = (length-width)/2, width/2
    c = Chain(part, True)
    c.line([x-a, y+r], [x+a, y+r], fid)
    c.arc([x+a,y],r,np.pi/2,-np.pi/2,fid)
    c.line([x+a,y-r],[x-a,y-r],fid)
    c.arc([x-a,y],r,-np.pi/2,-3*np.pi/2,fid)
    ring = c.close()
    # Rotate both analytic primitive definitions and sampled curve together.
    R = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    ring.nominal = (ring.nominal-center) @ R.T + center
    for pid in np.unique(ring.primitive):
        params = part.primitives[pid-1]["nominal_parameters"]
        for key in ("start", "end", "center"):
            if key in params:
                params[key] = ((np.asarray(params[key])-center) @ R.T + center).tolist()
        for key in ("start_angle", "end_angle"):
            if key in params:
                params[key] += angle
    part.relationships.append(dict(operator="parallelism_2d", feature_refs=[fid],
                                   primitive_refs=[int(ring.primitive[0]),int(np.unique(ring.primitive)[2])]))
    return fid

def polygon(part, vertices, semantics=None, parent_semantic="external_profile"):
    parent = part.feature(parent_semantic)
    c = Chain(part)
    vertices = np.asarray(vertices)
    for i, a in enumerate(vertices):
        b = vertices[(i+1)%len(vertices)]
        fid = part.feature(semantics[i] if semantics else "flat", parent)
        c.line(a,b,fid)
    c.close()
    return parent

def plate_outline(part, w, h, r, chamfer=False):
    root = part.feature("external_profile")
    parent = part.feature("polygon" if chamfer else "rounded_profile", root, width=w, height=h)
    c = Chain(part)
    # CCW: rightward across bottom, then around right/top/left.
    corners = [([w/2-r,-h/2+r],-np.pi/2,0), ([w/2-r,h/2-r],0,np.pi/2),
               ([-w/2+r,h/2-r],np.pi/2,np.pi), ([-w/2+r,-h/2+r],np.pi,3*np.pi/2)]
    starts = [[-w/2+r,-h/2],[w/2,-h/2+r],[w/2-r,h/2],[-w/2,h/2-r]]
    for a,(center,t0,t1) in zip(starts,corners):
        b = np.asarray(center)+r*np.array([np.cos(t0),np.sin(t0)])
        e = np.asarray(center)+r*np.array([np.cos(t1),np.sin(t1)])
        c.line(a,b,part.feature("flat",parent))
        f = part.feature("chamfer" if chamfer else "fillet",parent, radius=None if chamfer else r)
        if chamfer: c.line(b,e,f)
        else: c.arc(center,r,t0,t1,f)
    ring = c.close()
    ids = np.unique(ring.primitive)
    part.relationships.extend([dict(operator="parallelism_2d", primitive_refs=[int(ids[0]),int(ids[4])]),
                               dict(operator="perpendicularity_2d",primitive_refs=[int(ids[0]),int(ids[2])])])

def repeat_profile(count, pitch, depth, style, rng):
    """Generic linear teeth with independent nominal flank template and pitch errors."""
    fractions = {"v": [0,.10,.47,.53,.90,1], "trapezoid":[0,.18,.36,.64,.82,1],
                 "square":[0,.24,.24,.76,.76,1], "sawtooth":[0,.10,.22,.45,.90,1]}
    u = np.asarray(fractions[style]); z = np.array([0,0,1,1,0,0])
    points = []
    for i in range(count):
        points.extend(np.c_[(i+u[:-1])*pitch,z[:-1]*depth])
    points.append([count*pitch,0])
    return np.asarray(points)

def make_nominal(family, rng):
    p = Part(family)
    if family in ("stepped_shaft", "threaded_shaft", "grooved_pin", "tapered_pin"):
        L, r = rng.uniform(34,70), rng.uniform(4,9)
        parent = p.feature("external_profile", length=L)
        c = Chain(p)
        # Profile path in increasing x, mirrored top/bottom with separate instances
        # for surfaces, common compound feature for threads/grooves/tapers.
        if family == "stepped_shaft":
            x1,x2 = L*rng.uniform(.2,.35), L*rng.uniform(.60,.78)
            r2,r3 = r*rng.uniform(.6,.8),r*rng.uniform(.85,1.2)
            path = np.array([[0,r*.8],[.8,r],[x1,r],[x1,r2],[x2,r2],[x2,r3],[L-.8,r3],[L,r3-.8]])
            roles=["chamfer","flat","shoulder","flat","shoulder","flat","chamfer"]
        elif family == "tapered_pin":
            path=np.array([[0,r-.6],[.6,r],[L*.25,r],[L-.6,r*.48],[L,r*.48-.4]])
            roles=["chamfer","flat","taper","chamfer"]
        elif family == "grooved_pin":
            path=[[0,r-.6],[.6,r]]; roles=["chamfer"]
            for x in np.linspace(.23*L,.77*L,int(rng.integers(1,4))):
                width,depth = rng.uniform(.8,2), rng.uniform(.5,1.7)
                style=str(rng.choice(["rectangular","v","trapezoid","dovetail"]))
                bottom={"rectangular":width,"v":0.,"trapezoid":width*.45,"dovetail":width*1.35}[style]
                path.extend([[x-width/2,r],[x-bottom/2,r-depth]])
                roles.extend(["flat","groove"])
                if bottom>0:
                    path.append([x+bottom/2,r-depth]);roles.append("groove")
                path.append([x+width/2,r]);roles.append("groove")
            path.extend([[L-.6,r],[L,r-.6]]);roles.extend(["flat","chamfer"])
            path=np.array(path)
        else:
            n=int(rng.integers(9,27)); length=L*.72; pitch=length/n
            style=str(rng.choice(["v","trapezoid","square"]))
            depth=min(r*.27,pitch*rng.uniform(.35,.62))
            teeth=repeat_profile(n,pitch,depth,style,rng)
            teeth[:,0]+=L*.20;teeth[:,1]+=r-depth
            path=np.vstack([[0,r*.65],[L*.16,r*.65],[L*.20,r-depth],teeth[1:], [L,r-depth]])
            roles=["flat","taper"]+["thread"]*(len(teeth)-1)+["flat"]
            thread = p.feature("thread",parent, pitch=pitch,count=n,depth=depth,style=style,
                               threaded_length=length,domain="linear",standard_match=None)
        path[:,0]-=L/2
        shared={}
        if family == "threaded_shaft": shared["thread"]=thread
        # Start at top right and follow top right->left, then bottom left->right.
        topids=[]; bottomids=[]
        for side,ids in ((1,topids),(-1,bottomids)):
            indices=range(len(roles)-1,-1,-1) if side==1 else range(len(roles))
            for i in indices:
                role=roles[i]
                if role in shared: fid=shared[role]
                elif role=="groove":
                    # Three contiguous surfaces share one groove instance on each side.
                    first=i
                    while first>0 and roles[first-1]=="groove": first-=1
                    key=(side, first)
                    if key not in shared: shared[key]=p.feature(role,parent)
                    fid=shared[key]
                else: fid=p.feature(role,parent)
                a,b=path[i].copy(),path[i+1].copy();a[1]*=side;b[1]*=side
                if side==1: a,b=b,a
                c.line(a,b,fid);ids.append(fid)
            if side==1:
                c.line(path[0], path[0]*[1,-1], p.feature("flat",parent))
            else:
                c.line(path[-1]*[1,-1],path[-1],p.feature("flat",parent))
        c.close()
        p.relationships.append(dict(operator="symmetry_2d",feature_refs=topids+bottomids,nominal_axis=[[0,0],[1,0]]))
    elif family in ("washer","flange"):
        R=rng.uniform(12,27)
        outer=circle(p,[0,0],R,"external_profile")
        inner=circle(p,[0,0],R*rng.uniform(.23,.49),"hole",True)
        p.relationships.append(dict(operator="center_offset_2d",feature_refs=[outer,inner]))
        if family=="flange":
            n=int(rng.integers(3,9)); pitchR=R*.72; holeR=R*rng.uniform(.065,.10)
            ids=[]
            for t in np.arange(n)*2*np.pi/n:
                ids.append(circle(p,pitchR*np.array([np.cos(t),np.sin(t)]),holeR,"hole",True))
            p.patterns.append(dict(type="polar",feature_refs=ids,center=[0,0],radius=pitchR,count=n))
    elif family in ("drilled_plate","slotted_bracket"):
        w,h=rng.uniform(38,64),rng.uniform(23,36)
        if family=="slotted_bracket" and rng.random()<.5:
            # Re-entrant notch and protruding tab; all edges retain semantic ownership.
            polygon(p,[[-w/2,-h/2],[w/2,-h/2],[w/2,h*.12],[w*.38,h*.12],
                       [w*.38,h*.32],[w/2,h*.32],[w/2,h/2],[w*.1,h/2],
                       [w*.1,h*.63],[-w*.1,h*.63],[-w*.1,h/2],[-w/2,h/2]],
                    ["flat","flat","notch","notch","notch","flat","flat","tab","tab","tab","flat","flat"])
        else: plate_outline(p,w,h,rng.uniform(.7,3),rng.random()<.5)
        if family=="drilled_plate":
            cols,rows=int(rng.integers(2,5)),int(rng.integers(1,4))
            ids=[];rad=min(w/(cols+1),h/(rows+1))*rng.uniform(.12,.24)
            for y in (np.linspace(-h*.28,h*.28,rows) if rows>1 else [0]):
                for x in np.linspace(-w*.32,w*.32,cols): ids.append(circle(p,[x,y],rad,"hole",True))
            p.patterns.append(dict(type="grid",feature_refs=ids,rows=rows,columns=cols))
        else:
            for y in [-h*.23,h*.23]:slot(p,[0,y],w*rng.uniform(.42,.66),h*rng.uniform(.12,.20),rng.uniform(-.08,.08))
            circle(p,[-w*.39,0],h*.075,"hole",True)
    elif family=="polygon_spacer":
        n=int(rng.choice([3,4,5,6,8]));R=rng.uniform(13,27)
        t=np.arange(n)*2*np.pi/n
        polygon(p,np.c_[R*np.cos(t),R*np.sin(t)],parent_semantic="polygon")
        circle(p,[0,0],R*rng.uniform(.18,.36),"hole",True)
    elif family in ("toothed_disk","cam"):
        R=rng.uniform(13,27)
        if family=="toothed_disk":
            n=int(rng.integers(10,37));depth=R*rng.uniform(.07,.17)
            from .teeth import angular_teeth
            angular_teeth(p,R-depth,n,depth,style=str(rng.choice(['linear','involute'])))
        else:
            f=p.feature("cam_profile",style="harmonic_cam")
            t=np.linspace(0,2*np.pi,2401)
            r=R*(1+rng.uniform(.10,.20)*np.cos(2*t)+rng.uniform(.04,.09)*np.sin(3*t))
            c=Chain(p);c.add(np.c_[r*np.cos(t),r*np.sin(t)],f,'freeform');c.close()
        circle(p,[0,0],R*rng.uniform(.16,.33),"hole",True)
    elif family=="rack":
        n=int(rng.integers(9,26));pitch=rng.uniform(1.5,3);depth=pitch*.65
        teeth=repeat_profile(n,pitch,depth,"trapezoid",rng);L=n*pitch;h=rng.uniform(5,10)
        teeth[:,0]-=L/2;teeth[:,1]+=h/2
        parent=p.feature("external_profile")
        f=p.feature("tooth_profile",parent,count=n,pitch=pitch,domain="linear",style="generic_rack",standard_match=None)
        c=Chain(p)
        c.line([-L/2,-h/2],[L/2,-h/2],p.feature("flat",parent))
        c.line([L/2,-h/2],teeth[-1],p.feature("flat",parent))
        for a,b in zip(teeth[:0:-1],teeth[-2::-1]): c.line(a,b,f)
        c.line(teeth[0],[-L/2,-h/2],p.feature("flat",parent));c.close()
    else: raise ValueError(family)
    return p

def valid(part, actual=True):
    arrays=[r.actual if actual else r.nominal for r in part.rings]
    if any(a is None or not np.isfinite(a).all() for a in arrays):return False
    poly=Polygon(arrays[0],arrays[1:])
    return bool(poly.is_valid and poly.area>1 and all(Polygon(x).area>.05 for x in arrays))

def imperfect(part, rng, severity=1.):
    """Smooth correlated form errors plus sparse finite-support burrs/nicks.

CCW outer / CW holes yield the right-hand normal pointing out of MATERIAL
for either ring role. Positive displacement therefore always adds material.
"""
    shear=rng.normal(0,.003)*severity
    wedge=rng.normal(0,.008)*severity
    whole_extent=float(np.ptp(np.concatenate([r.nominal for r in part.rings]),axis=0).max())
    part.imperfections={"severity":severity,"affine_shear":shear,
                        "nonparallelism_bilinear":wedge,"field_extent_mm":whole_extent,"rings":[]}
    for ri,ring in enumerate(part.rings):
        q=ring.nominal.copy(); q[:,0]+=shear*q[:,1]
        # Unlike an affine transform, this produces unequal slopes on opposite
        # straight surfaces while remaining continuous at shared junctions.
        q[:,1]+=wedge*q[:,0]*q[:,1]/whole_extent
        delta=np.roll(ring.nominal,-1,axis=0)-ring.nominal
        lengths=np.linalg.norm(delta,axis=1)
        s=np.r_[0,np.cumsum(lengths[:-1])];perim=float(lengths.sum())
        t=2*np.pi*s/perim
        scale=float(np.ptp(q,axis=0).max())
        wave=rng.uniform(.00015,.0012)*scale*severity
        rough=rng.uniform(.000015,.00012)*scale*severity
        # Low harmonics model ovality, lobing and long-wave straightness error.
        harmonics=rng.normal(0,wave,5)/np.arange(1,6)
        phases=rng.uniform(0,2*np.pi,5)
        displacement=sum(a*np.sin(k*t+ph) for a,k,ph in zip(harmonics,range(2,7),phases))
        rough_phases=rng.uniform(0,2*np.pi,12)
        displacement+=sum(rough/np.sqrt(12)*np.sin(k*t+ph) for k,ph in zip(range(23,35),rough_phases))
        tangent=np.roll(q,-1,axis=0)-np.roll(q,1,axis=0)
        normal=np.c_[tangent[:,1],-tangent[:,0]]
        normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
        # A raw normal offset is discontinuous at a sharp corner. Fade local
        # offsets there; global affine/pitch fields still move the junctions.
        unit=delta/np.maximum(lengths[:,None],1e-12)
        sharp=np.flatnonzero(np.sum(unit*np.roll(unit,1,axis=0),axis=1)<.985)
        fade=np.ones(len(q))
        if len(sharp):
            nearest=np.full(len(q),perim)
            for j in sharp:
                dist=np.abs(s-s[j]);nearest=np.minimum(nearest,np.minimum(dist,perim-dist))
            fade=np.sin(np.pi/2*np.minimum(nearest/.18,1))**2
        q+=(displacement*fade)[:,None]*normal
        # Independent smooth displacement of each primitive, zero at its endpoints.
        # Breaks parallelism locally; preserves connected boundaries without re-fitting.
        edge_errors=[]
        for pid in np.unique(ring.primitive):
            idx=np.flatnonzero(ring.primitive==pid)
            if part.primitives[pid-1]["kind"]!="line":continue
            u=np.linspace(0,1,len(idx),endpoint=False)
            amp=rng.normal(0,min(.0007*scale,.008*lengths[idx].sum()))*severity
            skew=rng.normal(0,min(.0010*scale,.010*lengths[idx].sum()))*severity
            q[idx]+=(amp*np.sin(np.pi*u)+skew*np.sin(2*np.pi*u))[:,None]*normal[idx]
            edge_errors.append([int(pid),float(amp),float(skew)])
        # Ring translation varies bore concentricity and pattern position.
        offset=rng.normal(0,.0018*scale,2)*severity if ring.hole else np.zeros(2)
        q+=offset
        # Smooth longitudinal warp introduces nonuniform tooth/thread pitch.
        pitch_amp=rng.normal(0,.0018*scale)*severity
        q[:,0]+=pitch_amp*np.sin(2*np.pi*(ring.nominal[:,0]-ring.nominal[:,0].min())/max(scale,1e-6))
        ring.defect=np.zeros(len(q),np.uint8);ring.anomaly_instance=np.zeros(len(q),np.int32)
        occupied=np.zeros(len(q),bool)
        for _ in range(int(rng.poisson(.8)) if severity>0 else 0):
            center=rng.uniform(0,perim);width=rng.uniform(.12,.55)*max(.5,severity)
            dist=np.abs(s-center);dist=np.minimum(dist,perim-dist)
            support=dist<width
            if occupied[support].any() or support.sum()<3:continue
            kind=int(rng.integers(1,3));height=rng.uniform(.03,.20)*severity
            bump=np.where(support,(1+np.cos(np.pi*np.minimum(dist/width,1)))/2,0)
            q+=(1 if kind==1 else -1)*height*(bump*fade)[:,None]*normal
            ring.defect[support]=kind;occupied|=support
            aid=len(part.anomalies)+1;ring.anomaly_instance[support]=aid
            part.anomalies.append(dict(id=aid,ring_index=ri,kind="burr" if kind==1 else "nick",
                                       center_arclength_mm=center,half_width_mm=width,requested_height_mm=height,
                                       realized_peak_mm=float(height*np.max(bump*fade))))
        ring.actual=q
        part.imperfections["rings"].append(dict(ring_index=ri,form_harmonics_mm=harmonics.tolist(),
            form_phases=phases.tolist(),roughness_amplitude_mm=rough,roughness_phases=rough_phases.tolist(),
            primitive_bow_skew_mm=edge_errors,translation_mm=offset.tolist(),pitch_warp_mm=pitch_amp))

def generate(family, seed, severity=None, max_attempts=50):
    rng=np.random.default_rng(seed)
    if severity is None: severity=float(rng.choice([0.,.4,1.,2.5],p=[.08,.27,.45,.20]))
    for attempt in range(max_attempts):
        p=make_nominal(family,rng)
        if not valid(p,False):continue
        imperfect(p,rng,severity)
        if valid(p):
            p.imperfections["rejection_attempts"]=attempt
            return p
    raise RuntimeError(f"Could not generate valid {family} after {max_attempts} attempts")
