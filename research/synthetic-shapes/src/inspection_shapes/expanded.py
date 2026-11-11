"""Expanded planar part grammars. Names describe silhouettes, not certified standards."""
import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from .geometry import Part,Chain,circle,slot,polygon,plate_outline,repeat_profile,make_nominal

NEW_FAMILIES = {
 'perforated_grid_plate':'Rectangular tooling/filter grids, up to 576 individual bores',
 'staggered_filter_plate':'Staggered perforations with hundreds of individual bores',
 'multi_bolt_circle_flange':'Several concentric bolt circles and a central bore',
 'slotted_grille':'Dense arrays of individually measurable ventilation slots',
 'mixed_aperture_panel':'Mixed circle, slot, polygon and rounded-profile openings',
 'honeycomb_screen':'Dense arrays of hexagonal through openings',
 'spoke_wheel':'Radial elongated openings and a central bore',
 'internal_spline_ring':'Internal repeated tooth contour and circular outside',
 'ratchet_wheel':'Asymmetric angular repeated tooth profile',
 'timing_pulley':'Rounded angular teeth and an optional bore',
 'serrated_washer':'Independent internal and external serration profiles',
 'lobed_coupling':'Smooth three-to-twelve-lobe external profile',
 'eccentric_cam':'Asymmetric harmonic cam with eccentric bores',
 'polygon_broach_plate':'Arrays of triangular, square, hexagonal and octagonal apertures',
 'd_profile_spacer':'D and double-D sections with optional internal openings',
 'clevis_link':'Elongated rounded links with bores and central slots',
 'dogbone_link':'Unequal lobes joined by a narrow web, with eccentric bores',
 'l_bracket':'Re-entrant L-shaped profiles with mounting bores',
 't_bracket':'Asymmetric T-shaped profiles with mounting bores',
 'c_frame':'Deep re-entrant C-shaped profiles',
 'u_channel_section':'U-shaped channel sections with mounting bores',
 'comb_plate':'Dozens of fingers and individually grouped open notches',
 'connector_strip':'Repeated tabs, notches and rows of mounting bores',
 'heat_sink_section':'Dense fins, thin lands and open channels',
 'multi_step_shaft':'Twelve to ninety-six axial stations with independent shoulders',
 'groove_stack_pin':'Twelve to eighty individual grooves per observed side',
 'multi_thread_shaft':'Two to eight separate threaded zones with intervening lands',
 'stepped_taper_arbor':'Many alternating flat lands, steps and tapered transitions',
 'sawtooth_rack':'Twenty to two hundred asymmetric repeated teeth',
 'index_plate':'Several independently clocked polar opening patterns',
 'labyrinth_plate':'Repeated serpentine through channels',
}


def closed_profile(p,vertices,semantic='rounded_profile',hole=False,**params):
    q=np.asarray(vertices,float)
    if np.allclose(q[0],q[-1]):q=q[:-1]
    area=np.sum(q[:,0]*np.roll(q[:,1],-1)-q[:,1]*np.roll(q[:,0],-1))
    if (area<0)!=hole:q=q[::-1]
    fid=p.feature(semantic,**params);c=Chain(p,hole)
    from .primitive_labels import add_polycurve
    add_polycurve(c,q,fid,arbitrary=semantic=='cam_profile');c.close();return fid


def translate(p,delta):
    for ring in p.rings:ring.nominal+=delta
    for primitive in p.primitives:
        params=primitive['nominal_parameters']
        for key in ('start','end','center'):
            if key in params:params[key]=(np.asarray(params[key])+delta).tolist()


def opening(p,center,r,kind,rng):
    if kind=='hole':return circle(p,center,r,'hole',True)
    if kind=='slot':return slot(p,center,r*3.2,r*1.3,float(rng.choice([0,np.pi/2])))
    if kind=='polygon':
        n=int(rng.choice([3,4,6,8]));t=np.arange(n)*2*np.pi/n+rng.uniform(0,2*np.pi)
        # Use densely sampled line segments, one compound ID for the aperture.
        f=p.feature('polygon',side_count=n);c=Chain(p,True);q=np.c_[r*np.cos(t),r*np.sin(t)]+center
        q=q[::-1]
        for a,b in zip(q,np.roll(q,-1,axis=0)):c.line(a,b,f)
        c.close();return f
    g=Point(0,0).buffer(r,quad_segs=32).intersection(box(-r,-r,r*.5,r))
    return closed_profile(p,np.asarray(g.exterior.coords)+center,hole=True,style='D')


def shaft(p,rng,mode,complexity):
    count=int(rng.integers(12,97)) if mode=='steps' else int(rng.integers(12,81))
    if complexity=='extreme':count=96 if mode=='steps' else 80
    if mode=='threads':count=int(rng.integers(2,9))
    if mode=='tapers':count=int(rng.integers(12,49))
    if complexity=='moderate':count=min(count,int(rng.integers(3,17)) if mode!='threads' else 3)
    r=float(rng.uniform(4,12));pitch=float(rng.uniform(1.2,3.0));q=[[0,r]];roles=[];groups=[];thread_parameters={}
    def add(x,y,role,group=None):q.append([x,y]);roles.append(role);groups.append(group)
    for i in range(count):
        x=q[-1][0];old=q[-1][1]
        if mode in ('steps','tapers'):
            new=float(rng.uniform(.45,1.15)*r)
            add(x+pitch,old,'flat');add(x+pitch+(pitch*.45 if mode=='tapers' and i%2 else 0),new,'taper' if mode=='tapers' and i%2 else 'shoulder')
        elif mode=='grooves':
            depth=rng.uniform(.12,.35)*r
            add(x+pitch*.45,r,'flat');add(x+pitch*.55,r-depth,'groove',i)
            add(x+pitch*.85,r-depth,'groove',i);add(x+pitch,r,'groove',i)
        else:
            n=int(rng.integers(6,31));tp=rng.uniform(.6,1.8);depth=min(r*.25,tp*.5)
            style=str(rng.choice(['v','square','trapezoid']))
            thread_parameters[i]={'style':style,'pitch':tp,'count':n,'depth':depth,'domain':'linear'}
            add(x+tp*2,r-depth,'taper');teeth=repeat_profile(n,tp,depth,style,rng)
            for point in teeth[1:]:add(x+tp*2+point[0],r-depth+point[1],'thread',i)
            add(q[-1][0]+tp,r,'taper');add(q[-1][0]+tp,r,'flat')
    add(q[-1][0]+pitch,q[-1][1],'flat')
    q=np.array(q);q[:,0]-=(q[0,0]+q[-1,0])/2
    root=p.feature('external_profile');c=Chain(p,spacing=.08);shared={}
    for side in (1,-1):
        for j in (range(len(roles)-1,-1,-1) if side==1 else range(len(roles))):
            role=roles[j];key=(role,groups[j],0 if role=='thread' else side)
            if role in ('thread','groove'):
                if key not in shared:shared[key]=p.feature(role,root,zone=groups[j],**(thread_parameters[groups[j]] if role=='thread' else {}))
                fid=shared[key]
            else:fid=p.feature(role,root)
            a=q[j]*[1,side];b=q[j+1]*[1,side]
            c.line(b if side==1 else a,a if side==1 else b,fid)
        c.line(q[0] if side==1 else q[-1]*[1,-1],q[0]*[1,-1] if side==1 else q[-1],p.feature('flat',root))
    c.close();p.patterns.append({'type':'linear','station_count':count,'mode':mode})
    return p


def expanded_nominal(family,rng,complexity='ordinary'):
    p=Part(family)
    if family in ('multi_step_shaft','groove_stack_pin','multi_thread_shaft','stepped_taper_arbor','stepped_shaft','grooved_pin'):
        mode={'multi_step_shaft':'steps','stepped_shaft':'steps','groove_stack_pin':'grooves','grooved_pin':'grooves','multi_thread_shaft':'threads','stepped_taper_arbor':'tapers'}[family]
        return shaft(p,rng,mode,complexity)
    panels=('perforated_grid_plate','staggered_filter_plate','mixed_aperture_panel','honeycomb_screen','polygon_broach_plate','drilled_plate')
    if family in panels:
        rows=int(rng.integers(3,25));cols=int(rng.integers(3,25))
        if complexity=='extreme':rows=cols=24
        if complexity=='moderate':rows,cols=min(rows,int(rng.integers(2,9))),min(cols,int(rng.integers(2,11)))
        if family in ('mixed_aperture_panel','polygon_broach_plate'):rows=min(rows,14);cols=min(cols,14)
        pitch=float(rng.uniform(3,6));w=(cols+1)*pitch;h=(rows+1)*pitch;plate_outline(p,w,h,pitch*.2,rng.random()<.5)
        refs=[]
        for iy in range(rows):
            for ix in range(cols):
                x=(ix-(cols-1)/2)*pitch;y=(iy-(rows-1)/2)*pitch
                if family=='staggered_filter_plate':x+=(iy%2-.5)*pitch*.35
                kind={'honeycomb_screen':'polygon','polygon_broach_plate':'polygon'}.get(family,'hole')
                if family=='mixed_aperture_panel':kind=str(rng.choice(['hole','slot','polygon','D']))
                if family=='honeycomb_screen':
                    t=np.arange(6)*np.pi/3;refs.append(closed_profile(p,np.c_[np.cos(t),np.sin(t)]*pitch*.32+[x,y],'polygon',True,side_count=6))
                else:refs.append(opening(p,[x,y],pitch*rng.uniform(.19,.28),kind,rng))
        p.patterns.append({'type':'grid','rows':rows,'columns':cols,'feature_refs':refs});return p
    if family in ('multi_bolt_circle_flange','index_plate','flange'):
        nr=int(rng.integers(2,10));nr=10 if complexity=='extreme' else nr
        if complexity=='moderate':nr=min(nr,int(rng.integers(1,4)))
        pitch=rng.uniform(3,5);R=(nr+2)*pitch;circle(p,[0,0],R,'external_profile');circle(p,[0,0],pitch*.7,'hole',True)
        for row in range(nr):
            radius=(row+2)*pitch;n=min(64,max(6,int(2*np.pi*radius/(pitch*1.1))))
            phase=rng.uniform(0,2*np.pi) if family=='index_plate' else 0.;ids=[]
            for t in np.arange(n)*2*np.pi/n+phase:ids.append(circle(p,radius*np.array([np.cos(t),np.sin(t)]),pitch*.23,'hole',True))
            p.patterns.append({'type':'polar','feature_refs':ids,'radius':radius,'count':n})
        return p
    if family in ('slotted_grille','slotted_bracket'):
        rows=int(rng.integers(5,61));cols=int(rng.integers(1,5))
        if complexity=='moderate':rows=min(rows,int(rng.integers(3,17)));cols=min(cols,2)
        pitch=rng.uniform(2,4);w=cols*pitch*5+pitch*2;h=(rows+2)*pitch
        plate_outline(p,w,h,pitch*.3);ids=[]
        for y in (np.arange(rows)-(rows-1)/2)*pitch:
            for x in (np.arange(cols)-(cols-1)/2)*pitch*5:ids.append(slot(p,[x,y],pitch*4,pitch*.45,rng.uniform(-.035,.035)))
        p.patterns.append({'type':'grid','feature_refs':ids,'rows':rows,'columns':cols});return p
    if family in ('comb_plate','connector_strip','heat_sink_section'):
        n=int(rng.integers(12,121));pitch=rng.uniform(.7,2)
        if complexity=='moderate':n=min(n,int(rng.integers(6,41)))
        h=rng.uniform(4,16);depth=h*rng.uniform(.4,.85);w=n*pitch
        root=p.feature('external_profile');c=Chain(p,spacing=.07)
        c.line([0,0],[w,0],p.feature('flat',root));c.line([w,0],[w,h],p.feature('flat',root))
        continuation=None
        for i in range(n-1,-1,-1):
            f=p.feature('notch',root);x=i*pitch
            c.line([x+pitch,h],[x+pitch*.65,h],continuation if continuation is not None else p.feature('flat',root))
            c.line([x+pitch*.65,h],[x+pitch*.65,h-depth],f);c.line([x+pitch*.65,h-depth],[x+pitch*.25,h-depth],f)
            c.line([x+pitch*.25,h-depth],[x+pitch*.25,h],f)
            continuation=p.feature('flat',root);c.line([x+pitch*.25,h],[x,h],continuation)
        c.line([0,h],[0,0],p.feature('flat',root));ring=c.close();translate(p,[-w/2,-h/2])
        if family=='connector_strip':
            for x in np.linspace(-w*.45,w*.45,min(n//2,40)):circle(p,[x,-h*.38],min(pitch*.25,h*.04),'hole',True)
        return p
    if family in ('internal_spline_ring','ratchet_wheel','timing_pulley','serrated_washer','toothed_disk','lobed_coupling','eccentric_cam','cam'):
        R=rng.uniform(12,40);n=int(rng.integers(12,193))
        if complexity=='moderate':n=min(n,int(rng.integers(10,73)))
        t=np.linspace(0,2*np.pi,max(1800,n*32),endpoint=False)
        def teeth(radius,inside=False):
            from .teeth import angular_teeth
            style='smooth' if family=='timing_pulley' else 'linear' if family=='ratchet_wheel' else str(rng.choice(['linear','involute']))
            return angular_teeth(p,radius,n,min(radius*.12,2*np.pi*radius/n*.4),inside,style)
        if family=='internal_spline_ring':circle(p,[0,0],R,'external_profile');teeth(R*.65,True)
        elif family in ('lobed_coupling','eccentric_cam','cam'):
            lobes=int(rng.integers(3,13)) if family=='lobed_coupling' else 2
            r=R*(1+.13*np.cos(lobes*t)+.06*np.sin(3*t+.7));closed_profile(p,np.c_[r*np.cos(t),r*np.sin(t)],'cam_profile')
            circle(p,[R*.15,0],R*.18,'hole',True)
        else:
            teeth(R);teeth(R*.5,True) if family=='serrated_washer' else circle(p,[0,0],R*.22,'hole',True)
        return p
    if family in ('sawtooth_rack','rack'):
        n=int(rng.integers(20,201));pitch=rng.uniform(.5,2)
        if complexity=='moderate':n=min(n,int(rng.integers(8,61)))
        teeth=repeat_profile(n,pitch,pitch*.7,'sawtooth' if family=='sawtooth_rack' else 'trapezoid',rng)
        root=p.feature('external_profile');fid=p.feature('tooth_profile',root,count=n,pitch=pitch,domain='linear');c=Chain(p,spacing=.07);L=n*pitch
        c.line([0,-4],[L,-4],p.feature('flat',root));c.line([L,-4],teeth[-1],p.feature('flat',root))
        for a,b in zip(teeth[:0:-1],teeth[-2::-1]):c.line(a,b,fid)
        c.line(teeth[0],[0,-4],p.feature('flat',root));c.close();translate(p,[-L/2,0]);return p
    if family=='spoke_wheel':
        R=rng.uniform(20,45);n=int(rng.integers(5,25));circle(p,[0,0],R,'external_profile');circle(p,[0,0],R*.15,'hole',True)
        for t in np.arange(n)*2*np.pi/n:slot(p,R*.62*np.array([np.cos(t),np.sin(t)]),R*.45,min(R*.1,R*1.5/n),t)
        return p
    if family in ('clevis_link','dogbone_link'):
        R=rng.uniform(5,12);L=R*rng.uniform(3,6)
        g=unary_union([Point(-L/2,0).buffer(R,quad_segs=48),Point(L/2,0).buffer(R*(.75 if family=='dogbone_link' else 1),quad_segs=48),box(-L/2,-R*.4,L/2,R*.4)])
        closed_profile(p,np.asarray(g.exterior.coords),'external_profile')
        for x in (-L/2,L/2):circle(p,[x,0],R*.35,'hole',True)
        slot(p,[0,0],L*.45,R*.35);return p
    if family=='d_profile_spacer':
        R=rng.uniform(12,30);g=Point(0,0).buffer(R,quad_segs=150).intersection(box(-R*(.55 if rng.random()<.5 else 1),-R,R*.6,R))
        closed_profile(p,np.asarray(g.exterior.coords),'rounded_profile',style='D_or_double_D');opening(p,[0,0],R*.25,str(rng.choice(['hole','polygon','D'])),rng);return p
    if family in ('l_bracket','t_bracket','c_frame','u_channel_section'):
        w=rng.uniform(25,70);h=rng.uniform(25,60);a=w*rng.uniform(.2,.4);b=h*rng.uniform(.2,.4)
        shapes={'l_bracket':[[0,0],[w,0],[w,b],[a,b],[a,h],[0,h]],
                't_bracket':[[0,h-b],[w*.35,h-b],[w*.35,0],[w*.65,0],[w*.65,h-b],[w,h-b],[w,h],[0,h]],
                'c_frame':[[0,0],[w,0],[w,b],[a,b],[a,h-b],[w,h-b],[w,h],[0,h]],
                'u_channel_section':[[0,0],[w,0],[w,h],[w-a,h],[w-a,b],[a,b],[a,h],[0,h]]}
        polygon(p,np.array(shapes[family])-[w/2,h/2]);outer=Polygon(p.rings[0].nominal)
        for _ in range(20):
            point=[rng.uniform(-w/2,w/2),rng.uniform(-h/2,h/2)];r=min(a,b)*.12;probe=Point(point).buffer(r*1.8)
            if outer.contains(probe) and not any(probe.intersects(Polygon(ring.nominal)) for ring in p.rings[1:]):circle(p,point,r,'hole',True)
        return p
    if family=='labyrinth_plate':
        from shapely.geometry import LineString
        n=int(rng.integers(3,14));pitch=4.;w=50.;h=(n+2)*pitch;plate_outline(p,w,h,1)
        for i in range(n):
            y=(i-(n-1)/2)*pitch;g=LineString([[-20,y],[-10,y+.8],[0,y-.8],[10,y+.8],[20,y]]).buffer(.5,quad_segs=12)
            closed_profile(p,np.asarray(g.exterior.coords),'rounded_profile',True,style='serpentine_channel')
        return p
    return make_nominal(family,rng)
