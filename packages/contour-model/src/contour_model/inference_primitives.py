"""Primitive selection and secondary compounds, using observed coordinates only."""
import hashlib
import numpy as np
from scipy.signal import find_peaks
from .sampling import prepare_adaptive
from .postprocess import (mapped_probabilities, mapped_boundary_events, runs,
                          describe_selection, observed_turn)

BASE_NAMES_V1=['unlabeled','line','arc','circle','arbitrary_profile']
BASE_NAMES=['unlabeled','line','arc','circle']
COMPOUND_NAMES=['none','full_round_slot','triangular_thread','trapezoidal_thread','tooth_profile']
LABEL_VERSION='primitives-compounds-v2'


def base_names_for_schema(schema):
    if schema=='primitives-compounds-v1':return BASE_NAMES_V1
    if schema=='primitives-compounds-v2':return BASE_NAMES
    raise ValueError(f'Unsupported primitive schema: {schema!r}')


def compatible_thread_sides(first,second,features,raw,edge_weights):
    """Opposite observed sides of one zone share a longitudinal interval.

    This only constrains grouping; it never assigns the compound's class.
    """
    supports=[]
    for members in (first,second):
        ids=np.concatenate([features[i]['raw_edge_indices'] for i in members])
        if len(ids)<4:return False
        q=raw[ids];w=edge_weights[ids];center=np.average(q,axis=0,weights=w)
        _,vectors=np.linalg.eigh(((q-center)*w[:,None]).T@(q-center)/w.sum())
        supports.append((q,center,vectors[:,-1]))
    (a,ca,axis),(b,cb,other)=supports
    if abs(np.dot(axis,other))<.96:return False
    ua,ub=a@axis,b@axis
    overlap=min(ua.max(),ub.max())-max(ua.min(),ub.min())
    # A side may be recognized only in part; it must lie within the other side's
    # span. extend_thread_side then recovers the rest of that side.
    if overlap<.75*min(np.ptp(ua),np.ptp(ub)):return False
    # Two neighboring subdivisions on the same side are not opposite faces.
    normal=np.array([-axis[1],axis[0]])
    separation=abs(np.dot(ca-cb,normal))
    return separation>max(np.ptp(a@normal),np.ptp(b@normal))


def extend_thread_side(side,opposite,features,raw,owner,offsets,taken):
    """Extend one paired thread side along its ring over the opposite side's span.

    The per-edge compound posterior can fade at a zone's ends, leaving one side
    partly recognized. Contiguous line/arc/profile instances are added while they
    stay within the zone's axial span and the side's radial band.
    """
    # Center each side separately: the thread's width must not rotate its axis.
    q=np.concatenate([p-p.mean(0) for p in (raw[np.concatenate([features[i]['raw_edge_indices'] for i in members])]
                                            for members in (side,opposite))])
    axis=np.linalg.eigh(q.T@q)[1][:,-1];normal=np.array([-axis[1],axis[0]])
    span=[raw[np.concatenate([features[i]['raw_edge_indices'] for i in members])]@axis for members in (side,opposite)]
    side_points=raw[np.concatenate([features[i]['raw_edge_indices'] for i in side])]
    # Endpoints on the zone boundary must survive axis-fit roundoff.
    slack=2*float(np.median(np.linalg.norm(np.diff(side_points,axis=0),axis=1)))
    low,high=min(v.min() for v in span)-slack,max(v.max() for v in span)+slack
    band=side_points@normal
    margin=.25*np.ptp(band)
    ring=features[side[0]]['contour_spans'][0]['ring_index'];lo,hi=offsets[ring:ring+2];n=hi-lo
    extended=list(side);inside=np.isin(owner[lo:hi],side)
    for step in (1,-1):
        # Start from the run's last edge in the walking direction.
        edge=int(np.flatnonzero(inside&~np.roll(inside,-step))[0])
        for _ in range(n):
            edge=(edge+step)%n;index=owner[lo+edge]
            if index<0:continue
            if index in extended:break
            f=features[index];ids=np.asarray(f['raw_edge_indices']);p=raw[ids]
            if (index in taken or f['semantic_id'] not in (1,2,4) or (p@axis).min()<low or (p@axis).max()>high or
                    (p@normal).min()<band.min()-margin or (p@normal).max()>band.max()+margin):
                break
            extended.append(index)
            positions=ids-lo
            edge=int(positions[np.argmax((positions-edge)*step%n)])
    return extended


def softmax(values):
    values=np.exp(values-values.max(-1,keepdims=True))
    return values/values.sum(-1,keepdims=True)


def sigmoid(values):
    return 1/(1+np.exp(-values))


def model_outputs(model,prepared,instances):
    """Token logits as float32 arrays from a torch module or a torch-free runtime."""
    if hasattr(model,'run_prepared'):return model.run_prepared(prepared,instances=instances)
    import torch
    device=next(model.parameters()).device
    with torch.no_grad():
        out=model(torch.as_tensor(prepared.x,device=device),prepared.offsets,instances=instances)
    return {key:value.float().cpu().numpy() for key,value in out.items() if torch.is_tensor(value)}


# merge_tol=1: autoresearch P005 (edge-model bench F1 .8638 -> .8966); see RESEARCH.md.
DEFAULT_POST=dict(peak_height=.5,peak_distance=2,cut_merge=0.,min_piece=0.,min_run=0.,merge_tol=1.,relabel_tol=0.)


def primitive_residual(points,kind):
    """Maximum orthogonal residual of a total-least-squares line or algebraic circle fit."""
    if len(points)<3:return 0.
    c=points-points.mean(0)
    if kind==1:
        normal=np.linalg.eigh(c.T@c)[1][:,0]
        return float(np.abs(c@normal).max())
    scale=max(float(np.abs(c).max()),1e-12);u=c/scale
    A=np.c_[u,np.ones(len(u))];b=(u**2).sum(1)
    sol,*_=np.linalg.lstsq(A,b,rcond=None);center=sol[:2]/2;radius=np.sqrt(max(sol[2]+center@center,0.))
    return float(np.abs(np.linalg.norm(u-center,axis=1)-radius).max()*scale)


def refine_ring_pieces(pieces,q,lo,post,spacing):
    """Fit-aware cleanup of one ring's (ids, class) pieces, in cyclic order.

    Adjacent same-class pieces merge when one primitive fits their union within
    ``merge_tol`` point spacings (collinear lines, cocircular arcs). Optionally
    an arc that a line fits within ``relabel_tol`` spacings becomes a line.
    """
    n=len(q);pieces=sorted(pieces,key=lambda p:p[0][0])
    def points(ids):
        local=np.asarray(ids)-lo;return q[np.r_[local,(local[-1]+1)%n]]
    if post['relabel_tol']>0:
        pieces=[(ids,1 if cls==2 and primitive_residual(points(ids),1)<=post['relabel_tol']*spacing else cls) for ids,cls in pieces]
    tol=post['merge_tol']*spacing
    if tol<=0 or len(pieces)<2:return pieces
    # Greedy best-first merging; only pairs touching a merged piece are refit.
    cache={}
    def pair(i):
        j=(i+1)%len(pieces);(a,ca),(b,cb)=pieces[i],pieces[j]
        if (len(pieces)==2 and i==1) or ca!=cb or ca not in (1,2) or (a[-1]-lo+1)%n!=b[0]-lo:return None
        key=(a[0],len(a),b[0],len(b))
        if key not in cache:cache[key]=primitive_residual(points(np.r_[a,b]),ca)
        return cache[key]
    while len(pieces)>1:
        scores=[(r,i) for i in range(len(pieces)) if (r:=pair(i)) is not None and r<=tol]
        if not scores:break
        _,i=min(scores);j=(i+1)%len(pieces)
        merged=(np.r_[pieces[i][0],pieces[j][0]],pieces[i][1])
        if j==0:pieces=[merged]+pieces[1:i]
        else:pieces=pieces[:i]+[merged]+pieces[j+1:]
    return pieces


def smooth_ring_classes(labels,lengths,probabilities,minimum):
    """Relabel class runs shorter than ``minimum`` arc length to a neighboring class.

    Shortest runs are absorbed first; the neighbor with the higher mean posterior
    over the short run wins. Complete single-class rings are unchanged.
    """
    labels=labels.copy()
    while minimum>0:
        starts=np.flatnonzero(labels!=np.roll(labels,1))
        if len(starts)<2:break
        ends=np.r_[starts[1:],starts[0]+len(labels)]
        size=np.array([lengths[np.arange(a,b)%len(labels)].sum() for a,b in zip(starts,ends)])
        k=int(size.argmin())
        if size[k]>=minimum:break
        run=np.arange(starts[k],ends[k])%len(labels)
        left,right=labels[(starts[k]-1)%len(labels)],labels[ends[k]%len(labels)]
        choice=left if probabilities[run,left].mean()>=probabilities[run,right].mean() else right
        labels[run]=choice
    return labels


def predict(model,contours,tokens=None,threshold=.7,instances=True,units='mm',post=None):
    if units not in ('mm','px'):raise ValueError('Units must be mm or px')
    base_names=base_names_for_schema(model.config.label_schema)
    post=dict(DEFAULT_POST,**(post or {}))
    prepared=prepare_adaptive(contours,version=model.config.sampling_version)
    out=model_outputs(model,prepared,instances)
    primary=mapped_probabilities(prepared,softmax(out['primary']))
    compound=mapped_probabilities(prepared,softmax(out['compound']))
    semantic=mapped_probabilities(prepared,sigmoid(out['semantic']))
    defect=mapped_probabilities(prepared,softmax(out['defect']))
    token_boundary=sigmoid(out['boundary'])
    boundary=mapped_boundary_events(prepared,token_boundary)
    raw=np.asarray(contours['actual_xy_mm'],np.float64);offsets=np.asarray(contours['ring_offsets'],np.int64)
    classes=primary.argmax(-1);features=[];owner=np.full(len(raw),-1,int)
    edge_weights=np.empty(len(raw))

    def add_base(ids,cls):
        if not len(ids):return
        score=float(np.average(primary[ids,cls],weights=edge_weights[ids]))
        routes={1:'line',2:'arc',3:'circle',4:'profile'}
        f={'id':f'base_{len(features)+1:05d}','semantic_id':int(cls),'semantic':base_names[cls],
           'label_level':'base','confidence':score,'status':'accepted' if score>=threshold else 'abstain',
           'confidence_calibrated':False,'measurement_route':routes[int(cls)],
           'raw_edge_indices':ids.tolist(),'contour_spans':describe_selection(ids,contours),
           'defect_edge_indices':ids[defect[ids,1:].max(-1)>.5].tolist(),
           'parent':'part','units':units,'fit_geometry':'original_observations',
           'grouping':'primitive_contour_span','boundary_search_regions':[]}
        owner[ids]=len(features);features.append(f)

    for ri,(lo,hi) in enumerate(zip(offsets[:-1],offsets[1:])):
        q=raw[lo:hi];lengths=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1);edge_weights[lo:hi]=lengths
        arc=np.r_[0,np.cumsum(lengths)]
        avg=np.average(primary[lo:hi],axis=0,weights=lengths)
        if avg.argmax()==3 and np.mean(classes[lo:hi]==3)>.8:
            classes[lo:hi]=3;add_base(np.arange(lo,hi),3);continue
        # A partial circular span is an Arc, never a partial Circle instance.
        classes[lo:hi][classes[lo:hi]==3]=2
        tlo,thi=prepared.offsets[ri:ri+2]
        spacing=arc[-1]/(thi-tlo);window=2*spacing
        if post['min_run']>0:
            ring_probs=primary[lo:hi].copy();ring_probs[:,2]+=ring_probs[:,3]
            classes[lo:hi]=smooth_ring_classes(classes[lo:hi],lengths,ring_probs,post['min_run']*spacing)
        turn=observed_turn(q,arc,np.arange(len(q)),window)
        # Learned events select neighborhoods; observed turning refines sharp corners.
        b=token_boundary[tlo:thi];peaks=find_peaks(np.tile(b,3),height=post['peak_height'],distance=post['peak_distance'])[0]
        peaks=peaks[(peaks>=len(b))&(peaks<2*len(b))]-len(b)
        cuts=[];strength=[]
        for peak in peaks:
            support=prepared.overlap.getrow(tlo+int(peak)).indices
            support=support[(support>=lo)&(support<hi)]
            if not len(support):continue
            local=support-lo
            anchor=int(local[np.argmax(turn[local])]) if turn[local].max()>.12 else int(local[len(local)//2])
            cuts.append(anchor);strength.append(float(b[peak]))
        if post['cut_merge']>0 and len(cuts)>1:
            # One corner may excite neighboring boundary tokens; keep the strongest event.
            order=np.argsort(strength)[::-1];kept=[]
            for i in order:
                d=np.abs(arc[cuts[i]]-arc[[cuts[j] for j in kept]]) if kept else np.array([])
                if not len(d) or np.minimum(d,arc[-1]-d).min()>post['cut_merge']*spacing:kept.append(i)
            cuts=[cuts[i] for i in sorted(kept)]
        min_piece=post['min_piece']*spacing;pieces=[]
        for cls in np.unique(classes[lo:hi]):
            if cls==0:continue
            for run in runs(classes[lo:hi]==cls):
                positions={int(v):i for i,v in enumerate(run)}
                cut=sorted({positions[c] for c in cuts if c in positions})
                if min_piece>0 and cut:
                    # Drop cuts that would leave a piece shorter than min_piece.
                    cum=np.r_[0,np.cumsum(lengths[run])];closed=len(run)==hi-lo;kept=[]
                    for v in cut:
                        prev=kept[-1] if kept else (None if closed else 0)
                        if prev is not None and cum[v]-cum[prev]<min_piece:continue
                        if not closed and cum[-1]-cum[v]<min_piece:continue
                        kept.append(v)
                    if closed and len(kept)>1 and cum[-1]-cum[kept[-1]]+cum[kept[0]]<min_piece:kept.pop()
                    cut=kept
                if len(run)==hi-lo and cut:
                    for a,b in zip(cut,cut[1:]+cut[:1]):
                        chosen=run[np.arange(a,b if b>a else b+len(run))%len(run)]
                        pieces.append((lo+chosen,int(cls)))
                else:
                    cut=sorted({0,len(run),*[v for v in cut if 2<=v<len(run)-2]})
                    for a,b in zip(cut[:-1],cut[1:]):pieces.append((lo+run[a:b],int(cls)))
        if post['merge_tol']>0 or post['relabel_tol']>0:
            pieces=refine_ring_pieces(pieces,q,lo,post,float(np.median(lengths)))
        for ids,cls in pieces:add_base(ids,cls)
    relations=[]
    for ri,(lo,hi) in enumerate(zip(offsets[:-1],offsets[1:])):
        o=owner[lo:hi]
        for i in np.flatnonzero(o!=np.roll(o,1)):
            left,right=o[(i-1)%len(o)],o[i]
            if min(left,right)<0:continue
            relations.append({'type':'contour_adjacency','feature_a':features[left]['id'],
                              'feature_b':features[right]['id'],'ring_index':ri,'at_edge':int(i)})
        perimeter=edge_weights[lo:hi].sum();radius=2*perimeter/(prepared.offsets[ri+1]-prepared.offsets[ri])
        arc=np.r_[0,np.cumsum(edge_weights[lo:hi])]
        for f in features:
            for span in f['contour_spans']:
                if span['ring_index']!=ri or span['closed_ring']:continue
                for at in (span['start_edge'],(span['start_edge']+span['edge_count'])%(hi-lo)):
                    distance=np.abs(arc[:-1]-arc[at]);distance=np.minimum(distance,perimeter-distance)
                    f['boundary_search_regions'].append({'ring_index':ri,'anchor_edge':int(at),
                        'search_radius':float(radius),'raw_edge_indices':(lo+np.flatnonzero(distance<=radius)).tolist()})
    compounds=[];assigned=set()
    def add_compound(members,cls,score,grouping):
        if not members:return
        ids=np.unique(np.concatenate([features[i]['raw_edge_indices'] for i in members])).astype(int)
        f={'id':f'compound_{len(compounds)+1:04d}','semantic_id':int(cls),'semantic':COMPOUND_NAMES[cls],
           'label_level':'compound','base_instance_ids':[features[i]['id'] for i in members],
           'confidence':float(score),'status':'accepted','confidence_calibrated':False,
           'classification_policy':'most_likely_compound','raw_edge_indices':ids.tolist(),
           'contour_spans':describe_selection(ids,contours),'parent':'part','units':units,
           'measurement_route':{1:'slot',2:'thread_pitch',3:'thread_pitch',4:'repeated_profile'}[int(cls)],
           'fit_geometry':'original_observations','grouping':grouping,
           'boundary_search_regions':[s for i in members for s in features[i]['boundary_search_regions']]}
        compounds.append(f);assigned.update(members)
        for i in members:relations.append({'type':'compound_membership','feature_a':features[i]['id'],'feature_b':f['id']})
    if instances:
        # Each opening supplies its own candidate; there is no slot count ceiling.
        for ri,(lo,hi) in enumerate(zip(offsets[:-1],offsets[1:])):
            if not prepared.hole[ri]:continue
            probs=np.average(compound[lo:hi],axis=0,weights=edge_weights[lo:hi])
            members=sorted(set(owner[lo:hi])-{ -1 })
            if probs.argmax()==1 and members and all(features[i]['semantic_id'] in (1,2) for i in members):
                add_compound(members,1,probs[1],'closed_slot_loop')
        query=softmax(out['query_class']);qclasses=query.argmax(-1)
        candidates=[i for i in np.argsort(-query[:,2:].max(1)) if qclasses[i]>=2]
        masks=mapped_probabilities(prepared,sigmoid(out['query_masks'][candidates]).T).T if candidates else []
        # Classify complete base instances, then associate complete contiguous
        # repeat regions. Query-mask erosion must not fragment a threaded zone.
        repeat_members=set()
        for i,f in enumerate(features):
            if i in assigned or f['semantic_id']==3:continue
            ids=f['raw_edge_indices'];cp=np.average(compound[ids],axis=0,weights=edge_weights[ids])
            if cp[2:].sum()>cp[:2].sum():repeat_members.add(i)
        components=[]
        for lo,hi in zip(offsets[:-1],offsets[1:]):
            for run in runs(np.isin(owner[lo:hi],list(repeat_members))):
                members=sorted(set(owner[lo+run]))
                components.append(members)
        info=[];supported=[]
        for members in components:
            ids=np.concatenate([features[i]['raw_edge_indices'] for i in members])
            cp=np.average(compound[ids],axis=0,weights=edge_weights[ids])
            cls=int(cp[2:].argmax()+2)
            # Repeated thread flanks need at least two pairs of constituent
            # spans on a side. Isolated corners are not repeat candidates,
            # even when their compound posterior is confident. This is a
            # detection constraint; detected compounds never abstain.
            if cls in (2,3) and sum(features[i]['semantic_id'] in (1,2) for i in members)<4:continue
            supported.append(members)
            info.append((cls,features[members[0]]['contour_spans'][0]['ring_index'],ids))
        components=supported
        for i,members in enumerate(components):
            if any(v in assigned for v in members):continue
            cls,ring,ids=info[i];partners=[]
            for j,other in enumerate(components):
                if j<=i or any(v in assigned for v in other):continue
                other_cls,other_ring,other_ids=info[j]
                if ring!=other_ring or (cls==4)!=(other_cls==4):continue
                if cls==4:
                    # Query evidence may bridge missed semantic intervals of a
                    # tooth profile, but never merges separate opening rings.
                    if any(np.average(mask[ids],weights=edge_weights[ids])>.5 and
                           np.average(mask[other_ids],weights=edge_weights[other_ids])>.5 for mask in masks):
                        partners.append(j)
                elif compatible_thread_sides(members,other,features,raw,edge_weights):partners.append(j)
            # A thread zone has at most one opposite contour-side partner.
            if cls!=4 and len(partners)>1:
                partners=sorted(partners,key=lambda j:np.linalg.norm(raw[ids].mean(0)-raw[info[j][2]].mean(0)))[:1]
            if cls!=4 and partners:
                other=components[partners[0]];taken=assigned|set(members)|set(other)
                members=extend_thread_side(members,other,features,raw,owner,offsets,taken-set(members))
                other=extend_thread_side(other,members,features,raw,owner,offsets,(taken|set(members))-set(other))
                combined=members+other
            else:
                combined=members+[v for j in partners for v in components[j]]
            ids=np.concatenate([features[v]['raw_edge_indices'] for v in combined])
            probs=np.average(compound[ids],axis=0,weights=edge_weights[ids]);cls=int(probs[2:].argmax()+2)
            add_compound(combined,cls,probs[cls],'observed_repeat_regions_with_association')
    result={'schema_version':2,'label_version':model.config.label_schema,'units':units,
            'coordinate_hash':hashlib.sha256(raw.tobytes()+offsets.tobytes()).hexdigest(),
            'part':{'id':'part','ring_count':len(offsets)-1,'ring_is_hole':prepared.hole.tolist(),
                    'bounds':[raw.min(0).tolist(),raw.max(0).tolist()]},
            'features':features,'compound_features':compounds,'relations':relations,
            'base_names':base_names,'compound_names':COMPOUND_NAMES,'threshold':threshold,
            'confidence_calibrated':False,'compound_grouping_trained':instances,
            'measurement_policy':'Measure original observed edges; compounds reference retained base instances',
            'unselected_edge_fraction':float(np.mean(primary.max(1)<threshold))}
    return result,dict(primary=primary,semantic=semantic,compound=compound,defect=defect,boundary=boundary,classes=classes)
