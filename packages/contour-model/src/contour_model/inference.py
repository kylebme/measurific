"""Inference never reads synthetic feature labels, primitives, nominal CAD or family."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from .data import prepare,NAMES,LABEL_VERSION
from .train import load_model,model_device
from .postprocess import (runs,describe_selection,mapped_probabilities,mapped_boundary_events,
                          split_surface_run,observed_turn,surface_run_turn)

ROUTES={2:'circle',3:'slot',4:'line',5:'line',6:'arc',7:'line',8:'groove',9:'line',10:'thread_pitch',11:'repeated_profile',12:'polygon',13:'rounded_profile',14:'profile',15:'tab',16:'notch',1:'external_profile'}


@torch.no_grad()
def predict(model,contours,tokens=1024,threshold=.7,instances=False,units='mm'):
    if model.config.label_schema in ('primitives-compounds-v1','primitives-compounds-v2'):
        from .inference_primitives import predict as primitive_predict
        return primitive_predict(model,contours,tokens,threshold,instances,units)
    if units not in ('mm','px'):raise ValueError('Units must be mm or px')
    if model.config.adaptive_sampling:
        from .sampling import prepare_adaptive
        prepared=prepare_adaptive(contours,version=model.config.sampling_version)
    else:prepared=prepare(contours,tokens)
    device=next(model.parameters()).device
    output=model(torch.as_tensor(prepared.x,device=device),prepared.offsets,instances=instances)
    primary=mapped_probabilities(prepared,output['primary'].softmax(-1).cpu().numpy())
    semantic=mapped_probabilities(prepared,output['semantic'].sigmoid().cpu().numpy())
    defects=mapped_probabilities(prepared,output['defect'].softmax(-1).cpu().numpy())
    boundary_values=output['boundary'].sigmoid().cpu().numpy()
    boundary=(mapped_boundary_events(prepared,boundary_values) if model.config.sampling_version=='adaptive-local-v2' and model.config.adaptive_sampling
              else mapped_probabilities(prepared,boundary_values[:,None])[:,0])
    classes=primary.argmax(-1);confidence=primary.max(-1);features=[]
    def add(indices,cls,score,grouping):
        if not len(indices) or cls==0:return
        defect_flags=defects[indices,1:].max(-1)>.5
        # Keep all raw points; the measurement module decides whether robust fitting is appropriate.
        f={'id':f'feature_{len(features)+1:04d}','semantic_id':int(cls),'semantic':NAMES[int(cls)],
           'confidence':float(score),'status':'accepted' if score>=threshold else 'abstain',
           'confidence_calibrated':False,'grouping':grouping,'measurement_route':ROUTES.get(int(cls),'profile'),
           'raw_edge_indices':[int(x) for x in indices], 'contour_spans':describe_selection(indices,contours),
           'defect_edge_indices':[int(i) for i in indices[defect_flags]],'parent':'part',
           'fit_geometry':'original_observations','units':units}
        features.append(f)
    # Geometry proposes closed-loop support; learned semantics decide hole vs slot.
    components=[]
    for ri,(lo,hi) in enumerate(zip(contours['ring_offsets'][:-1],contours['ring_offsets'][1:])):
        loop_class=int(np.argmax(primary[lo:hi].mean(0)))
        if prepared.hole[ri] and loop_class in (2,3,11,12,13) and np.mean(classes[lo:hi]==loop_class)>.9:
            indices=np.arange(lo,hi)
            add(indices,loop_class,float(np.quantile(primary[indices,loop_class],.2)),'closed_ring_consensus')
            continue
        q=np.asarray(contours['actual_xy_mm'])[lo:hi]
        seg=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1);arc=np.r_[0,np.cumsum(seg)]
        window=2*arc[-1]/(prepared.offsets[ri+1]-prepared.offsets[ri])
        turn=observed_turn(q,arc,np.arange(len(q)),window)
        for c in np.unique(classes[lo:hi]):
            if c==0:continue
            for run in runs(classes[lo:hi]==c):
                indices=lo+run;peaks=[]
                # Repeated profiles, slots and grooves must not split at their internal corners.
                if c in (4,5,7,9):
                    local_turn=surface_run_turn(q,arc,run,window) if model.config.adaptive_sampling else turn[run]
                    eligible=(boundary[indices]>.65)&(local_turn>.25)&(defects[indices,1:].max(-1)<.5)
                    peaks=np.flatnonzero(eligible & (local_turn>=np.roll(local_turn,1)) & (local_turn>np.roll(local_turn,-1)))
                for selected in split_surface_run(indices,peaks,hi-lo):
                    if len(selected)>=(1 if model.config.adaptive_sampling else 3):
                        components.append((int(c),selected,float(np.quantile(confidence[selected],.2))))
    assigned=set()
    if instances:
        qc=output['query_class'].softmax(-1).cpu().numpy();qclass=qc.argmax(-1)
        candidates=[int(i) for i in np.argsort(-qc.max(-1)) if int(qclass[i]) in (10,11) and qc[i].max()>=threshold]
        masks=mapped_probabilities(prepared,output['query_masks'][candidates].sigmoid().cpu().numpy().T).T if candidates else []
        for mask,qi in zip(masks,candidates):
            cls=int(qclass[qi]);score=float(qc[qi,cls])
            # Learned queries associate complete, already selected semantic runs.
            # They never replace the original edge support with an eroded query mask.
            if cls not in (10,11) or score<threshold:continue
            matches=[j for j,(c,ids,conf) in enumerate(components) if j not in assigned and c==cls
                     and np.mean(mask[ids]>.5)>.6 and conf>=threshold]
            if len(matches)>=2:
                indices=np.concatenate([components[j][1] for j in matches]);assigned.update(matches)
                add(indices,cls,min(score,min(components[j][2] for j in matches)),'learned_compound_association')
    for j,(c,indices,score) in enumerate(components):
        if j not in assigned:add(indices,c,score,'observed_contiguous_span')
    # Ancestor classes have their own overlapping support, independent of leaf IDs.
    for ri,(lo,hi) in enumerate(zip(contours['ring_offsets'][:-1],contours['ring_offsets'][1:])):
        if prepared.hole[ri]:continue
        for cls in (12,13):
            score=float(np.quantile(semantic[lo:hi,cls-1],.2))
            if score<threshold:continue
            add(np.arange(lo,hi),cls,score,'semantic_parent')
            parent=features[-1]['id']
            for feature in features[:-1]:
                if all(lo<=i<hi for i in feature['raw_edge_indices']):feature['parent']=parent
    raw=np.asarray(contours['actual_xy_mm'],np.float64);offsets=np.asarray(contours['ring_offsets'],np.int64)
    owner=np.full(len(raw),-1,int)
    for fi,feature in enumerate(features):
        if feature['grouping']=='semantic_parent':continue
        owner[feature['raw_edge_indices']]=fi
        searches=[]
        for span in feature['contour_spans']:
            if span['closed_ring']:continue
            ri=span['ring_index'];lo,hi=offsets[ri:ri+2];q=raw[lo:hi]
            arc=np.r_[0,np.cumsum(np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1))]
            radius=2*arc[-1]/(prepared.offsets[ri+1]-prepared.offsets[ri])
            for endpoint in (span['start_edge'],(span['start_edge']+span['edge_count'])%(hi-lo)):
                distance=np.abs(arc[:-1]-arc[endpoint]);distance=np.minimum(distance,arc[-1]-distance)
                searches.append({'ring_index':int(ri),'anchor_edge':int(endpoint),'search_radius':float(radius),
                                 'raw_edge_indices':(lo+np.flatnonzero(distance<=radius)).tolist()})
        feature['boundary_search_regions']=searches
    relations=[]
    for ri,(lo,hi) in enumerate(zip(offsets[:-1],offsets[1:])):
        ring_owner=owner[lo:hi]
        for at in np.flatnonzero(ring_owner!=np.roll(ring_owner,1)):
            left=ring_owner[(at-1)%(hi-lo)];right=ring_owner[at]
            if left>=0 and right>=0:
                relations.append({'type':'contour_adjacency','feature_a':features[left]['id'],
                                  'feature_b':features[right]['id'],'ring_index':ri,'at_edge':int(at)})
    source_hash=hashlib.sha256(raw.tobytes()+offsets.tobytes()).hexdigest()
    result={'schema_version':1,'label_version':LABEL_VERSION,'units':units,'coordinate_hash':source_hash,
            'part':{'id':'part','ring_count':len(offsets)-1,'ring_is_hole':prepared.hole.tolist(),
                    'bounds':[raw.min(0).tolist(),raw.max(0).tolist()]},
            'features':features,'relations':relations,'threshold':threshold,'confidence_calibrated':False,
            'measurement_policy':'Use raw selected edges; no nominal geometry or model-regressed dimensions',
            'compound_grouping_trained':instances,'grouping_policy':'closed-ring consensus; geometry-supported surface boundaries; learned repeat associations','unselected_edge_fraction':float((confidence<threshold).mean())}
    return result,{'primary':primary,'semantic':semantic,'defect':defects,'boundary':boundary,'classes':classes}


def predict_file(checkpoint,contours,output,device='cpu',tokens=None,threshold=.7,units='mm',measure=False):
    torch.set_num_threads(4)
    dev=model_device(device);model,ckpt=load_model(checkpoint,dev)
    if model.config.adaptive_sampling and tokens is not None:
        raise ValueError('Adaptive checkpoint selects per-ring resolution from geometry; omit --tokens')
    required=ckpt.get('input_contract',{}).get('required_token_budget')
    if required is not None and tokens is not None and tokens!=required:
        raise ValueError(f'Exported checkpoint requires --tokens {required}; changing sampling scale requires retraining/validation')
    with np.load(contours,allow_pickle=False) as a:
        # Strict input allowlist. Extra ground-truth arrays cannot influence predictions.
        raw={'actual_xy_mm':a['actual_xy_mm'].copy(),'ring_offsets':a['ring_offsets'].copy()}
    result,arrays=predict(model,raw,tokens or ckpt.get('settings',{}).get('tokens',1024),threshold,ckpt['instances_trained'],units)
    result['source_contours']=str(Path(contours).resolve());result['checkpoint']=str(Path(checkpoint).resolve())
    if measure:
        from .measurement import MeasurementRouter
        router=MeasurementRouter();result['measurements']=[router.measure(raw,f) for f in result['features']+result.get('compound_features',[])]
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
    np.savez_compressed(out.with_suffix('.probabilities.npz'),**arrays)
    print(f"Wrote {len(result['features'])} feature selections to {output}")
    return result
