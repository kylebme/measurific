"""Render imperfect observations, run production PAE, transfer original identities.

Strict pilot: reject changed topology, lost instances or nonmonotone mappings.
No nominal fitting or prediction is involved in transferring labels.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFilter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from .data import load_cached,ring_geometry
from .contact_sheets import source_paths
from .sampling import prepare_adaptive
from .corpus_primitives import targets

FIELDS=('edge_base_class','edge_base_instance','edge_compound_class','edge_compound_instance',
        'edge_feature_id','edge_anomaly_class')


def render(raw,seed,side=None):
    rng=np.random.default_rng(seed);side=int(side or rng.choice([640,960,1280]))
    xy,offsets,holes=ring_geometry(raw); extent=np.ptp(xy,axis=0)
    ppm=(side-48)/extent.max(); origin=xy.min(0)
    truth=(xy-origin)*[ppm,-ppm]+[24+rng.random(),24+extent[1]*ppm+rng.random()]
    shape=np.ceil(np.ptp(truth,axis=0)+50).astype(int)
    dark=int(rng.integers(15,65));light=int(rng.integers(190,245));aa=3
    image=Image.new('L',tuple(shape*aa),light);draw=ImageDraw.Draw(image)
    for hole in (False,True):
        for i,(lo,hi) in enumerate(zip(offsets[:-1],offsets[1:])):
            if holes[i]==hole:draw.polygon([tuple(p) for p in truth[lo:hi]*aa],fill=light if hole else dark)
    image=image.resize(tuple(shape),Image.Resampling.LANCZOS)
    blur=float(rng.uniform(.35,1.1));noise=float(rng.uniform(0,1.2))
    image=image.filter(ImageFilter.GaussianBlur(blur))
    image=np.clip(np.asarray(image,dtype=float)+rng.normal(0,noise,(shape[1],shape[0])),0,255).astype(np.uint8)
    return image,truth,dict(side=side,ppm=ppm,blur=blur,noise=noise,dark=dark,light=light,seed=seed)


def runs(ids):
    starts=np.flatnonzero(ids!=np.roll(ids,1))
    if not len(starts):return [int(ids[0])]
    values=ids[starts].tolist();i=int(np.argmin(values));return values[i:]+values[:i]


def project(q,ref):
    """Dense search samples prevent long source edges being missed by vertex KNN."""
    edge=np.roll(ref,-1,axis=0)-ref;length=np.linalg.norm(edge,axis=1)
    count=np.maximum(1,np.ceil(length/.5).astype(int))
    owner=np.repeat(np.arange(len(ref)),count)
    fraction=np.concatenate([np.arange(n)/n for n in count])
    dense=ref[owner]+fraction[:,None]*edge[owner]
    _,nearest=cKDTree(dense).query(q,k=min(8,len(dense)))
    candidates=owner[np.asarray(nearest).reshape(len(q),-1)]
    delta=q[:,None]-ref[candidates]
    t=np.clip((delta*edge[candidates]).sum(-1)/length[candidates]**2,0,1)
    distance=np.linalg.norm(delta-t[:,:,None]*edge[candidates],axis=-1)
    best=distance.argmin(1);idx=candidates[np.arange(len(q)),best]
    s=np.r_[0,np.cumsum(length)][idx]+t[np.arange(len(q)),best]*length[idx]
    return idx,s,distance.min(1),length.sum()


def transfer(raw,truth,observed,obs_offsets,max_distance=1.5):
    offsets=raw['ring_offsets'];_,_,roles=ring_geometry(dict(actual_xy_mm=truth,ring_offsets=offsets))
    _,_,oroles=ring_geometry(dict(actual_xy_mm=observed,ring_offsets=obs_offsets))
    if len(roles)!=len(oroles):raise ValueError('Changed ring topology')
    refs=[truth[a:b] for a,b in zip(offsets[:-1],offsets[1:])]
    obs=[observed[a:b] for a,b in zip(obs_offsets[:-1],obs_offsets[1:])]
    cost=np.array([[np.linalg.norm(a.mean(0)-b.mean(0))+(1e9 if roles[i]!=oroles[j] else 0)
        for j,b in enumerate(obs)] for i,a in enumerate(refs)])
    rows,cols=linear_sum_assignment(cost);out={k:[] for k in FIELDS};coordinates=[];new_offsets=[0];errors=[]
    area=lambda q:np.sum(q[:,0]*np.roll(q[:,1],-1)-q[:,1]*np.roll(q[:,0],-1))
    for ri,oi in zip(rows,cols):
        ref=refs[ri];q=obs[oi].copy()
        if np.sign(area(ref))!=np.sign(area(q)):q=q[::-1].copy()
        _,s,d,perimeter=project(q,ref)
        ds=(np.roll(s,-1)-s+perimeter/2)%perimeter-perimeter/2
        if d.max()>max_distance:raise ValueError('Correspondence exceeds 1.5 px')
        if ds.min()<-.15 or abs(ds.sum()-perimeter)>1e-4*perimeter:
            raise ValueError('Nonmonotone or incomplete ring correspondence')
        idx,_,distance,_=project((q+np.roll(q,-1,axis=0))/2,ref)
        if distance.max()>max_distance:raise ValueError('Edge correspondence exceeds 1.5 px')
        source_ids=offsets[ri]+idx
        if runs(raw['edge_base_instance'][source_ids])!=runs(raw['edge_base_instance'][offsets[ri]:offsets[ri+1]]):
            raise ValueError('Lost or reordered base instance')
        for k in FIELDS:out[k].append(raw[k][source_ids])
        coordinates.append(q);new_offsets.append(new_offsets[-1]+len(q));errors.extend(d.tolist())
    result={k:np.concatenate(v) for k,v in out.items()}
    result.update(actual_xy_mm=np.concatenate(coordinates),ring_offsets=np.array(new_offsets))
    return result,dict(max_distance_px=float(max(errors)),rms_distance_px=float(np.sqrt(np.mean(np.square(errors)))))


def write_one(job):
    entry,source,output=job;root=Path(output);meta_path,raw_path=source_paths(source,entry)
    raw=load_cached(raw_path);meta=json.loads(meta_path.read_text());seed=int(entry['id'])+381207
    image,truth,settings=render(raw,seed)
    ids=raw['edge_base_instance'];weights=np.concatenate([np.linalg.norm(np.roll(raw['actual_xy_mm'][a:b],-1,axis=0)-raw['actual_xy_mm'][a:b],axis=1) for a,b in zip(raw['ring_offsets'][:-1],raw['ring_offsets'][1:])])
    _,inverse=np.unique(ids,return_inverse=True)
    minimum=float(np.bincount(inverse,weights=weights).min()*settings['ppm'])
    audit=dict(id=entry['id'],family=entry['family'],split=entry['split'],minimum_feature_px=minimum,**settings)
    try:
        from inspection_measurement import contours_from_image
        extracted,diagnostics=contours_from_image(image)
        observed,evidence=transfer(raw,truth,extracted.xy,extracted.offsets)
        # Restore physical scale and y-up convention, but do not alter PAE shapes.
        observed['actual_xy_mm']=(observed['actual_xy_mm']-truth[0])*[1/settings['ppm'],-1/settings['ppm']]+raw['actual_xy_mm'][0]
        annotations={k:observed[k] for k in FIELDS if k.startswith('edge_base') or k.startswith('edge_compound')}
        prepared=prepare_adaptive(observed)
        if len(prepared.x)>8192:raise ValueError('PAE output exceeds pilot 8192-token budget')
        t,_,_=targets(meta,observed,prepared,annotation_override=(annotations,meta['compound_instances']))
        ident=str(int(entry['id'])+100000000)
        folder=root/'sources';folder.mkdir(exist_ok=True)
        np.savez_compressed(root/(ident+'.npz'),x=prepared.x,offsets=prepared.offsets,weights=prepared.weights,**t)
        np.savez_compressed(folder/(ident+'.contours.npz'),**observed)
        Image.fromarray(image).save(folder/(ident+'.png'))
        meta.update(sample_id=ident,parent_sample_id=entry['id'],pae_render=settings,pae_transfer=evidence,source_contours=str(raw_path),
            observation_pipeline='render-to-production-PAE',extraction_method=diagnostics['method'])
        (folder/(ident+'.json')).write_text(json.dumps(meta))
        return dict(entry,id=ident,parent_id=entry['id'],file=ident+'.npz',tokens=len(prepared.x)),dict(audit,accepted=True,**evidence)
    except ValueError as exc:
        return None,dict(audit,accepted=False,reason=str(exc))


def build(source,output,limit=2400,workers=2):
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    all_entries=json.loads((source/'index.json').read_text())
    # Balanced round-robin family/split selection, randomized within each cell.
    rng=np.random.default_rng(1881);rng.shuffle(all_entries);buckets={}
    for e in all_entries:buckets.setdefault((e['family'],e['split']),[]).append(e)
    entries=[]
    while len(entries)<limit and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and len(entries)<limit:entries.append(buckets[key].pop())
    config=json.loads((source/'config.json').read_text());config.update(source='render-production-PAE-v1',
        parent_source=str(source),source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),attempted=len(entries))
    (output/'config.json').write_text(json.dumps(config,indent=2));rows=[];audit=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for row,result in pool.map(write_one,((e,str(source),str(output)) for e in entries)):
            if row:rows.append(row)
            audit.append(result)
            if len(audit)%10==0:print(f'PAE {len(audit)}/{len(entries)} attempted, {len(rows)} accepted',flush=True)
    (output/'index.json').write_text(json.dumps(rows,indent=2));(output/'audit.json').write_text(json.dumps(audit,indent=2))
    print(dict(accepted=len(rows),attempted=len(entries),rejections=dict(Counter(a.get('reason') for a in audit if not a['accepted']))),flush=True)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');p.add_argument('--limit',type=int,default=2400);p.add_argument('--workers',type=int,default=2)
    build(**vars(p.parse_args()))
