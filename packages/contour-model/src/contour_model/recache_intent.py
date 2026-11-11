"""Immutable overlay: correct reference-free targets, keep explicit nominal intent.

Unchanged cache entries reference the parent disk corpus by absolute path. Only
corrected entries and their provenance are written anew; parent must stay mounted.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import numpy as np
from .corpus_primitives import targets
from .data import load_cached
from .geometric import GeometricConfig, _optimize, segment_ring
from .sampling import prepare_adaptive

POLICY='nominal-freeform-fit-v1: line first; tolerance max(0.01mm, 0.5% open chord); closed 0.01mm; split if neither line nor circle fits'


def corrected_annotations(metadata, raw):
    annotations={k:raw[k].copy() for k in ('edge_base_class','edge_base_instance',
        'edge_compound_class','edge_compound_instance')}
    next_id=int(annotations['edge_base_instance'].max())+1
    nominal=raw['nominal_xy_mm']; pids=raw['edge_primitive_id']
    for p in metadata['primitives']:
        if p['kind']!='freeform': continue
        for lo,hi in zip(raw['ring_offsets'][:-1],raw['ring_offsets'][1:]):
            selected=pids[lo:hi]==p['id']; n=hi-lo
            if not selected.any(): continue
            if selected.all():
                parts,_=segment_ring(nominal[lo:hi],GeometricConfig(tolerance=.01,refinement_passes=0))
                runs=[(ids+lo,fit['kind']) for ids,fit in parts]
            else:
                runs=[]
                for start in np.flatnonzero(selected & ~np.roll(selected,1)):
                    count=1
                    while selected[(start+count)%n]: count+=1
                    ids=lo+(start+np.arange(count))%n
                    q=nominal[np.r_[ids,lo+(start+count)%n]]
                    cfg=GeometricConfig(tolerance=max(.01,.005*np.linalg.norm(q[-1]-q[0])),refinement_passes=0)
                    spans,_=_optimize(q,cfg)
                    runs.extend((ids[a:b],fit['kind']) for a,b,fit in spans)
            for ids,kind in runs:
                annotations['edge_base_class'][ids]={'line':1,'arc':2,'circle':3}[kind]
                annotations['edge_base_instance'][ids]=next_id; next_id+=1
    groups=[]
    for group in metadata['compound_instances']:
        mask=annotations['edge_compound_instance']==group['id']
        groups.append(dict(group,base_instance_ids=np.unique(annotations['edge_base_instance'][mask]).tolist()))
    return annotations,groups


def write_one(job):
    entry,source,output=job; source=Path(source); output=Path(output)
    original=source/entry['file']; raw_path=original.parent/'sources'/(entry['id']+'.contours.npz')
    meta_path=raw_path.with_name(entry['id']+'.json'); meta=json.loads(meta_path.read_text())
    if not any(p['kind']=='freeform' for p in meta['primitives']):
        return dict(entry,file=str(original.resolve()),intent_relabelled=False)
    raw=load_cached(raw_path); annotation=corrected_annotations(meta,raw)
    prepared=prepare_adaptive(raw)
    t,annotations,groups=targets(meta,raw,prepared,annotation_override=annotation)
    cached=load_cached(original)
    # Relabeling must not move observations, resample differently or leak labels into inputs.
    np.testing.assert_allclose(prepared.x,cached['x'],rtol=0,atol=1e-6)
    np.testing.assert_array_equal(prepared.offsets,cached['offsets'])
    destination=output/'corrected'/(entry['id']+'.npz'); destination.parent.mkdir(exist_ok=True)
    np.savez_compressed(destination,x=prepared.x,offsets=prepared.offsets,weights=prepared.weights,**t)
    raw.update(annotations); meta.update(compound_instances=groups,freeform_policy=POLICY)
    folder=destination.parent/'sources'; folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/raw_path.name,**raw)
    (folder/meta_path.name).write_text(json.dumps(meta))
    return dict(entry,file=str(destination.relative_to(output)),intent_relabelled=True)


def build(source,output,workers=2,limit=None):
    source=Path(source); output=Path(output); output.mkdir(parents=True,exist_ok=False)
    config=json.loads((source/'config.json').read_text())
    config.update(source='intent-overlay-v1',parent_source=str(source.resolve()),freeform_policy=POLICY,
        source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'config.json').write_text(json.dumps(config,indent=2))
    entries=json.loads((source/'index.json').read_text())
    if limit: entries=entries[:limit]
    rows=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for row in pool.map(write_one,((e,str(source),str(output)) for e in entries),chunksize=1):
            rows.append(row)
            if len(rows)%100==0: print(f'Intent cache {len(rows)}/{len(entries)}',flush=True)
    (output/'index.json').write_text(json.dumps(rows,indent=2))
    print(f'Completed {len(rows)} records; corrected {sum(e["intent_relabelled"] for e in rows)}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output')
    p.add_argument('--workers',type=int,default=2);p.add_argument('--limit',type=int)
    build(**vars(p.parse_args()))
