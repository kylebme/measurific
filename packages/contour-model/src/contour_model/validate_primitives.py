"""Raw-edge instance validation and CPU timing, separate from token metrics."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from .train import load_model
from .data import remap_path
from .inference import predict
from .inference_primitives import base_names_for_schema, COMPOUND_NAMES


def instance_counts(truth_ids,truth_classes,features,weights,names):
    actual_ids=np.unique(truth_ids);actual_ids=actual_ids[actual_ids!=0]
    owner=np.full(len(truth_ids),-1,int)
    for i,f in enumerate(features):owner[f['raw_edge_indices']]=i
    counts=np.zeros((len(names),3),int)
    for cls in range(1,len(names)):
        tids=[v for v in actual_ids if truth_classes[np.flatnonzero(truth_ids==v)[0]]==cls]
        pids=[i for i,f in enumerate(features) if f['semantic_id']==cls]
        counts[cls,1:]=[len(pids),len(tids)]
        if not tids or not pids:continue
        tmap=np.full(int(truth_ids.max())+1,-1,int);tmap[tids]=np.arange(len(tids))
        pmap=np.full(len(features),-1,int);pmap[pids]=np.arange(len(pids))
        t=tmap[truth_ids];p=np.full(len(owner),-1,int);selected=owner>=0;p[selected]=pmap[owner[selected]]
        valid=(t>=0)&(p>=0)
        inter=np.bincount(p[valid]*len(tids)+t[valid],weights=weights[valid],minlength=len(pids)*len(tids)).reshape(len(pids),len(tids))
        ps=np.bincount(p[p>=0],weights=weights[p>=0],minlength=len(pids))
        ts=np.bincount(t[t>=0],weights=weights[t>=0],minlength=len(tids))
        iou=inter/np.maximum(ps[:,None]+ts[None]-inter,1e-12)
        ri,ci=linear_sum_assignment(-iou);counts[cls,0]=int((iou[ri,ci]>=.5).sum())
    return counts


def summarize(counts,names):
    def one(v):
        tp,p,t=map(int,v)
        return dict(matched=tp,predicted=p,reference=t,precision=tp/max(p,1),recall=tp/max(t,1),f1=2*tp/max(p+t,1))
    per_class={name:one(counts[i]) for i,name in enumerate(names) if i}
    present=[v['f1'] for v in per_class.values() if v['reference']]
    return {'overall':one(counts.sum(0)), 'macro_f1':float(np.mean(present)) if present else 0.,'classes':per_class}


def validate(checkpoint,cache,output,split='test',limit=200,device='cpu'):
    torch.set_num_threads(4);model,ckpt=load_model(checkpoint,device)
    base_names=base_names_for_schema(model.config.label_schema)
    cache=Path(cache);entries=[e for e in json.loads((cache/'index.json').read_text()) if e['split']==split]
    np.random.default_rng(811).shuffle(entries)
    if limit:entries=entries[:limit]
    base=np.zeros((len(base_names),3),int);compound=np.zeros((5,3),int);latency=[];rows=[]
    for e in entries:
        path=Path(e['file']);path=remap_path(path if path.is_absolute() else cache/path)
        source=path.parent/'sources'/(e['id']+'.contours.npz')
        with np.load(source) as z:c={k:z[k] for k in z.files}
        raw={k:c[k] for k in ('actual_xy_mm','ring_offsets')}
        start=time.perf_counter();result,_=predict(model,raw,instances=True);seconds=time.perf_counter()-start
        latency.append(seconds)
        weights=np.concatenate([np.linalg.norm(np.roll(c['actual_xy_mm'][lo:hi],-1,axis=0)-c['actual_xy_mm'][lo:hi],axis=1)
                                for lo,hi in zip(c['ring_offsets'][:-1],c['ring_offsets'][1:])])
        bc=instance_counts(c['edge_base_instance'],c['edge_base_class'],result['features'],weights,base_names)
        cc=instance_counts(c['edge_compound_instance'],c['edge_compound_class'],result['compound_features'],weights,COMPOUND_NAMES)
        base+=bc;compound+=cc
        rows.append(dict(id=e['id'],family=e['family'],seconds=seconds,points=len(weights),tokens=e['tokens'],
                         base=summarize(bc,base_names)['overall'],compound=summarize(cc,COMPOUND_NAMES)['overall']))
        if len(rows)%25==0:print(f'Validated {len(rows)}/{len(entries)}',flush=True)
    report=dict(samples=len(entries),split=split,checkpoint=str(Path(checkpoint).resolve()),device=device,
                checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                source_step=ckpt['step'],label_version=ckpt['label_version'],
                data_fingerprint=hashlib.sha256((cache/'config.json').read_bytes()+(cache/'index.json').read_bytes()).hexdigest(),
                base_instances_iou50=summarize(base,base_names),compound_instances_iou50=summarize(compound,COMPOUND_NAMES),
                inference_seconds={'median':float(np.median(latency)),'p95':float(np.quantile(latency,.95)),'max':float(max(latency))},
                rows=rows)
    Path(output).write_text(json.dumps(report,indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('cache');p.add_argument('output')
    p.add_argument('--split',default='test');p.add_argument('--limit',type=int,default=200);p.add_argument('--device',default='cpu')
    validate(**vars(p.parse_args()))
