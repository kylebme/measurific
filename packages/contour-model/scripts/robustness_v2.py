"""Probe contour-format invariance on fixed held-out parts, without nominal input."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from contour_model.validation import indexed_records
from contour_model.data import adapted_labels
from contour_model.inference import predict
from contour_model.train import load_model,scores


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('checkpoint');parser.add_argument('source');parser.add_argument('output')
    parser.add_argument('--limit',type=int,default=47);parser.add_argument('--split',default='test')
    args=parser.parse_args();torch.set_num_threads(2);model,ckpt=load_model(args.checkpoint)
    names=['baseline','rotated','cyclic_start','reversed','double_density','ring_order','scale_translation']
    confusion={k:np.zeros((17,17)) for k in names};counts=[];seen=set()
    for meta,c in indexed_records(args.source,args.split):
        if meta['family'] in seen:continue
        seen.add(meta['family']);_,truth,_=adapted_labels(meta,c);xy=c['actual_xy_mm'];off=c['ring_offsets']
        weight=np.concatenate([np.linalg.norm(np.roll(xy[lo:hi],-1,axis=0)-xy[lo:hi],axis=1) for lo,hi in zip(off[:-1],off[1:])])
        cyclic=[];rev=[];revmap=[];dense=[];order=[];lengths=[]
        for lo,hi in zip(off[:-1],off[1:]):
            q=xy[lo:hi];n=hi-lo;cyclic.extend(np.roll(np.arange(lo,hi),n//3))
            rev.extend(np.arange(lo,hi)[::-1]);revmap.extend(lo+(np.arange(n)[::-1]-1)%n)
            dense.append(np.stack([q,(q+np.roll(q,-1,axis=0))/2],1).reshape(-1,2))
        for lo,hi in reversed(list(zip(off[:-1],off[1:]))):order.extend(range(lo,hi));lengths.append(hi-lo)
        angle=.731;rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        variants={'baseline':(xy,off,truth,weight),'rotated':(xy@rotation.T,off,truth,weight),
                  'cyclic_start':(xy[cyclic],off,truth[cyclic],weight[cyclic]),
                  'reversed':(xy[rev],off,truth[revmap],weight[revmap]),
                  'double_density':(np.concatenate(dense),off*2,np.repeat(truth,2),np.repeat(weight/2,2)),
                  'ring_order':(xy[order],np.r_[0,np.cumsum(lengths)],truth[order],weight[order]),
                  'scale_translation':(xy*7.3+[123,-49],off,truth,weight)}
        row={'id':meta['sample_id'],'family':meta['family'],'accepted_features':{}}
        for name,(points,offsets,t,w) in variants.items():
            result,p=predict(model,{'actual_xy_mm':points,'ring_offsets':offsets},instances=ckpt['instances_trained'])
            confusion[name]+=np.bincount(17*t+p['classes'],weights=w,minlength=289).reshape(17,17)
            row['accepted_features'][name]=sum(f['status']=='accepted' and f['grouping']!='semantic_parent' for f in result['features'])
        counts.append(row)
        print(f"{len(counts)}/{args.limit} {meta['family']}",flush=True)
        if len(counts)>=args.limit:break
    report={'samples':len(counts),'checkpoint':str(Path(args.checkpoint).resolve()),'split':args.split,
            'results':{k:scores(v) for k,v in confusion.items()},'sample_results':counts,
            'notes':['First part per family in ID order; same underlying held-out geometry across probes.',
                     'Semantic scores use original physical arc lengths. Instance counts are a diagnostic, not correspondence accuracy.',
                     'Adaptive sampling protocol comes from the checkpoint. Not a real-image or calibration test.']}
    Path(args.output).write_text(json.dumps(report,indent=2))
    print({k:v['macro_f1'] for k,v in report['results'].items()},flush=True)

if __name__=='__main__':main()
