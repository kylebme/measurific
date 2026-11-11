import json
from pathlib import Path
import numpy as np,torch
from contour_model.data import records,adapted_labels
from contour_model.inference import predict
from contour_model.train import load_model,scores
import argparse
p=argparse.ArgumentParser(description='Fixed-part sampling and contour robustness probe')
p.add_argument('checkpoint');p.add_argument('source');p.add_argument('output');p.add_argument('--limit',type=int,default=100)
args=p.parse_args();torch.set_num_threads(4)
model,ckpt=load_model(args.checkpoint);conf={k:np.zeros((17,17)) for k in ['512','1024','2048','rotated','cyclic_start','reversed','double_density']};count=0
for m,c in records(args.source):
    if m['split']!='test':continue
    _,truth,_=adapted_labels(m,c);xy=c['actual_xy_mm'];off=c['ring_offsets'];w=np.zeros(len(xy));perm=[];rev=[];revmap=[];dense=[];dense_truth=[]
    for lo,hi in zip(off[:-1],off[1:]):
        q=xy[lo:hi];n=hi-lo;w[lo:hi]=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
        perm.extend(np.roll(np.arange(lo,hi),n//3));rev.extend(np.arange(lo,hi)[::-1]);revmap.extend(lo+(np.arange(n)[::-1]-1)%n)
        dense.append(np.stack([q,(q+np.roll(q,-1,axis=0))/2],1).reshape(-1,2));dense_truth.extend(np.repeat(truth[lo:hi],2))
    angle=.731;rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    variants={'512':(xy,off,truth,w,512),'1024':(xy,off,truth,w,1024),'2048':(xy,off,truth,w,2048),
        'rotated':(xy@rot.T,off,truth,w,1024),'cyclic_start':(xy[perm],off,truth[perm],w[perm],1024),
        'reversed':(xy[rev],off,truth[revmap],w[revmap],1024),'double_density':(np.concatenate(dense),off*2,np.array(dense_truth),np.repeat(w/2,2),1024)}
    for key,(points,offsets,t,weight,tokens) in variants.items():
        _,p=predict(model,{'actual_xy_mm':points,'ring_offsets':offsets},tokens=tokens,instances=True)
        conf[key]+=np.bincount(17*t+p['classes'],weights=weight,minlength=289).reshape(17,17)
    count+=1
    if count==args.limit:break
report={'samples':count,'results':{k:scores(v) for k,v in conf.items()},'note':'Same synthetic test parts across probes; not calibration or real-data validation.'}
Path(args.output).write_text(json.dumps(report,indent=2));print({k:v['macro_f1'] for k,v in report['results'].items()})
