"""Correct old buffered-outline labels without modifying any observation."""
import numpy as np
from scipy.spatial import cKDTree
from .primitive_labels import add_polycurve


class Capture:
    spacing=.055
    def __init__(self):self.spans=[]
    def line(self,a,b,fid):
        self.spans.append((np.asarray(a),np.asarray(b),'line',dict(start=list(a),end=list(b))))
    def add(self,points,fid,kind,**params):
        self.spans.append((np.asarray(points[0]),np.asarray(points[-1]),kind,params))


def repair(metadata,contours):
    primitives={p['id']:p for p in metadata['primitives']}
    edge_ids=contours['edge_primitive_id'].copy();xy=contours['nominal_xy_mm']
    changed=[];next_id=max(primitives)+1
    for ri,(lo,hi) in enumerate(zip(contours['ring_offsets'][:-1],contours['ring_offsets'][1:])):
        q=xy[lo:hi];lengths=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
        bad=[]
        for pid in np.unique(edge_ids[lo:hi]):
            if primitives[int(pid)]['kind']!='circular_arc':continue
            selected=lengths[edge_ids[lo:hi]==pid]
            if len(selected)==2 and selected.max()>1.01*selected.min():bad.append(int(pid))
        if not bad:continue
        fids=np.unique(contours['edge_feature_id'][lo:hi])
        if len(fids)!=1:raise ValueError('Unexpected buffered-outline feature ownership')
        fid=int(fids[0]);capture=Capture();add_polycurve(capture,q,fid)
        tree=cKDTree(q);assigned=np.zeros(len(q),bool)
        for a,b,kind,params in capture.spans:
            distances,indices=tree.query(np.stack([a,b]))
            if distances.max()>1e-6:raise ValueError('Decomposition endpoint lost source correspondence')
            start,end=map(int,indices);edges=np.arange(start,end if end>start else end+len(q))%len(q)
            if assigned[edges].any():raise ValueError('Overlapping repaired spans')
            assigned[edges]=True;edge_ids[lo+edges]=next_id
            primitives[next_id]=dict(id=next_id,kind=kind,kind_id={'line':1,'circular_arc':2,'freeform':3}[kind],
                                      feature_id=fid,nominal_parameters=params)
            next_id+=1
        if not assigned.all():raise ValueError('Incomplete repaired contour')
        changed.append({'ring_index':ri,'replaced_primitive_ids':bad})
    if changed:
        used=np.unique(edge_ids);mapping=np.zeros(int(used.max())+1,np.int32)
        result=[]
        for i,pid in enumerate(used,1):
            mapping[pid]=i;p=primitives[int(pid)].copy();p['id']=i;result.append(p)
        contours['edge_primitive_id']=mapping[edge_ids]
        metadata['primitives']=result
        metadata['primitive_label_repair']={'version':'buffer-chord-audit-v1','rings':changed,
                                             'observations_unchanged':True}
    return len(changed)
