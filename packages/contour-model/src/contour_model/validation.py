"""Raw-contour selection, measurement and invariance validation."""
from pathlib import Path
import json,time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from .data import records,adapted_labels,NAMES
from .inference import predict,describe_selection,ROUTES
from .measurement import MeasurementRouter
from .train import load_model


def observable_reference_labels(metadata,contours):
    """Combine unobservable nominal subdivisions of one flat measurement surface.

    Nominal coordinates are reference-only: they never enter prediction or fitting.
    The frozen training cache can retain older primitive subdivisions.
    """
    ids,classes,semantic=adapted_labels(metadata,contours)
    nominal=contours.get('nominal_xy_mm')
    if nominal is None:return ids,classes,semantic
    groups=flat_surface_groups(ids,classes,nominal,contours['ring_offsets'])
    unique,inverse=np.unique(ids,return_inverse=True)
    ids=np.array([groups.get(int(i),int(i)) for i in unique],dtype=ids.dtype)[inverse]
    return ids,classes,semantic


def flat_surface_groups(ids,classes,nominal,offsets):
    """Reference feature-ID aliases for exact nominal collinear flat continuations."""
    parent={int(i):int(i) for i in np.unique(ids)}
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for lo,hi in zip(offsets[:-1],offsets[1:]):
        q=nominal[lo:hi];d=np.roll(q,-1,axis=0)-q;d/=np.maximum(np.linalg.norm(d,axis=1,keepdims=True),1e-15)
        before=np.roll(d,1,axis=0);collinear=(np.abs(d[:,0]*before[:,1]-d[:,1]*before[:,0])<1e-8)&((d*before).sum(1)>1-1e-8)
        labels=ids[lo:hi];cl=classes[lo:hi]
        for at in np.flatnonzero(collinear&(cl==4)&(np.roll(cl,1)==4)&(labels!=np.roll(labels,1))):
            a=root(int(labels[at]));b=root(int(labels[(at-1)%len(labels)]));parent[max(a,b)]=min(a,b)
    return {i:root(i) for i in parent if root(i)!=i}


def indexed_records(source,split,sample_ids=None):
    source=Path(source)
    if (source/'index.json').exists():
        entries=json.loads((source/'index.json').read_text())
        for entry in sorted(entries,key=lambda e:e['id']):
            if entry['split']!=split or (sample_ids is not None and entry['id'] not in sample_ids):continue
            directory=source/Path(entry['file']).parent/'sources';path=Path(entry.get('source_contours',directory/(entry['id']+'.contours.npz')))
            if not path.exists():raise FileNotFoundError(path)
            meta=json.loads(Path(entry.get('source_metadata',path.with_name(entry['id']+'.json'))).read_text())
            with np.load(path,allow_pickle=False) as a:yield meta,{k:a[k] for k in a.files}
    else:
        for meta,c in records(source):
            if meta['split']==split and (sample_ids is None or meta['sample_id'] in sample_ids):yield meta,c


def selection_report(checkpoint,source,output,split='test',limit=150,tokens=1024,sample_ids=None,threshold=.7,measure=True):
    torch.set_num_threads(4);model,ckpt=load_model(checkpoint);router=MeasurementRouter()
    count=0;tp=0;npred=0;ngt=0;measurements=[];matched_by_class={};timings=[];examples=[];sample_results=[];class_counts={};strict_tp=np.zeros(2,int);matched_ious=[]
    for meta,c in indexed_records(source,split,sample_ids):
        begin=time.perf_counter();result,probs=predict(model,c,tokens,threshold=threshold,instances=ckpt['instances_trained']);timings.append(time.perf_counter()-begin)
        ids,classes,_=observable_reference_labels(meta,c);gt=[]
        for iid in np.unique(ids):
            edges=np.flatnonzero(ids==iid);cls=int(classes[edges[0]])
            gt.append({'id':str(iid),'semantic_id':cls,'semantic':NAMES[cls],'raw_edge_indices':edges.tolist(),
                       'contour_spans':describe_selection(edges,c),'confidence':1.,'status':'accepted',
                       'measurement_route':ROUTES[cls],'units':meta.get('units','mm')})
        pred=[f for f in result['features'] if f['status']=='accepted' and f['grouping']!='semantic_parent']
        xy=c['actual_xy_mm'];weight=np.zeros(len(xy))
        for lo,hi in zip(c['ring_offsets'][:-1],c['ring_offsets'][1:]):weight[lo:hi]=np.linalg.norm(np.roll(xy[lo:hi],-1,axis=0)-xy[lo:hi],axis=1)
        # Ground truth is an edge partition: histogram intersections avoid a dense
        # raw_edges × instances matrix multiplication on hundreds-hole parts.
        truth_owner=np.empty(len(xy),np.int64)
        for i,f in enumerate(gt):truth_owner[f['raw_edge_indices']]=i
        truth_length=np.bincount(truth_owner,weights=weight,minlength=len(gt))
        inter=np.zeros((len(pred),len(gt)));pred_length=np.zeros(len(pred))
        for i,f in enumerate(pred):
            edges=f['raw_edge_indices'];inter[i]=np.bincount(truth_owner[edges],weights=weight[edges],minlength=len(gt));pred_length[i]=weight[edges].sum()
        union=pred_length[:,None]+truth_length[None]-inter
        iou=inter/np.maximum(union,1e-8)
        if len(pred):iou*=np.array([f['semantic_id'] for f in pred])[:,None]==np.array([f['semantic_id'] for f in gt])[None]
        rows,cols=linear_sum_assignment(-iou);valid=iou[rows,cols]>=.5
        strict_tp+=[int((iou[rows,cols]>=level).sum()) for level in (.75,.9)]
        matched_ious.extend(iou[rows[valid],cols[valid]].tolist())
        tp+=valid.sum();npred+=len(pred);ngt+=len(gt)
        for f in gt:class_counts.setdefault(f['semantic'],[0,0,0])[2]+=1
        for f in pred:class_counts.setdefault(f['semantic'],[0,0,0])[1]+=1
        for col in cols[valid]:class_counts[gt[col]['semantic']][0]+=1
        sample_results.append({'id':meta['sample_id'],'family':meta['family'],'rings':len(c['ring_offsets'])-1,
                               'stress':meta.get('stress',False),'complexity':meta.get('complexity','ordinary'),
                               'matched':int(valid.sum()),'predicted':len(pred),'reference':len(gt)})
        for r,col in zip(rows[valid],cols[valid]):
            semantic=gt[col]['semantic'];matched_by_class[semantic]=matched_by_class.get(semantic,0)+1
            if not measure:continue
            estimated=router.measure(c,pred[r]);oracle=router.measure(c,gt[col])
            if oracle['status']!='measured':continue
            entry={'sample':meta['sample_id'],'semantic':semantic,'predicted_status':estimated['status'],'units':gt[col]['units']}
            if estimated['status']=='measured':
                keys=('diameter','width','overall_length','pitch','projected_length','rms_profile_residual')
                entry['absolute_selection_differences']={k:abs(estimated['values'][k]-oracle['values'][k]) for k in keys if k in oracle['values'] and k in estimated['values']}
            measurements.append(entry)
        if len(examples)<3:examples.append({'sample':meta['sample_id'],'selections':len(pred),'reference_instances':len(gt)})
        count+=1
        if count>=limit:break
    differences={}
    for item in measurements:
        for key,value in item.get('absolute_selection_differences',{}).items():differences.setdefault(key,[]).append(value)
    summary={'samples':count,'accepted_instance_precision_iou50':float(tp/max(npred,1)),
             'accepted_instance_recall_iou50':float(tp/max(ngt,1)),'matched_features':int(tp),
             'reference_features':ngt,'accepted_features':npred,'matched_by_class':matched_by_class,
             'reference_measurable_matches':len(measurements),'successful_predicted_measurements':sum(m['predicted_status']=='measured' for m in measurements),
             'selection_induced_measurement_difference':{k:{'median':float(np.median(v)),'p95':float(np.quantile(v,.95)),'count':len(v)} for k,v in differences.items()},
             'cpu_seconds_per_part':{'median':float(np.median(timings)),'p95':float(np.quantile(timings,.95)),'threads':4},
             'notes':['Reference and predicted selectors use identical imperfect raw observations.','Projected line length is selected support extent, not a virtual-sharp feature length.','These are selection errors, not metrology accuracy or standardized GD&T validation.'],
             'examples':examples,'measurements':measurements,'sample_results':sample_results,
             'checkpoint':str(Path(checkpoint).resolve()),'split':split,'threshold':threshold,'measurement_comparison_enabled':measure,
             'reference_policy':'observable-flat-surfaces-v2: adjacent nominally collinear flat subdivisions share one measurement instance'}
    def metrics(values):
        matched,predicted,reference=map(int,values)
        return {'matched':matched,'predicted':predicted,'reference':reference,
                'precision_iou50':matched/max(predicted,1),'recall_iou50':matched/max(reference,1)}
    summary['by_class']={key:metrics(values) for key,values in class_counts.items()}
    summary['stricter_instance_matching']={str(level):{'matched':int(n),'precision':float(n/max(npred,1)),'recall':float(n/max(ngt,1))}
                                           for level,n in zip((.75,.9),strict_tp)}
    summary['matched_iou']={'median':float(np.median(matched_ious)),'p10':float(np.quantile(matched_ious,.1))} if matched_ious else None
    grouped={};group_counts={}
    for row in sample_results:
        holes=row['rings']-1
        groups=['family:'+row['family'],'form:'+('stress' if row['stress'] else 'ordinary'),
                'holes:'+('0' if holes==0 else '1-99' if holes<100 else '100-299' if holes<300 else '300+'),
                'features:'+('<50' if row['reference']<50 else '50-199' if row['reference']<200 else '200+')]
        for group in groups:
            grouped.setdefault(group,np.zeros(3,int));grouped[group]+=[row['matched'],row['predicted'],row['reference']]
            group_counts[group]=group_counts.get(group,0)+1
    summary['strata']={key:{'samples':group_counts[key],**metrics(values)} for key,values in grouped.items()}
    Path(output).write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='measurements'},indent=2))
    return summary
