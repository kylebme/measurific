"""CPU threshold-extraction bridge: model inputs remain contour-only.

This is a replaceable reference extractor, not a claim of station calibration.
Ring-constrained, direction-checked label transfer rejects topology changes and
ignores uncertain intervals rather than marking them as semantic background.
"""
from pathlib import Path
from io import BytesIO
import json,tarfile
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from PIL import Image
from .data import prepare,targets,ring_geometry,LABEL_VERSION


def extract(image,min_area=8):
    import cv2
    _,binary=cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    contours,hierarchy=cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:raise ValueError('No material component')
    rings=[]
    for c,h in zip(contours,hierarchy[0]):
        if abs(cv2.contourArea(c))<min_area:continue
        q=c[:,0,:].astype(float);q[:,1]*=-1 # x right, y up, pixel units
        rings.append(q)
    raw={'actual_xy_mm':np.concatenate(rings),'ring_offsets':np.r_[0,np.cumsum([len(r) for r in rings])]}
    ring_geometry(raw)
    return raw


def transfer(source,observed,max_distance=2.):
    truth=source['actual_xy_px'].copy();truth[:,1]*=-1
    offsets=source['ring_offsets'];_,_,roles=ring_geometry({'actual_xy_mm':truth,'ring_offsets':offsets})
    xy,obs_offsets,obs_roles=ring_geometry(observed)
    if len(roles)!=len(obs_roles):raise ValueError('Extraction changed ring topology')
    refs=[truth[a:b] for a,b in zip(offsets[:-1],offsets[1:])]
    obs=[xy[a:b].copy() for a,b in zip(obs_offsets[:-1],obs_offsets[1:])]
    cost=np.array([[np.linalg.norm(a.mean(0)-b.mean(0))+(1e6 if roles[i]!=obs_roles[j] else 0)
                    for j,b in enumerate(obs)] for i,a in enumerate(refs)])
    rows,cols=linear_sum_assignment(cost)
    output={k:[] for k in ('actual_xy_mm','edge_feature_id','edge_semantic_bits','edge_anomaly_class','edge_valid')}
    outoffsets=[0]
    for ri,oi in zip(rows,cols):
        q=obs[oi];ref=refs[ri]
        area=lambda a:np.sum(a[:,0]*np.roll(a[:,1],-1)-a[:,1]*np.roll(a[:,0],-1))
        if np.sign(area(q))!=np.sign(area(ref)):q=q[::-1].copy()
        midpoint=(q+np.roll(q,-1,axis=0))/2
        _,candidate=cKDTree(ref).query(midpoint,k=min(8,len(ref)))
        edge=ref[(candidate+1)%len(ref)]-ref[candidate]
        fraction=np.clip(((midpoint[:,None]-ref[candidate])*edge).sum(-1)/np.maximum((edge*edge).sum(-1),1e-12),0,1)
        projection=ref[candidate]+fraction[:,:,None]*edge
        distance=np.linalg.norm(projection-midpoint[:,None],axis=-1);best=distance.argmin(1);idx=candidate[np.arange(len(q)),best]
        seglen=np.linalg.norm(np.roll(ref,-1,axis=0)-ref,axis=1);arc=np.r_[0,np.cumsum(seglen)];s=arc[idx]+fraction[np.arange(len(q)),best]*seglen[idx]
        ds=(np.roll(s,-1)-s+arc[-1]/2)%arc[-1]-arc[-1]/2
        valid=(distance.min(1)<=max_distance)&(ds>=-.25)&(ds<=8)
        valid &= np.roll(valid,1)
        if valid.mean()<.8:raise ValueError('Ambiguous/non-monotone contour correspondence')
        source_indices=offsets[ri]+idx
        output['actual_xy_mm'].append(q);output['edge_valid'].append(valid)
        for key in ('edge_feature_id','edge_semantic_bits','edge_anomaly_class'):output[key].append(source[key][source_indices])
        outoffsets.append(outoffsets[-1]+len(q))
    output={k:np.concatenate(v) for k,v in output.items()};output['ring_offsets']=np.asarray(outoffsets,np.int64)
    return output


def build_bridge(source,output,tokens=1024,limit=300):
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=True);(output/'sources').mkdir(exist_ok=True)
    manifest=json.loads((source/'manifest.json').read_text());index=[];rejected=[]
    config={'source':str(source.resolve()),'tokens':tokens,'label_version':LABEL_VERSION,'extractor':'OpenCV Otsu + CHAIN_APPROX_NONE','units':'px'}
    (output/'config.json').write_text(json.dumps(config,indent=2))
    attempted=0
    for shard in manifest['shards']:
        with tarfile.open(source/shard['file']) as tar:
            for member in tar:
                if not member.name.endswith('.contours.npz'):continue
                prefix=member.name[:-len('.contours.npz')];meta=json.load(tar.extractfile(prefix+'.json'))
                with np.load(BytesIO(tar.extractfile(member).read()),allow_pickle=False) as a:c={k:a[k] for k in a.files}
                image=np.array(Image.open(BytesIO(tar.extractfile(prefix+'.image.png').read())))
                attempted+=1
                try:
                    raw=transfer(c,extract(image));p=prepare(raw,tokens);t=targets(meta,raw,p)
                    reliable=np.asarray(p.overlap@raw['edge_valid'].astype(float)).ravel()>.99
                    weights=p.weights*reliable
                    ident='bridge-'+meta['sample_id'];file=ident+'.npz'
                    np.savez_compressed(output/file,x=p.x,offsets=p.offsets,weights=weights,**t)
                    np.savez_compressed(output/'sources'/(ident+'.contours.npz'),**raw)
                    meta={**meta,'sample_id':ident,'units':'px'};(output/'sources'/(ident+'.json')).write_text(json.dumps(meta))
                    index.append({'id':ident,'file':file,'split':meta['split'],'family':meta['family'],'design_id':meta['design_id'],'tokens':len(p.x),'instances':len(t['instance_classes']),'units':'px'})
                except ValueError as error:rejected.append({'id':meta['sample_id'],'reason':str(error)})
                if attempted%50==0:print(f'Bridge: {len(index)} accepted / {attempted} attempted',flush=True)
                if attempted>=limit:break
        if attempted>=limit:break
    (output/'index.json').write_text(json.dumps(index,indent=2));(output/'rejections.json').write_text(json.dumps(rejected,indent=2))
    return {'accepted':len(index),'attempted':attempted,'rejected':len(rejected)}
