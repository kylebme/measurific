"""Label-blind ring resampling, v1 training-label adapter, compact data caches."""
from dataclasses import dataclass
from pathlib import Path
from io import BytesIO
import hashlib
import json
import tarfile
import numpy as np
from scipy.sparse import csr_matrix
from shapely.geometry import Polygon, Point
from shapely.strtree import STRtree

LABEL_VERSION = "observed-compounds-v1"
NAMES = ["unlabeled", "external_profile", "hole", "slot", "flat", "chamfer", "fillet",
         "shoulder", "groove", "taper", "thread", "tooth_profile", "polygon",
         "rounded_profile", "cam_profile", "tab", "notch"]
INPUT_DIM = 12

@dataclass
class Prepared:
    x: np.ndarray
    offsets: np.ndarray
    weights: np.ndarray
    overlap: csr_matrix
    center: np.ndarray
    scale: float
    hole: np.ndarray
    xy: np.ndarray


def ring_geometry(contours):
    xy=np.asarray(contours['actual_xy_mm'],dtype=np.float64)
    offsets=np.asarray(contours['ring_offsets'],dtype=np.int64)
    if xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all():raise ValueError('Expected finite Nx2 coordinates')
    if len(offsets)<2 or offsets[0]!=0 or offsets[-1]!=len(xy) or (np.diff(offsets)<3).any():
        raise ValueError('Invalid ring offsets')
    rings=[xy[a:b] for a,b in zip(offsets[:-1],offsets[1:])]
    polys=[Polygon(r) for r in rings]
    if any(not p.is_valid or p.area<1e-12 for p in polys):raise ValueError('Invalid closed contour')
    boundaries=[p.boundary for p in polys]
    pairs=STRtree(boundaries).query(boundaries,predicate='intersects')
    if np.any(pairs[0]!=pairs[1]):raise ValueError('Touching/intersecting rings are unsupported')
    outer=int(np.argmax([p.area for p in polys]));hole=np.arange(len(rings))!=outer
    if not all(polys[outer].contains(polys[i]) for i in range(len(polys)) if i!=outer):
        raise ValueError('Expected one part with non-nested through holes')
    if not Polygon(rings[outer],[r for i,r in enumerate(rings) if i!=outer]).is_valid:
        raise ValueError('Touching/intersecting or nested rings are unsupported')
    return xy,offsets,hole


def encode_geometry(mid, offsets, hole, lengths):
    center=np.average(mid,axis=0,weights=lengths)
    scale=float(np.linalg.norm(mid-center,axis=1).max())
    if scale<1e-12:raise ValueError('Degenerate part')
    normalized=(mid-center)/scale
    out=[]
    for i,(lo,hi) in enumerate(zip(offsets[:-1],offsets[1:])):
        q=normalized[lo:hi]
        d=np.roll(q,-1,axis=0)-np.roll(q,1,axis=0)
        d/=np.maximum(np.linalg.norm(d,axis=1,keepdims=True),1e-8)
        curv=[]
        for lag in (1,4,16):
            lag=min(lag,max(1,len(q)//8))
            before=q-np.roll(q,lag,axis=0);after=np.roll(q,-lag,axis=0)-q
            den=np.maximum(np.linalg.norm(before,axis=1)*np.linalg.norm(after,axis=1),1e-8)
            curv.extend([(before[:,0]*after[:,1]-before[:,1]*after[:,0])/den,(before*after).sum(axis=1)/den])
        out.append(np.c_[q,d,lengths[lo:hi]/scale,np.full(len(q),float(hole[i])),np.array(curv).T])
    return np.concatenate(out).astype(np.float32),center,scale


def prepare(contours, tokens=2048, min_ring=32, ring_counts=None, ring_grids=None):
    """Input geometry alone controls every sample location; overlap transfers targets later."""
    xy,raw_offsets,hole=ring_geometry(contours)
    if ring_counts is None and tokens<len(hole)*min_ring:raise ValueError('Token budget cannot retain all rings')
    lengths=[];rings=[];maps=[]
    for i,(a,b) in enumerate(zip(raw_offsets[:-1],raw_offsets[1:])):
        q=xy[a:b].copy();edge_map=np.arange(a,b)
        area=np.sum(q[:,0]*np.roll(q[:,1],-1)-q[:,1]*np.roll(q[:,0],-1))/2
        if (area<0)!=hole[i]:
            q=q[::-1].copy();edge_map=np.roll(edge_map[::-1],-1)
        seg=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
        if (seg<=1e-10).any():raise ValueError('Duplicate consecutive contour points')
        lengths.append(seg);rings.append(q);maps.append(edge_map)
    perimeters=np.array([v.sum() for v in lengths])
    # Budget is rounded up to complete local windows. Small rings remain represented.
    counts=np.maximum(min_ring,np.ceil(tokens*perimeters/perimeters.sum()/32).astype(int)*32)
    if ring_counts is not None:
        counts=np.asarray(ring_counts,dtype=int)
        if counts.shape!=(len(hole),) or (counts<min_ring).any():raise ValueError('Invalid per-ring token counts')
    mids=[];weight=[];row=[];col=[];val=[];start=0;offsets=[0]
    for ri,(q,leng,edge_map,n) in enumerate(zip(rings,lengths,maps,counts)):
        s=np.r_[0,np.cumsum(leng)];step=s[-1]/n;target=np.arange(n+1)*step
        if ring_grids is not None:
            target=np.asarray(ring_grids[ri],float)
            if len(target)!=n+1 or target[0]!=0 or not np.isclose(target[-1],s[-1]) or (np.diff(target)<=0).any():
                raise ValueError('Invalid adaptive arc-length grid')
        steps=np.diff(target)
        centers=(target[:-1]+target[1:])/2
        closed=np.vstack([q,q[0]])
        mids.append(np.c_[np.interp(centers,s,closed[:,0]),np.interp(centers,s,closed[:,1])])
        weight.extend(steps)
        for j in range(n):
            first=max(0,np.searchsorted(s,target[j],side='right')-1)
            last=min(len(q)-1,np.searchsorted(s,target[j+1],side='left'))
            for k in range(first,last+1):
                amount=min(s[k+1],target[j+1])-max(s[k],target[j])
                if amount>1e-12:row.append(start+j);col.append(edge_map[k]);val.append(amount/steps[j])
        start+=n;offsets.append(start)
    mids=np.concatenate(mids);offsets=np.asarray(offsets,np.int64);weight=np.asarray(weight,np.float32)
    x,center,scale=encode_geometry(mids,offsets,hole,weight)
    overlap=csr_matrix((val,(row,col)),shape=(len(x),len(xy)))
    return Prepared(x,offsets,weight,overlap,center,scale,hole,mids)


def adapted_labels(metadata, contours):
    """Compound instances for contiguous tabs/notches; observed grooves stay separate.

Do not infer across-ring grouping from a part-family label. Threads preserve their
explicit disconnected original instance membership. Outer-profile ancestry is
uniform, independent of which root semantic the generator happened to emit.
"""
    ids=contours['edge_feature_id'].astype(np.int64).copy()
    classes=np.array([0]+[f['semantic_id'] for f in metadata['features']],np.int64)
    primary=classes[ids]
    for lo,hi in zip(contours['ring_offsets'][:-1],contours['ring_offsets'][1:]):
        for semantic in (15,16):
            mask=primary[lo:hi]==semantic
            starts=np.flatnonzero(mask & ~np.roll(mask,1))
            for start in starts:
                run=[];j=int(start)
                while mask[j]:
                    run.append(lo+j);j=(j+1)%len(mask)
                    if j==start:break
                ids[run]=int(ids[run].min())
    bits=contours['edge_semantic_bits'].astype(np.uint32).copy()
    _,offsets,hole=ring_geometry(contours)
    for h,lo,hi in zip(hole,offsets[:-1],offsets[1:]):
        if not h:bits[lo:hi]|=np.uint32(1<<1)
    semantic=((bits[:,None]>>np.arange(1,17))&1).astype(np.float32)
    return ids,primary,semantic


def targets(metadata, contours, prepared):
    ids,primary,semantic=adapted_labels(metadata,contours)
    W=prepared.overlap
    result={'semantic':np.asarray(W@semantic,dtype=np.float32),
            'primary':np.asarray(W@np.eye(17,dtype=np.float32)[primary],dtype=np.float32)}
    instance_ids=np.unique(ids);masks=np.asarray(W@(ids[:,None]==instance_ids[None,:]),dtype=np.float32).T
    iclasses=np.array([primary[np.flatnonzero(ids==i)[0]] for i in instance_ids],np.int64)
    result['instance_masks']=masks;result['instance_classes']=iclasses
    boundary=np.zeros(W.shape[0],np.float32)
    # Feature transitions at source vertices, transferred to token endpoints by arc length.
    for lo,hi in zip(contours['ring_offsets'][:-1],contours['ring_offsets'][1:]):
        change=np.flatnonzero(ids[lo:hi]!=np.roll(ids[lo:hi],1))+lo
        for edge in change:
            hit=W[:,edge].nonzero()[0]
            if len(hit):boundary[hit[0]]=1
    # One-token tolerance, no wrap between distinct rings.
    for lo,hi in zip(prepared.offsets[:-1],prepared.offsets[1:]):
        b=boundary[lo:hi].copy();boundary[lo:hi]=np.maximum(b,.5*np.maximum(np.roll(b,1),np.roll(b,-1)))
    result['boundary']=boundary
    defect=contours.get('edge_anomaly_class',np.zeros(len(ids),np.uint8))
    result['defect']=np.asarray(W@np.eye(3,dtype=np.float32)[defect],dtype=np.float32)
    return result


def records(source):
    """Read only JSON/contours, never grayscale or image labels. Supports tar or unpacked datasets."""
    source=Path(source)
    if (source/'manifest.json').exists():
        manifest=json.loads((source/'manifest.json').read_text())
        for shard in manifest['shards']:
            with tarfile.open(source/shard['file']) as tar:
                for member in tar:
                    if member.name.endswith('.contours.npz'):
                        prefix=member.name[:-len('.contours.npz')]
                        meta=json.load(tar.extractfile(prefix+'.json'))
                        with np.load(BytesIO(tar.extractfile(member).read()),allow_pickle=False) as a:
                            yield meta,{k:a[k] for k in a.files}
    else:
        for path in sorted(source.rglob('*.contours.npz')):
            meta=json.loads(path.with_name(path.name.replace('.contours.npz','.json')).read_text())
            with np.load(path,allow_pickle=False) as a:yield meta,{k:a[k] for k in a.files}


def build_cache(source,output,tokens=2048,limit=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    config={'label_version':LABEL_VERSION,'tokens':tokens,'source':str(Path(source).resolve()),'input_dim':INPUT_DIM}
    path=output/'config.json'
    if path.exists() and json.loads(path.read_text())!=config:raise ValueError('Cache config mismatch')
    path.write_text(json.dumps(config,indent=2))
    index=[];design_splits={}
    for meta,c in records(source):
        key=(str(meta.get('corpus_id','default')),int(meta['design_id']))
        if design_splits.setdefault(key,meta['split'])!=meta['split']:raise ValueError('Design leaks across splits')
        ident=str(meta['sample_id']);file=output/(ident+'.npz')
        if any(r['id']==ident for r in index):raise ValueError('Duplicate sample ID')
        p=prepare(c,tokens);t=targets(meta,c,p)
        # Original/source labels are not model inputs. Raw contours remain in the source dataset.
        np.savez_compressed(file,x=p.x,offsets=p.offsets,weights=p.weights,**t)
        index.append({'id':ident,'file':file.name,'split':meta['split'],'family':meta['family'],
                      'design_id':meta['design_id'],'tokens':len(p.x),'instances':len(t['instance_classes'])})
        if len(index)%100==0:print('Prepared',len(index),flush=True)
        if limit and len(index)>=limit:break
    (output/'index.json').write_text(json.dumps(index,indent=2))
    print(f'Cached {len(index)} contours at {output}',flush=True)
    return index


_DATA_ROOTS=('/Volumes/T7 Shield','/media/T7Shield','/media/kyle/T7 Shield')


def remap_path(path):
    """Caches record absolute paths from the machine that built them (macOS or Linux)."""
    text=str(path)
    for root in _DATA_ROOTS:
        if text.startswith(root):
            for candidate in _DATA_ROOTS:
                moved=Path(candidate+text[len(root):])
                if moved.exists():return moved
    return Path(path)


def load_cached(path):
    with np.load(remap_path(path),allow_pickle=False) as a:return {k:a[k] for k in a.files}


TOKEN_KEYS=('x','weights','semantic','primary','boundary','defect','compound')


def mirror(sample):
    """Reflect a prepared sample (x -> -x), keeping canonical ring winding.

    Reflection reverses traversal, so each ring's tokens are reversed. A boundary
    target marks the start of a token; after reversal it marks the token's end,
    which is the start of the following token.
    """
    out={k:v.copy() for k,v in sample.items()};offsets=out['offsets']
    order=np.concatenate([np.arange(hi-1,lo-1,-1) for lo,hi in zip(offsets[:-1],offsets[1:])])
    for key in TOKEN_KEYS:
        if key in out:out[key]=out[key][order]
    if 'instance_masks' in out:out['instance_masks']=out['instance_masks'][:,order]
    for lo,hi in zip(offsets[:-1],offsets[1:]):out['boundary'][lo:hi]=np.roll(out['boundary'][lo:hi],1)
    x=out['x'];coords=x[:,:2].copy();coords[:,0]*=-1
    hole=np.array([bool(x[lo,5]) for lo in offsets[:-1]])
    out['x']=encode_geometry(coords,offsets,hole,x[:,4])[0]
    return out


def augment(sample,rng,noise=0.0005,rotation=True,relative_noise=False,reflect=False):
    """Observation augmentation after label-blind resampling; labels follow cyclic shifts."""
    if reflect and rng.random()<.5:sample=mirror(sample)
    out={k:v.copy() for k,v in sample.items()}
    x=out['x'];coords=x[:,:2].copy();offsets=out['offsets']
    if rotation:
        t=rng.uniform(-np.pi,np.pi);R=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
        coords=coords@R.T
    if noise:
        from scipy.ndimage import gaussian_filter1d
        for lo,hi in zip(offsets[:-1],offsets[1:]):
            sigma=min(noise,.08*float(np.median(x[lo:hi,4]))) if relative_noise else noise
            coords[lo:hi]+=gaussian_filter1d(rng.normal(0,sigma,(hi-lo,2)),1,axis=0,mode='wrap')
    hole=np.array([bool(x[lo,5]) for lo in offsets[:-1]])
    out['x']=encode_geometry(coords,offsets,hole,x[:,4])[0]
    permutation=[]
    order=rng.permutation(len(hole))
    for i in order:
        lo,hi=offsets[i:i+2];permutation.extend(np.roll(np.arange(lo,hi),int(rng.integers(hi-lo))))
    permutation=np.array(permutation);out['offsets']=np.r_[0,np.cumsum(np.diff(offsets)[order])]
    for key in ('x','weights','semantic','primary','boundary','defect'):out[key]=out[key][permutation]
    out['instance_masks']=out['instance_masks'][:,permutation]
    return out
