"""Expanded, mostly mild-form-error corpus with explicit complexity/stress holdouts."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing,hashlib,json,os,argparse
import numpy as np
from inspection_shapes import geometry
from inspection_shapes.expanded import expanded_nominal,NEW_FAMILIES
from inspection_shapes.render import feature_bits
from .corpus import FAMILIES as LEGACY,composed
from .data import targets,LABEL_VERSION,INPUT_DIM
from .sampling import prepare_adaptive,VERSION

FAMILIES=LEGACY+list(NEW_FAMILIES)

class DefectBudget:
    def __init__(self,rng,rings,count):
        self.rng=rng;self.i=0;self.hits=np.bincount(rng.integers(rings,size=count),minlength=rings)
    def poisson(self,*args,**kwargs):
        n=int(self.hits[self.i]);self.i+=1;return n
    def __getattr__(self,name):return getattr(self.rng,name)


def create_record(index,seed=9237):
    rng=np.random.default_rng(np.random.SeedSequence([seed,index,12345]));family=FAMILIES[index%len(FAMILIES)]
    h=int.from_bytes(hashlib.blake2b(f'{seed}:{index}:split'.encode(),digest_size=8).digest(),'big')%100
    split='train' if h<80 else 'val' if h<88 else 'test' if h<96 else 'ood'
    # Stress is an independently reported test stratum, not 20% of ordinary training.
    stress=(split=='ood' and index%2==0) or (split=='train' and rng.random()<.03)
    severity=float(rng.uniform(.8,2.) if stress else rng.choice([0.,.03,.08,.16,.30],p=[.05,.20,.40,.25,.10]))
    complexity='extreme' if split=='ood' and not stress else 'ordinary'
    for attempt in range(50):
        if family in LEGACY and (family.startswith('mixed_') or family=='perforated_disk' or rng.random()<.45):p=composed(family,rng)
        else:p=expanded_nominal(family,rng,complexity)
        if not geometry.valid(p,False):continue
        extent=max(np.ptp(np.concatenate([r.nominal for r in p.rings]),axis=0))
        effective=severity*min(1.,80/extent)
        defects=int(rng.poisson(2. if stress else .35)) if effective else 0
        geometry.imperfect(p,DefectBudget(rng,len(p.rings),defects),effective)
        if geometry.valid(p):break
    else:raise ValueError(f'Generation failed: {family}, {index}')
    c={'actual_xy_mm':np.concatenate([r.actual for r in p.rings]),'nominal_xy_mm':np.concatenate([r.nominal for r in p.rings]),
       'ring_offsets':np.r_[0,np.cumsum([len(r.actual) for r in p.rings])].astype(np.int64),
       'edge_feature_id':np.concatenate([r.feature for r in p.rings]),
       'edge_primitive_id':np.concatenate([r.primitive for r in p.rings]),
       'edge_anomaly_class':np.concatenate([r.defect for r in p.rings])}
    c['edge_semantic_bits']=feature_bits(p)[c['edge_feature_id']]
    delta=np.linalg.norm(c['actual_xy_mm']-c['nominal_xy_mm'],axis=1)
    meta={'sample_id':f'{index:09d}','design_id':index,'corpus_id':f'expanded-{seed}','family':family,'split':split,
          'features':p.features,'primitives':p.primitives,'patterns':p.patterns,'relationships':p.relationships,'seed':seed,'severity':severity,
          'effective_severity':effective,'stress':stress,'complexity':complexity,'attempts':attempt,
          'form_displacement_mm':{'p95':float(np.quantile(delta,.95)),'max':float(delta.max())},
          'anomalies':p.anomalies,'imperfections':p.imperfections}
    return meta,c


def write_one(job):
    index,seed,root=job;root=Path(root);ident=f'{index:09d}';directory=root/'shards'/f'{index//500:05d}'
    receipt=directory/(ident+'.record.json');target=directory/(ident+'.npz')
    if receipt.exists() and target.exists():return json.loads(receipt.read_text())
    source=directory/'sources';source.mkdir(parents=True,exist_ok=True)
    m,c=create_record(index,seed);p=prepare_adaptive(c);t=targets(m,c,p)
    np.savez_compressed(source/(ident+'.contours.npz'),**c)
    (source/(ident+'.json')).write_text(json.dumps(m))
    temp=directory/(ident+'.partial.npz');np.savez_compressed(temp,x=p.x,offsets=p.offsets,weights=p.weights,**t);temp.replace(target)
    row={'id':ident,'file':str(target.relative_to(root)),'family':m['family'],'split':m['split'],'design_id':index,
         'tokens':len(p.x),'instances':len(t['instance_classes']),'rings':len(p.offsets)-1,
         'stress':m['stress'],'complexity':m['complexity'],'severity':m['severity'],'form_p95_mm':m['form_displacement_mm']['p95']}
    receipt.write_text(json.dumps(row));return row


def build(output,count=50000,seed=9237,workers=6):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    from inspection_shapes import expanded
    from . import data,sampling
    fingerprint=hashlib.sha256(b''.join(Path(x.__file__).read_bytes() for x in [expanded,geometry,data,sampling])+Path(__file__).read_bytes()).hexdigest()
    config={'source':'expanded-v2','seed':seed,'source_hash':fingerprint,'sampling':VERSION,'label_version':LABEL_VERSION,'input_dim':INPUT_DIM,'tokens':'adaptive'}
    path=root/'config.json'
    if path.exists() and json.loads(path.read_text())!=config:raise ValueError('Configuration mismatch')
    lock=root/'.corpus.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        path.write_text(json.dumps(config,indent=2));rows=[]
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for i,row in enumerate(pool.map(write_one,((i,seed,str(root)) for i in range(count)),chunksize=2)):
                rows.append(row)
                if (i+1)%250==0:print(f'Generated {i+1}/{count}',flush=True)
        temp=root/'index.json.partial';temp.write_text(json.dumps(rows,indent=2));temp.replace(root/'index.json')
    finally:os.close(fd);lock.unlink(missing_ok=True)
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--count',type=int,default=50000);p.add_argument('--seed',type=int,default=9237);p.add_argument('--workers',type=int,default=6)
    build(**vars(p.parse_args()))
