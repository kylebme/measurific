"""CPU-only compositional contour corpus; no images or torch imports."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import hashlib
import json
import os
import numpy as np
from shapely.geometry import Polygon, Point
from inspection_shapes.geometry import Part,Chain,make_nominal,imperfect,valid,circle,slot,plate_outline,repeat_profile
from inspection_shapes.render import feature_bits
from inspection_shapes.taxonomy import PARTS
from .data import prepare,targets,LABEL_VERSION,INPUT_DIM

FAMILIES=list(PARTS)+['mixed_shaft','mixed_plate','perforated_disk']


def mixed_shaft(rng,holdout=False):
    p=Part('mixed_shaft');L=rng.uniform(38,76);r=rng.uniform(4,9)
    root=p.feature('external_profile');thread=p.feature('thread',root)
    depth=rng.uniform(.4,1.1);pitch=rng.uniform(1.0,2.2);n=int(rng.integers(7,15));tlength=n*pitch
    L=max(L,tlength*2.5);x0=-L/2
    profile=[[x0,r-.5],[x0+.5,r],[x0+L*.18,r],[x0+L*.18,r*.8],
             [x0+L*.27,r*.8],[x0+L*.28,r*.8-depth],[x0+L*.31,r*.8-depth],[x0+L*.32,r*.8],
             [x0+L*.43,r*.8],[x0+L*.45,r*.8-depth]]
    roles=['chamfer','flat','shoulder','flat','groove','groove','groove','flat','taper']
    style=str(rng.choice(['v','trapezoid','square']))
    teeth=repeat_profile(n,pitch,depth,style,rng);teeth[:,0]+=x0+L*.45;teeth[:,1]+=r*.8-depth
    profile.extend(teeth[1:]);roles.extend(['thread']*(len(teeth)-1))
    profile.append([L/2-.5,r*.8-depth]);roles.append('flat')
    profile.append([L/2,r*.8-depth-.4]);roles.append('chamfer')
    q=np.asarray(profile);c=Chain(p);grooves={}
    p.features[thread-1]['nominal_parameters']={'pitch':pitch,'count':n,'style':style,'domain':'linear'}
    for side in (1,-1):
        indices=range(len(roles)-1,-1,-1) if side==1 else range(len(roles))
        for i in indices:
            role=roles[i]
            if role=='thread':fid=thread
            elif role=='groove':
                if side not in grooves:grooves[side]=p.feature('groove',root)
                fid=grooves[side]
            else:fid=p.feature(role,root)
            a=q[i]*[1,side];b=q[i+1]*[1,side]
            if side==1:a,b=b,a
            c.line(a,b,fid)
        a=q[0] if side==1 else q[-1]*[1,-1]
        b=q[0]*[1,-1] if side==1 else q[-1]
        c.line(a,b,p.feature('flat',root))
    c.close()
    if holdout:circle(p,[x0+L*.1,0],r*.22,'hole',True)
    return p


def add_openings(part,rng,holdout=False):
    outer=Polygon(part.rings[0].nominal);bounds=outer.bounds
    w=bounds[2]-bounds[0];h=bounds[3]-bounds[1];scale=min(w,h)
    for i in range(int(rng.integers(1,6))):
        for attempt in range(20):
            x=rng.uniform(bounds[0]+w*.2,bounds[2]-w*.2);y=rng.uniform(bounds[1]+h*.2,bounds[3]-h*.2)
            rad=scale*rng.uniform(.025,.07)
            length=rad*rng.uniform(3,7);angle=rng.uniform(-np.pi,np.pi)
            use_slot=rng.random()<(.8 if holdout else .45)
            probe=Part('probe')
            if use_slot:slot(probe,[x,y],length,rad*2,angle)
            else:circle(probe,[x,y],rad,'hole',True)
            poly=Polygon(probe.rings[0].nominal)
            if not outer.buffer(-scale*.035).contains(poly):continue
            if any(poly.buffer(scale*.045).intersects(Polygon(r.nominal)) for r in part.rings[1:]):continue
            if use_slot:slot(part,[x,y],length,rad*2,angle)
            else:circle(part,[x,y],rad,'hole',True)
            break


def composed(family,rng,holdout=False):
    if family=='mixed_shaft':return mixed_shaft(rng,holdout)
    if family=='mixed_plate':
        p=Part(family);plate_outline(p,rng.uniform(28,72),rng.uniform(20,48),rng.uniform(.5,4),rng.random()<.5)
        add_openings(p,rng,holdout);return p
    if family=='perforated_disk':
        p=Part(family);circle(p,[0,0],rng.uniform(12,28),'external_profile');add_openings(p,rng,holdout);return p
    return make_nominal(family,rng)


def create_record(index,seed):
    split_hash=int.from_bytes(hashlib.blake2b(f'{seed}:{index}:split'.encode(),digest_size=8).digest(),'big')%100
    split='train' if split_hash<80 else 'val' if split_hash<88 else 'test' if split_hash<96 else 'ood'
    rng=np.random.default_rng(np.random.SeedSequence([seed,index,919]))
    family=FAMILIES[index%len(FAMILIES)] if split!='ood' else ['mixed_shaft','mixed_plate','perforated_disk'][index%3]
    severity=float(rng.choice([0.,.4,1.,2.5],p=[.08,.27,.45,.20]))
    for attempt in range(50):
        p=composed(family,rng,split=='ood')
        if not valid(p,False):continue
        imperfect(p,rng,severity)
        if valid(p):break
    else:raise ValueError(f'Failed composition {family}, seed {seed}, design {index}')
    c={'actual_xy_mm':np.concatenate([r.actual for r in p.rings]),
       'ring_offsets':np.r_[0,np.cumsum([len(r.actual) for r in p.rings])].astype(np.int64),
       'edge_feature_id':np.concatenate([r.feature for r in p.rings]),
       'edge_anomaly_class':np.concatenate([r.defect for r in p.rings])}
    c['edge_semantic_bits']=feature_bits(p)[c['edge_feature_id']]
    meta={'schema_version':1,'sample_id':f'{index:09d}','design_id':index,'split':split,'family':family,
          'corpus_id':f'composed-{seed}','features':p.features,'seed':seed,'severity':severity,'rejected':attempt}
    return meta,c


def _write_one(job):
    index,seed,output,tokens,legacy_count=job;output=Path(output);ident=f'{index:09d}'
    directory=output if index<legacy_count else output/'shards'/f'{index//1000:05d}'
    target=directory/(ident+'.npz');receipt=directory/(ident+'.record.json')
    if target.exists() and receipt.exists():return json.loads(receipt.read_text())
    directory.mkdir(parents=True,exist_ok=True);sources=directory/'sources';sources.mkdir(exist_ok=True)
    meta,c=create_record(index,seed);prepared=prepare(c,tokens);t=targets(meta,c,prepared)
    np.savez_compressed(sources/(ident+'.contours.npz'),**c)
    (sources/(ident+'.json')).write_text(json.dumps(meta))
    temp=directory/(ident+'.partial.npz');np.savez_compressed(temp,x=prepared.x,offsets=prepared.offsets,weights=prepared.weights,**t);temp.replace(target)
    row={'id':ident,'file':str(target.relative_to(output)),'split':meta['split'],'family':meta['family'],'design_id':index,
         'tokens':len(prepared.x),'instances':len(t['instance_classes'])}
    receipt.write_text(json.dumps(row));return row


def build_corpus(output,count=50000,seed=8173,tokens=1024,workers=4,migrate_layout=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256(Path(__file__).read_bytes()+Path(__file__).with_name('data.py').read_bytes()).hexdigest()
    path=output/'config.json';old=json.loads(path.read_text()) if path.exists() else {}
    previous=old.get('previous_source_hash');legacy_count=old.get('legacy_count',0)
    if migrate_layout:
        if old.get('layout')=='sharded-v2':raise ValueError('Layout already migrated')
        if old.get('seed')!=seed or old.get('tokens')!=tokens:raise ValueError('Migration seed/token mismatch')
        receipts=sorted(output.glob('*.record.json'))
        if not receipts:raise ValueError('No completed legacy records')
        # Prove that layout changes have not changed geometry/labels before resuming.
        for receipt in [receipts[0],receipts[len(receipts)//2],receipts[-1]]:
            index=int(receipt.name.split('.')[0]);meta,c=create_record(index,seed)
            with np.load(output/'sources'/f'{index:09d}.contours.npz',allow_pickle=False) as saved:
                for key in c:np.testing.assert_array_equal(c[key],saved[key])
        legacy_count=max(int(p.name.split('.')[0]) for p in receipts)+1
        previous=old['source_hash']
        print(f'Verified legacy generation; preserving {len(receipts)} committed records',flush=True)
    config={'label_version':LABEL_VERSION,'tokens':tokens,'seed':seed,'input_dim':INPUT_DIM,'source':'compositional',
            'source_hash':fingerprint,'layout':'sharded-v2','legacy_count':legacy_count,'previous_source_hash':previous}
    if old and not migrate_layout and old!=config:raise ValueError('Corpus configuration mismatch')
    lock=output/'.corpus.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode())
    try:
        path.write_text(json.dumps(config,indent=2));rows=[]
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for first in range(0,count,workers*8):
                jobs=[(i,seed,str(output),tokens,legacy_count) for i in range(first,min(count,first+workers*8))]
                rows.extend(pool.map(_write_one,jobs,chunksize=2))
                if first//1000!=(first+len(jobs))//1000:print(f'Generated {len(rows)}/{count} contour records',flush=True)
        temp=output/'index.json.partial';temp.write_text(json.dumps(rows,indent=2));temp.replace(output/'index.json')
        print(f'Corpus complete: {count} records',flush=True)
        return rows
    finally:
        os.close(fd);lock.unlink(missing_ok=True)
