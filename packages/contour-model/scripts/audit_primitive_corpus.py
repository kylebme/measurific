"""Audit the delivered cache and whole-base compound membership."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import numpy as np


def check(job):
    root,e,nbase=job;root=Path(root);path=root/e['file']
    with np.load(path) as z:
        n=len(z['x'])
        assert z['primary'].shape==(n,nbase)
        assert z['compound'].shape==(n,5)
        assert np.allclose(z['primary'].sum(1),1,atol=1e-5)
        assert np.allclose(z['compound'].sum(1),1,atol=1e-5)
        assert len(z['instance_classes'])<=32
        base=(z['primary']*z['weights'][:,None]).sum(0)
        compound=(z['compound']*z['weights'][:,None]).sum(0)
    with np.load(path.parent/'sources'/(e['id']+'.contours.npz')) as z:
        ids=z['edge_base_instance'];ci=z['edge_compound_instance'];bc=z['edge_base_class'];cc=z['edge_compound_class']
        pairs=np.unique(np.c_[ids,ci],axis=0)
        assert len(pairs)==len(np.unique(ids)),e['id']
        assert set(np.unique(bc))<=set(range(1,nbase))
        assert set(np.unique(cc))<={0,1,2,3,4}
    if e['family']=='labyrinth_plate':
        metadata=json.loads((path.parent/'sources'/(e['id']+'.json')).read_text())
        assert all(p['kind']!='circular_arc' or p['nominal_parameters']['radius']<=1.000001
                   for p in metadata['primitives']),e['id']
    return base,compound


def main():
    p=argparse.ArgumentParser();p.add_argument('cache');p.add_argument('output');p.add_argument('--workers',type=int,default=6)
    args=p.parse_args();root=Path(args.cache);entries=json.loads((root/'index.json').read_text())
    config=json.loads((root/'config.json').read_text());nbase=len(config['base_names'])
    assert len({e['design_id'] for e in entries})==len(entries)
    assert not any(e['family']=='keyhole_panel' for e in entries)
    base=np.zeros(nbase);compound=np.zeros(5)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,(b,c) in enumerate(pool.map(check,((str(root),e,nbase) for e in entries),chunksize=10)):
            base+=b;compound+=c
            if (i+1)%1000==0:print(f'Audited {i+1}/{len(entries)}',flush=True)
    report=dict(samples=len(entries),families=len({e['family'] for e in entries}),label_version=config['label_version'],
                base_names=config['base_names'],arbitrary_profile_removed='arbitrary_profile' not in config['base_names'],
                split_counts={s:sum(e['split']==s for e in entries) for s in ('train','val','test','ood')},
                base_support_length=base.tolist(),compound_support_length=compound.tolist(),
                whole_base_membership=True,design_split_isolation=True,cache_targets_normalized=True,
                max_tokens=max(e['tokens'] for e in entries),max_rings=max(e['rings'] for e in entries))
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
