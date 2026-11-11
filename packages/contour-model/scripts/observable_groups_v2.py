"""Export observable flat-surface aliases without modifying frozen contours or caches."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,json,hashlib,multiprocessing
import numpy as np
from contour_model.validation import flat_surface_groups


def one(job):
    source,entry=job;root=Path(source)/Path(entry['file']).parent/'sources';ident=entry['id']
    meta=json.loads((root/(ident+'.json')).read_text())
    if sum(f['semantic_id']==4 for f in meta['features'])<2:return None
    with np.load(root/(ident+'.contours.npz'),allow_pickle=False) as c:
        ids=c['edge_feature_id'];classes=np.array([0]+[f['semantic_id'] for f in meta['features']])[ids]
        groups=flat_surface_groups(ids,classes,c['nominal_xy_mm'],c['ring_offsets'])
    return {'id':ident,'family':entry['family'],'aliases':groups} if groups else None


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('source');p.add_argument('output');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    source=Path(a.source);index=json.loads((source/'index.json').read_text());rows=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for row in pool.map(one,((str(source),e) for e in index),chunksize=32):
            if row:rows.append(row)
    report={'reference_policy':'observable-flat-surfaces-v2','source_index_sha256':hashlib.sha256((source/'index.json').read_bytes()).hexdigest(),
            'definition':'Alias only adjacent same-class flat feature IDs whose nominal tangent continuation is collinear; preserve observed coordinates and all semantic classes.',
            'source_arrays_modified':False,'samples_checked':len(index),'samples_with_aliases':len(rows),'aliases':rows}
    Path(a.output).write_text(json.dumps(report,indent=2));print('Checked',len(index),'parts;',len(rows),'have unobservable flat subdivisions',flush=True)
if __name__=='__main__':main()
