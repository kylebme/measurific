"""Same-input CPU comparison against the previous release."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from contour_model.train import load_model
from contour_model.inference import predict
from contour_model.sampling import prepare_adaptive


def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('previous');p.add_argument('cache');p.add_argument('output')
    args=p.parse_args();torch.set_num_threads(4);cache=Path(args.cache)
    rows=[e for e in json.loads((cache/'index.json').read_text()) if e['split']=='val']
    selected=[min(rows,key=lambda e:abs(e['tokens']-n)) for n in (768,1500,3000,6000,12000,24000,36000)]
    models={name:load_model(path)[0] for name,path in [('previous',args.previous),('primitive',args.checkpoint)]}
    report={'threads':4,'device':'cpu','samples':[],
            'timing_policy':'warm inference; five forward repeats and one full prediction per sample',
            'checkpoint_sha256':{name:hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                 for name,path in [('previous',args.previous),('primitive',args.checkpoint)]},
            'parameters':{name:sum(p.numel() for p in m.parameters()) for name,m in models.items()}}
    for e in selected:
        path=cache/e['file']
        with np.load(path.parent/'sources'/(e['id']+'.contours.npz')) as a:
            c={k:a[k] for k in ('actual_xy_mm','ring_offsets')}
        prepared=prepare_adaptive(c);x=torch.from_numpy(prepared.x)
        row=dict(id=e['id'],family=e['family'],tokens=e['tokens'],rings=e['rings'])
        for name,m in models.items():
            with torch.inference_mode():
                predict(m,c,instances=True)
                m(x,prepared.offsets,instances=True)
                times=[]
                for _ in range(5):
                    start=time.perf_counter();m(x,prepared.offsets,instances=True);times.append(time.perf_counter()-start)
                start=time.perf_counter();result,_=predict(m,c,instances=True);total=time.perf_counter()-start
            row[name]={'forward_median_ms':float(np.median(times)*1000),'end_to_end_ms':total*1000,
                       'base_or_legacy_features':len(result['features']),'compound_features':len(result.get('compound_features',[]))}
        report['samples'].append(row)
    report['median_forward_ms']={name:float(np.median([r[name]['forward_median_ms'] for r in report['samples']])) for name in models}
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
