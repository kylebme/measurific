"""Finite, detached MacBook pipeline; no agent polling required during training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
DISK=Path('/Volumes/T7 Shield/inspection-contour-training')
RUN=DISK/'experiments-intent-v1-20260920'
CACHE=DISK/'primitives-intent-v1-20260920'


def status(stage,**extra):
    value=dict(stage=stage,pid=os.getpid(),updated=time.time(),**extra)
    temporary=RUN/'status.tmp'; temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(RUN/'status.json')
    print(json.dumps(value),flush=True)


def command(*args):
    subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)


def pipeline():
    from contour_model.recache_intent import build
    from contour_model.train import load_model
    from contour_model.train_primitives import evaluate
    import numpy as np
    import torch
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS is required for this run')
    status('building_cache')
    build(DISK/'primitives-three-base-v2',CACHE,workers=2)
    status('training')
    command('-m','contour_model.train_primitives',CACHE,RUN/'candidate',
        '--steps','12000','--eval-every','500','--val-limit','200','--device','mps',
        '--init',ROOT/'contour-model/release-v3/model.pt','--direct-global-context',
        '--pooled-tokens','256','--dropout','.1','--form-error','.003',
        '--boundary-positive-weight','4','--boundary-weight','.3','--query-weight','1.5','--lr','.0001')
    status('evaluating')
    entries=[e for e in json.loads((CACHE/'index.json').read_text()) if e['split']=='test']
    np.random.default_rng(5721).shuffle(entries); entries=entries[:400]
    comparison={}
    for name,path in [('release-v3',ROOT/'contour-model/release-v3/model.pt'),('candidate',RUN/'candidate/best.pt')]:
        model,_=load_model(path,'mps')
        comparison[name]={probe:evaluate(model,entries,CACHE,'mps',form_error=amount)
                          for probe,amount in [('original',0),('waviness',.003),('strong_waviness',.006)]}
        del model; torch.mps.empty_cache()
        (RUN/'comparison.json').write_text(json.dumps(comparison,indent=2))
        command('-m','contour_model.validate_primitives',path,CACHE,RUN/(name+'-raw-test.json'),
                '--split','test','--limit','200','--device','mps')
    status('complete',note='Candidate is not promoted automatically; inspect comparison and raw instance reports.')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--launch',action='store_true');args=p.parse_args()
    if args.launch:
        RUN.mkdir(exist_ok=False)
        with (RUN/'pipeline.log').open('x') as log:
            process=subprocess.Popen(['/usr/bin/caffeinate','-i',sys.executable,'-u',str(Path(__file__).resolve())],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (RUN/'launcher.json').write_text(json.dumps(dict(pid=process.pid,started=time.time()),indent=2))
        print(f'Launched PID {process.pid}; log: {RUN / "pipeline.log"}')
        return
    try: pipeline()
    except BaseException as exc:
        status('failed',error=str(exc)); traceback.print_exc(); raise


if __name__=='__main__': main()
