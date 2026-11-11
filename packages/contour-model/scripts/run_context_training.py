"""Matched whole-part context pilot; finite detached MPS job."""
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
RUN=DISK/'experiments-context-v1-20260920'
SOURCE=DISK/'primitives-intent-v1-20260920'
CACHE=DISK/'primitives-context-pilot-v1-20260920'


def status(stage,**extra):
    value=dict(stage=stage,pid=os.getpid(),updated=time.time(),**extra)
    temp=RUN/'status.tmp';temp.write_text(json.dumps(value,indent=2));temp.replace(RUN/'status.json')
    print(json.dumps(value),flush=True)


def pipeline():
    import numpy as np
    import torch
    from contour_model.train import load_model
    from contour_model.train_primitives import evaluate
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():raise RuntimeError('MPS required')
    CACHE.mkdir(exist_ok=False)
    all_entries=json.loads((SOURCE/'index.json').read_text())
    entries=[dict(e,file=str((SOURCE/e['file']).resolve())) for e in all_entries if e['tokens']<=8192]
    (CACHE/'index.json').write_text(json.dumps(entries,indent=2))
    config=json.loads((SOURCE/'config.json').read_text());config.update(parent_source=str(SOURCE),
        source='whole-part-context-pilot',max_tokens=8192,excluded_samples=len(all_entries)-len(entries))
    (CACHE/'config.json').write_text(json.dumps(config,indent=2))
    tests=[e for e in entries if e['split']=='test'];np.random.default_rng(5721).shuffle(tests);tests=tests[:200]
    variants=[('control-256',256,False),('context-1024',1024,False),('full-context',256,True)]
    results={}
    baseline=ROOT/'contour-model/release-v4-candidate/model.pt'
    for name,tokens,full in variants:
        status('training',variant=name,retained_samples=len(entries),excluded_samples=len(all_entries)-len(entries))
        args=[sys.executable,'-u','-m','contour_model.train_primitives',str(CACHE),str(RUN/name),
            '--steps','4000','--seed','73','--eval-every','500','--val-limit','100','--device','mps',
            '--init',str(baseline),'--direct-global-context','--pooled-tokens',str(tokens),
            '--dropout','.1','--form-error','.003','--boundary-positive-weight','4',
            '--boundary-weight','.3','--query-weight','1.5','--lr','.00005']
        if full:args.append('--full-context')
        subprocess.run(args,cwd=ROOT,check=True)
    for name,path in [('baseline',baseline)]+[(n,RUN/n/'best.pt') for n,_,_ in variants]:
        status('evaluating',variant=name)
        model,_=load_model(path,'mps');start=time.monotonic()
        results[name]={probe:evaluate(model,tests,CACHE,'mps',form_error=amount)
            for probe,amount in [('original',0),('waviness',.003),('strong_waviness',.006)]}
        results[name]['evaluation_seconds']=time.monotonic()-start
        del model;torch.mps.empty_cache()
        (RUN/'comparison.json').write_text(json.dumps(results,indent=2))
        subprocess.run([sys.executable,'-m','contour_model.validate_primitives',str(path),str(CACHE),
            str(RUN/(name+'-raw-test.json')),'--limit','100','--device','mps'],cwd=ROOT,check=True)
    status('complete',next_stage='Review context comparison, then implement and pilot render-to-PAE label transfer. Do not auto-promote.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');args=p.parse_args()
    if args.launch:
        RUN.mkdir(exist_ok=False)
        with (RUN/'pipeline.log').open('x') as log:
            proc=subprocess.Popen(['/usr/bin/caffeinate','-i',sys.executable,'-u',str(Path(__file__).resolve())],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (RUN/'launcher.json').write_text(json.dumps(dict(pid=proc.pid,started=time.time()),indent=2))
        print(f'Launched PID {proc.pid}; log: {RUN / "pipeline.log"}');return
    try:pipeline()
    except BaseException as exc:
        status('failed',error=str(exc));traceback.print_exc();raise


if __name__=='__main__':main()
