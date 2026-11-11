"""Finite generation + matched original-only/mixed PAE training experiment."""
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
RUN=DISK/'experiments-pae-v1-20260921'
SOURCE=DISK/'primitives-context-pilot-v1-20260920'
PAE=DISK/'primitives-pae-v1-20260921'
INITIAL=DISK/'experiments-context-v1-20260920/full-context/best.pt'


def status(stage,**extra):
    value=dict(stage=stage,pid=os.getpid(),updated=time.time(),**extra)
    temp=RUN/'status.tmp';temp.write_text(json.dumps(value,indent=2));temp.replace(RUN/'status.json')
    print(json.dumps(value),flush=True)


def cache(name,entries):
    path=RUN/name;path.mkdir(exist_ok=False)
    config=json.loads((PAE/'config.json').read_text());config.update(source=name)
    (path/'config.json').write_text(json.dumps(config,indent=2));(path/'index.json').write_text(json.dumps(entries,indent=2))
    return path


def pipeline():
    import numpy as np
    import torch
    from contour_model.pae_corpus import build
    from contour_model.train import load_model
    from contour_model.train_primitives import evaluate
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():raise RuntimeError('MPS required')
    status('building_cache');rows=build(SOURCE,PAE,limit=2400,workers=2)
    originals={e['id']:e for e in json.loads((SOURCE/'index.json').read_text())}
    original=[];extracted=[]
    for row in rows:
        entry=originals[row['parent_id']]
        assert row['design_id']==entry['design_id'] and row['split']==entry['split']
        original.append(dict(entry,file=str((SOURCE/entry['file']).resolve())))
        extracted.append(dict(row,file=str((PAE/row['file']).resolve())))
    counts={split:sum(e['split']==split for e in original) for split in ('train','val','test')}
    if counts['train']<300 or counts['val']<40 or counts['test']<40:
        raise RuntimeError(f'Insufficient accepted designs for comparison: {counts}')
    # Both variants see the same designs. Validation is identical and contains
    # both observation domains; only mixed training includes extracted variants.
    control=cache('original-control-cache',original+[e for e in extracted if e['split']!='train'])
    mixed=cache('mixed-cache',original+extracted)
    original_eval=cache('original-eval-cache',original)
    status('cache_ready',accepted=len(rows),split_counts=counts)
    for name,data in [('original-control',control),('mixed-pae',mixed)]:
        status('training',variant=name)
        subprocess.run([sys.executable,'-u','-m','contour_model.train_primitives',str(data),str(RUN/name),
            '--steps','4000','--seed','91','--eval-every','500','--val-limit','200','--device','mps',
            '--init',str(INITIAL),'--direct-global-context','--full-context','--pooled-tokens','256',
            '--dropout','.1','--form-error','.003','--boundary-positive-weight','4','--boundary-weight','.3',
            '--query-weight','1.5','--lr','.00005'],cwd=ROOT,check=True)
    comparison={}
    for name,path in [('initial-full-context',INITIAL),('original-control',RUN/'original-control/best.pt'),('mixed-pae',RUN/'mixed-pae/best.pt')]:
        status('evaluating',variant=name);model,_=load_model(path,'mps');comparison[name]={}
        for domain,data,items in [('original',original_eval,original),('pae',PAE,extracted)]:
            test=[e for e in items if e['split']=='test']
            # Both lists have identical design order; same shuffle gives matched tests.
            np.random.default_rng(4921).shuffle(test);test=test[:200]
            comparison[name][domain]=evaluate(model,test,data,'mps')
            (RUN/'comparison.json').write_text(json.dumps(comparison,indent=2))
            subprocess.run([sys.executable,'-m','contour_model.validate_primitives',str(path),str(data),
                str(RUN/(name+'-'+domain+'-raw.json')),'--limit','100','--device','mps'],cwd=ROOT,check=True)
        del model;torch.mps.empty_cache()
    status('complete',note='Review paired results and rejection audit. Active GUI checkpoint was not changed.')


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
