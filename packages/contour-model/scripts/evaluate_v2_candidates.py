"""Evaluate immutable completed-run candidates; overlap CPU handoff with GPU metrics."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,hashlib,json,multiprocessing,shutil,time
import torch
from contour_model.train import load_model,evaluate,validate_cache_sampling
from contour_model.validation import selection_report


def handoff(checkpoint,raw,report,ids):
    result=selection_report(checkpoint,raw,report,split='val',limit=len(ids),sample_ids=set(ids))
    return {'checkpoint':checkpoint,'precision':result['accepted_instance_precision_iou50'],'recall':result['accepted_instance_recall_iou50']}


def completed(run,steps):
    while True:
        try:
            last=torch.load(run/'last.pt',weights_only=True,map_location='cpu')
            if last['step']==steps:
                best=torch.load(run/'best.pt',weights_only=True,map_location='cpu')
                if best['best_score']==last['best_score']:return
        except FileNotFoundError:pass
        time.sleep(5)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('runs_root');p.add_argument('cache');p.add_argument('raw');p.add_argument('output');p.add_argument('sample_ids')
    p.add_argument('--run',action='append',required=True,help='NAME=final_steps');p.add_argument('--device',default='cuda');p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();root=Path(a.runs_root);out=Path(a.output);dest=out/'candidates';dest.mkdir(parents=True,exist_ok=True);cache=Path(a.cache)
    entries=[e for e in json.loads((cache/'index.json').read_text()) if e['split']=='val'];ids=json.loads(Path(a.sample_ids).read_text())['splits']['val'];torch.set_num_threads(2)
    pending=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for spec in a.run:
            name,steps=spec.rsplit('=',1);run=root/name;print('Waiting for',name,flush=True);completed(run,int(steps));seen=set()
            for which in ['best','last']:
                ck=torch.load(run/(which+'.pt'),weights_only=True,map_location='cpu');step=ck['step']
                if step in seen:continue
                seen.add(step);stem=f'{name}-{step}';copy=dest/(stem+'.pt');digest=hashlib.sha256((run/(which+'.pt')).read_bytes()).hexdigest()
                if copy.exists():
                    if hashlib.sha256(copy.read_bytes()).hexdigest()!=digest:raise ValueError(f'Immutable checkpoint conflict: {copy}')
                else:shutil.copyfile(run/(which+'.pt'),copy)
                report=out/(stem+'-validation.json')
                valid=False
                if report.exists():
                    try:valid=json.loads(report.read_text()).get('checkpoint_sha256')==digest
                    except json.JSONDecodeError:pass
                if not valid:
                    model,ck=load_model(copy,a.device);validate_cache_sampling(model,cache)
                    result=evaluate(model,entries,cache,a.device,ck['instances_trained']);result.update(checkpoint=str(copy.resolve()),checkpoint_step=step,checkpoint_sha256=digest)
                    temp=report.with_suffix('.partial');temp.write_text(json.dumps(result,indent=2));temp.replace(report)
                    print(stem,'full validation',result['macro_f1'],flush=True);del model,ck
                    if a.device=='cuda':torch.cuda.empty_cache()
                report=out/(stem+'-handoff-val.json')
                if not report.exists():pending.append(pool.submit(handoff,str(copy.resolve()),a.raw,str(report.resolve()),ids))
        for future in pending:print('Completed handoff',future.result(),flush=True)

if __name__=='__main__':main()
