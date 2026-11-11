"""Small shared encoder, primitive segmentation, secondary compound classification.

Compound segmentation sees learned primitive probabilities and contour context.
Queries associate repeated spans; closed slot rings need no fixed query budget.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from inspection_shapes.primitive_labels import LABEL_VERSION, BASE_NAMES, COMPOUND_NAMES
from .model import ContourTransformer, ModelConfig
from .data import load_cached, augment
from .train import load_model, model_device


def query_loss(out, target):
    logits=out['query_class']; masks=out['query_masks']; truth=target['instance_masks']
    classes=target['instance_classes'].long(); weights=target['weights']/target['weights'].sum()
    labels=torch.zeros(len(logits),dtype=torch.long,device=logits.device)
    if len(classes)>len(logits):raise ValueError('Repeat query capacity exceeded')
    if not len(classes):return F.cross_entropy(logits,labels)
    with torch.no_grad():
        prob=masks.sigmoid(); inter=(prob*weights)@truth.T
        den=(prob*weights).sum(1)[:,None]+(truth*weights).sum(1)[None]
        cost=-logits.softmax(-1)[:,classes]+2*(1-(2*inter+1e-6)/(den+1e-6))
        ri,ci=linear_sum_assignment(cost.float().cpu().numpy())
    labels[ri]=classes[ci]
    ce=F.cross_entropy(logits,labels,weight=logits.new_tensor([.15,1,1,1,1]))
    selected=masks[ri]; actual=truth[ci]; prob=selected.sigmoid()
    bce=(F.binary_cross_entropy_with_logits(selected,actual,reduction='none')*weights).sum(-1).mean()
    dice=(1-(2*(prob*actual*weights).sum(-1)+1e-6)/((prob+actual).mul(weights).sum(-1)+1e-6)).mean()
    return ce+bce+2*dice


def losses(out,t,base_weights,compound_weights,query_weight=1.,boundary_positive_weight=8.,boundary_weight=.6):
    w=t['weights']/t['weights'].sum()
    base=-(t['primary']*F.log_softmax(out['primary'],-1)*base_weights).sum(-1).mul(w).sum()
    sem=F.binary_cross_entropy_with_logits(out['semantic'],t['semantic'],reduction='none').mean(-1).mul(w).sum()
    boundary=F.binary_cross_entropy_with_logits(out['boundary'],t['boundary'],reduction='none',
                                               pos_weight=w.new_tensor(boundary_positive_weight)).mul(w).sum()
    compound=-(t['compound']*F.log_softmax(out['compound'],-1)*compound_weights).sum(-1).mul(w).sum()
    defect=-(t['defect']*F.log_softmax(out['defect'],-1)*w.new_tensor([.2,2,2])).sum(-1).mul(w).sum()
    query=query_loss(out,t)
    return base+.25*sem+boundary_weight*boundary+compound+.15*defect+query_weight*query


def initialize_from_previous_schema(model, old_state):
    """Warm-start shared weights and the retained rows of the v1 heads."""
    state=model.state_dict()
    for key,value in old_state.items():
        if key in state and state[key].shape==value.shape:
            state[key]=value
    # v2 removes the final arbitrary-profile row from both base heads.
    for key in ('primary.weight','primary.bias','semantic.weight','semantic.bias'):
        if key in old_state and key in state and old_state[key].shape[1:]==state[key].shape[1:]:
            state[key]=old_state[key][:state[key].shape[0]]
    # Compound input is [encoder features, base probabilities]. Retain the
    # encoder and unlabeled/line/arc/circle columns, dropping only arbitrary.
    key='compound.0.weight'
    if key in old_state and key in state and old_state[key].shape[0]==state[key].shape[0]:
        state[key]=old_state[key][:,:state[key].shape[1]]
    model.load_state_dict(state)


def metrics(conf,names,skip_zero=True):
    tp=np.diag(conf);support=conf.sum(1);pred=conf.sum(0)
    f1=2*tp/np.maximum(support+pred,1e-12)
    present=support>0
    if skip_zero:present[0]=False
    return {'macro_f1':float(f1[present].mean()) if present.any() else 0.,
            'accuracy':float(tp.sum()/max(conf.sum(),1e-12)),
            'classes':{name:{'f1':float(f1[i]),'precision':float(tp[i]/max(pred[i],1e-12)),
                             'recall':float(tp[i]/max(support[i],1e-12)),
                             'support_length':float(support[i])} for i,name in enumerate(names)},
            'confusion':conf.tolist()}


@torch.no_grad()
def evaluate(model,entries,cache,device,form_error=0.):
    base_names=(BASE_NAMES if model.config.label_schema==LABEL_VERSION else
                ['unlabeled','line','arc','circle','arbitrary_profile'])
    nbase=len(base_names)
    model.eval();base=np.zeros((nbase,nbase));compound=np.zeros((5,5));boundary=np.zeros(4)
    query_counts=np.zeros(3);families={}
    for e in entries:
        a=load_cached(Path(cache)/e['file'])
        if form_error:
            from .intent import deform
            a=deform(a,np.random.default_rng(int(e['id'])+91823),form_error)
        out=model(torch.as_tensor(a['x'],device=device),a['offsets'],instances=True)
        w=a['weights']
        for key,conf in [('primary',base),('compound',compound)]:
            p=out[key].argmax(-1).cpu().numpy();t=a[key].argmax(-1)
            n=nbase if key=='primary' else 5
            c=np.bincount(n*t+p,weights=w,minlength=n*n).reshape(n,n);conf+=c
            if key=='primary':families.setdefault(e['family'],np.zeros((nbase,nbase)));families[e['family']]+=c
        p=out['boundary'].sigmoid().cpu().numpy()>.5;t=a['boundary']>.75
        for lo,hi in zip(a['offsets'][:-1],a['offsets'][1:]):
            pp,tt=p[lo:hi],t[lo:hi]
            pd=pp|np.roll(pp,1)|np.roll(pp,-1);td=tt|np.roll(tt,1)|np.roll(tt,-1)
            boundary+=[(pp&td).sum(),pp.sum(),(tt&pd).sum(),tt.sum()]
        cls=out['query_class'].argmax(-1).cpu().numpy();keep=cls>=2
        masks=out['query_masks'].sigmoid().cpu().numpy()[keep]>.5
        actual=a['instance_masks']>.5
        matched=0
        if len(masks) and len(actual):
            inter=(masks*w)@actual.T
            union=(masks*w).sum(1)[:,None]+(actual*w).sum(1)[None]-inter
            iou=inter/np.maximum(union,1e-8)
            iou*=cls[keep,None]==a['instance_classes'][None]
            ri,ci=linear_sum_assignment(-iou);matched=int((iou[ri,ci]>=.5).sum())
        query_counts += [matched,len(masks),len(actual)]
    precision=query_counts[0]/max(query_counts[1],1);recall=query_counts[0]/max(query_counts[2],1)
    result={'samples':len(entries),'base':metrics(base,base_names),
            'compound':metrics(compound,COMPOUND_NAMES),
            'repeat_instances':{'precision_iou50':float(precision),'recall_iou50':float(recall),
                                'f1_iou50':float(2*precision*recall/max(precision+recall,1e-12)),
                                'counts':query_counts.tolist()},
            'boundary':{'precision':float(boundary[0]/max(boundary[1],1)),
                        'recall':float(boundary[2]/max(boundary[3],1))},
            'by_family':{f:metrics(c,base_names)['macro_f1'] for f,c in families.items()}}
    result['selection_score']=.7*result['base']['macro_f1']+.2*result['compound']['macro_f1']+.1*result['repeat_instances']['f1_iou50']
    return result


def train(cache,output,steps=12000,seed=42,width=64,backbone='transformer',lr=3e-4,
          eval_every=1000,val_limit=200,device='cuda',init=None,query_weight=.5,
          direct_global_context=False,pooled_tokens=128,dropout=0.,form_error=0.,
          boundary_positive_weight=8.,boundary_weight=.6,full_context=False,resume=False,ema=0.,mirror=False,
          val_extractor=None):
    cache=Path(cache);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    resuming=resume and (output/'last.pt').exists()
    if (output/'config.json').exists() and not resuming:raise ValueError('Use a new experiment directory')
    torch.set_num_threads(2);torch.manual_seed(seed);rng=np.random.default_rng(seed);dev=model_device(device)
    config=json.loads((cache/'config.json').read_text())
    if config['label_version']!=LABEL_VERSION:raise ValueError('Wrong label schema')
    index=json.loads((cache/'index.json').read_text())
    training=[e for e in index if e['split']=='train'];validation=[e for e in index if e['split']=='val']
    # Stable shuffle spreads validation coverage over families and complexities.
    np.random.default_rng(100).shuffle(validation)
    # Mixed corpora: select on one observation source (e.g. the GUI's default PAE
    # extraction) and log the others; clean contours are the easiest case.
    tracked={}
    if val_extractor:
        groups={}
        for e in validation:groups.setdefault(e.get('extractor','clean'),[]).append(e)
        if val_extractor not in groups:raise ValueError(f'No validation entries from {val_extractor}')
        validation=groups.pop(val_extractor)
        tracked={k:(v[:val_limit] if val_limit else v) for k,v in groups.items()}
    if val_limit:validation=validation[:val_limit]
    model=ContourTransformer(ModelConfig(width=width,global_width=width*2,queries=32,
        adaptive_sampling=True,topology_instances=False,query_scope='compounds',
        sampling_version=config['sampling'],label_schema=LABEL_VERSION,backbone=backbone,
        direct_global_context=direct_global_context,pooled_tokens=pooled_tokens,dropout=dropout,
        full_context=full_context)).to(dev)
    if init and not resuming:
        old=torch.load(init,map_location='cpu',weights_only=True)
        initialize_from_previous_schema(model,old['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    # Optional exponential moving average of weights: evaluated and saved as 'model'
    # (the raw training weights are kept as 'raw_model' for resuming).
    averaged=None
    if ema:
        from torch.optim.swa_utils import AveragedModel,get_ema_multi_avg_fn
        averaged=AveragedModel(model,multi_avg_fn=get_ema_multi_avg_fn(ema),use_buffers=True)
    scaler=torch.amp.GradScaler('cuda',enabled=dev.type=='cuda')
    hist=np.zeros(len(BASE_NAMES));chist=np.zeros(5);total=0
    for e in training[:500]:
        a=load_cached(cache/e['file']);hist+=(a['primary']*a['weights'][:,None]).sum(0)
        chist+=(a['compound']*a['weights'][:,None]).sum(0);total+=a['weights'].sum()
    bw=torch.tensor(np.clip(np.sqrt(total/np.maximum(hist,1)),1,8),device=dev,dtype=torch.float32)
    cw=torch.tensor(np.clip(np.sqrt(total/np.maximum(chist,1)),1,8),device=dev,dtype=torch.float32)
    fingerprint=hashlib.sha256((cache/'config.json').read_bytes()+(cache/'index.json').read_bytes()).hexdigest()
    settings=dict(cache=str(cache.resolve()),steps=steps,seed=seed,width=width,backbone=backbone,lr=lr,device=str(dev),
                  query_weight=query_weight,init=str(init) if init else None,sampling=config['sampling'],
                  direct_global_context=direct_global_context,pooled_tokens=pooled_tokens,dropout=dropout,
                  full_context=full_context,ema=ema,mirror=mirror,val_extractor=val_extractor,
                  tracked_validation={k:[e['id'] for e in v] for k,v in tracked.items()},
                  form_error=form_error,boundary_positive_weight=boundary_positive_weight,boundary_weight=boundary_weight,
                  tokens=None,validation_ids=[e['id'] for e in validation],train_samples=len(training),
                  label_version=LABEL_VERSION,selection_policy='70% base macro F1 + 20% compound macro F1 + 10% repeat instance F1')
    if form_error: settings['selection_policy']='70% clean + 30% fixed form-error validation selection scores'
    first=1;best=-1
    if resuming:
        # Continue the identical schedule: weights, optimizer, scaler and RNG states.
        state=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state.get('raw_model',state['model']));optimizer.load_state_dict(state['optimizer'])
        if averaged is not None and 'ema_state' in state:averaged.load_state_dict(state['ema_state'])
        scaler.load_state_dict(state['scaler']);torch.set_rng_state(state['torch_rng_state'])
        rng.bit_generator.state=json.loads(state['numpy_rng_state']);first=state['step']+1
        if (output/'best-report.json').exists():best=json.loads((output/'best-report.json').read_text())['selection_score']
        print(f'Resuming at step {first}',flush=True)
    else:(output/'config.json').write_text(json.dumps(settings,indent=2))
    print(f'Training {sum(p.numel() for p in model.parameters()):,} parameters; {len(training)} train, {len(validation)} val',flush=True)
    start=time.monotonic();loss_sum=0
    for step in range(first,steps+1):
        model.train();a=load_cached(cache/training[int(rng.integers(len(training)))]['file'])
        if form_error and rng.random()<.75:
            from .intent import deform
            a=deform(a,rng,form_error*rng.uniform(.25,1))
        # Augmentation must apply the identical permutation to compound targets.
        semantic_count=len(BASE_NAMES)-1
        joined=np.concatenate([a['semantic'],a['compound']],axis=1);a['semantic']=joined
        a=augment(a,rng,relative_noise=True,reflect=mirror)
        a['compound']=a['semantic'][:,semantic_count:];a['semantic']=a['semantic'][:,:semantic_count]
        t={k:torch.as_tensor(v,device=dev) for k,v in a.items() if k not in ('x','offsets')}
        multiplier=min(1,step/max(1,steps*.03))*(.15+.85*(1+math.cos(math.pi*step/steps))/2)
        for g in optimizer.param_groups:g['lr']=lr*multiplier
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=dev.type=='cuda'):
            out=model(torch.as_tensor(a['x'],device=dev),a['offsets'],instances=True)
            loss=losses(out,t,bw,cw,query_weight,boundary_positive_weight,boundary_weight)
        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at step {step}')
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(optimizer);scaler.update()
        if averaged is not None:averaged.update_parameters(model)
        loss_sum+=float(loss.detach())
        if step%100==0:print(f'step {step}/{steps}, elapsed {time.monotonic()-start:.0f}s',flush=True)
        if step%eval_every==0 or step==steps:
            scored=averaged.module if averaged is not None else model
            report=evaluate(scored,validation,cache,dev)
            if form_error:
                robust=evaluate(scored,validation,cache,dev,form_error=form_error)
                report['clean_selection_score']=report['selection_score']
                report['form_error']=robust
                report['selection_score']=.7*report['selection_score']+.3*robust['selection_score']
            for k,entries in tracked.items():
                extra=evaluate(scored,entries,cache,dev)
                report[f'{k}_val']={key:extra[key] for key in ('selection_score','base','compound','repeat_instances','samples')}
            report.update(step=step,loss=loss_sum/min(step,eval_every),elapsed_seconds=time.monotonic()-start,
                          peak_cuda_mb=torch.cuda.max_memory_allocated()/1e6 if dev.type=='cuda' else 0)
            with (output/'history.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
            print(json.dumps({**{k:report[k] for k in ('step','loss','selection_score','elapsed_seconds')},
                              **{f'{k}_val':report[f'{k}_val']['selection_score'] for k in tracked}}),flush=True)
            checkpoint=dict(model=scored.state_dict(),model_config=model.metadata(),step=step,label_version=LABEL_VERSION,
                            instances_trained=True,settings=settings,data_fingerprint=fingerprint,report=report,
                            optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                            numpy_rng_state=json.dumps(rng.bit_generator.state),torch_rng_state=torch.get_rng_state())
            if averaged is not None:checkpoint.update(raw_model=model.state_dict(),ema_state=averaged.state_dict())
            torch.save(checkpoint,output/'last.pt')
            if report['selection_score']>best:
                best=report['selection_score'];torch.save(checkpoint,output/'best.pt')
                (output/'best-report.json').write_text(json.dumps(report,indent=2))
            loss_sum=0
    return output/'best.pt'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cache');p.add_argument('output')
    p.add_argument('--steps',type=int,default=12000);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--width',type=int,default=64);p.add_argument('--backbone',choices=['transformer','conv'],default='transformer')
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--eval-every',type=int,default=1000)
    p.add_argument('--val-limit',type=int,default=200);p.add_argument('--device',default='cuda')
    p.add_argument('--init');p.add_argument('--query-weight',type=float,default=.5)
    p.add_argument('--direct-global-context',action='store_true');p.add_argument('--pooled-tokens',type=int,default=128)
    p.add_argument('--full-context',action='store_true')
    p.add_argument('--dropout',type=float,default=0.);p.add_argument('--form-error',type=float,default=0.)
    p.add_argument('--boundary-positive-weight',type=float,default=8.);p.add_argument('--boundary-weight',type=float,default=.6)
    p.add_argument('--resume',action='store_true');p.add_argument('--ema',type=float,default=0.)
    p.add_argument('--mirror',action='store_true',help='Random reflection augmentation (corpus v3 recipe)')
    p.add_argument('--val-extractor',help="Select checkpoints on validation rows from this extractor ('pae'); log the rest")
    train(**vars(p.parse_args()))
