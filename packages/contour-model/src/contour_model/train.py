"""Reproducible finite-step training, validation selection, resumable checkpoints."""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import math
import random
import time
import numpy as np
import torch
from .data import load_cached,augment,NAMES,LABEL_VERSION
from .model import ContourTransformer,ModelConfig
from .losses import tensor_targets,segmentation_loss,instance_loss


def model_device(device):
    if device=='auto':
        device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; choose --device cpu or run on GPU host')
    if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; choose --device cpu')
    return torch.device(device)


def safe_load(path,device='cpu'):
    return torch.load(path,map_location=device,weights_only=True)


def load_model(path,device='cpu'):
    checkpoint=safe_load(path,device);model=ContourTransformer(ModelConfig(**checkpoint['model_config'])).to(device)
    model.load_state_dict(checkpoint['model']);model.eval()
    return model,checkpoint


def validate_cache_sampling(model,cache):
    config=json.loads((Path(cache)/'config.json').read_text())
    expected=model.config.sampling_version if model.config.adaptive_sampling else 'fixed-v1'
    actual=config.get('sampling','fixed-v1')
    if actual!=expected:raise ValueError(f'Cache sampling {actual!r} does not match model sampling {expected!r}')


def scores(confusion):
    tp=np.diag(confusion);support=confusion.sum(1);predicted=confusion.sum(0)
    precision=np.divide(tp,predicted,out=np.zeros_like(tp),where=predicted>0)
    recall=np.divide(tp,support,out=np.zeros_like(tp),where=support>0)
    f1=np.divide(2*tp,support+predicted,out=np.zeros_like(tp),where=(support+predicted)>0)
    iou=np.divide(tp,support+predicted-tp,out=np.zeros_like(tp),where=(support+predicted-tp)>0)
    present=support>0;present[0]=False
    return {'macro_f1':float(f1[present].mean()) if present.any() else 0.,
            'macro_iou':float(iou[present].mean()) if present.any() else 0.,
            'accuracy':float(tp.sum()/max(confusion.sum(),1e-8)),
            'classes':{NAMES[i]:{'f1':float(f1[i]),'precision':float(precision[i]),'recall':float(recall[i]),'support_length':float(support[i])} for i in range(17)}}


def checkpoint_score(metrics,instances=False,query_scope='all'):
    if not instances:return metrics['macro_f1'],'semantic_macro_f1'
    p=metrics['instances']['precision_iou50'];r=metrics['instances']['recall_iou50']
    if query_scope=='repeats':
        # Auxiliary repeat grouping must not select a checkpoint that forgets surface semantics.
        return .9*metrics['macro_f1']+.1*(2*p*r/max(p+r,1e-12)),'semantic_90_repeat_f1_10'
    return (metrics['macro_f1']+p+r)/3,'mean_semantic_instance_precision_recall'


@torch.no_grad()
def evaluate(model,entries,cache,device,instances=False):
    model.eval();conf=np.zeros((17,17),float);families={};strata={};family_counts={};stratum_counts={};multi_tp=np.zeros(16);multi_fp=np.zeros(16);multi_fn=np.zeros(16)
    boundary_counts=np.zeros(4);query_counts=np.zeros(3);count=0
    for entry in entries:
        a=load_cached(Path(cache)/entry['file']);x=torch.as_tensor(a['x'],device=device)
        out=model(x,a['offsets'],instances=instances)
        predicted=out['primary'].argmax(-1).cpu().numpy();truth=a['primary'].argmax(-1);weight=a['weights']
        c=np.bincount(17*truth+predicted,weights=weight,minlength=289).reshape(17,17)
        conf+=c;families.setdefault(entry['family'],np.zeros((17,17)));families[entry['family']]+=c
        family_counts[entry['family']]=family_counts.get(entry['family'],0)+1
        if 'rings' in entry:
            holes=entry['rings']-1;features=entry['instances']
            groups=['holes:'+('0' if holes==0 else '1-99' if holes<100 else '100-299' if holes<300 else '300+'),
                    'features:'+('<50' if features<50 else '50-199' if features<200 else '200+'),
                    'form:'+('stress' if entry.get('stress') else 'ordinary'),
                    'parameters:'+entry.get('complexity','ordinary')]
            for group in groups:
                strata.setdefault(group,np.zeros((17,17)));strata[group]+=c;stratum_counts[group]=stratum_counts.get(group,0)+1
        p=out['semantic'].sigmoid().cpu().numpy()>.5;t=a['semantic']>.5
        multi_tp+=(p&t).T@weight;multi_fp+=(p&~t).T@weight;multi_fn+=(~p&t).T@weight
        bp=out['boundary'].sigmoid().cpu().numpy()>.5;bt=a['boundary']>.75
        for lo,hi in zip(a['offsets'][:-1],a['offsets'][1:]):
            pp=bp[lo:hi];tt=bt[lo:hi];pd=pp|np.roll(pp,1)|np.roll(pp,-1);td=tt|np.roll(tt,1)|np.roll(tt,-1)
            boundary_counts+=[(pp&td).sum(),pp.sum(),(tt&pd).sum(),tt.sum()]
        if instances:
            from scipy.optimize import linear_sum_assignment
            qp=out['query_class'].softmax(-1).cpu().numpy();cls=qp.argmax(-1)
            keep=(cls!=17)&(qp.max(-1)>.35)
            if model.config.query_scope=='repeats':keep&=np.isin(cls,[10,11])
            masks=out['query_masks'].sigmoid().cpu().numpy()[keep]>.5;cls=cls[keep]
            targets=a['instance_masks']>.5;target_cls=a['instance_classes']
            if model.config.query_scope=='repeats':
                selected=np.isin(target_cls,[10,11]);targets=targets[selected];target_cls=target_cls[selected]
            if len(masks) and len(targets):
                inter=(masks*weight)@targets.T;union=(masks*weight).sum(1)[:,None]+(targets*weight).sum(1)[None]-inter
                iou=inter/np.maximum(union,1e-8);iou*=cls[:,None]==target_cls[None]
                ri,ci=linear_sum_assignment(-iou);matched=int((iou[ri,ci]>=.5).sum())
            else:matched=0
            query_counts += [matched,len(masks),len(targets)]
        count+=1
    result=scores(conf);result['samples']=count;result['by_family']={k:scores(c)['macro_f1'] for k,c in families.items()}
    result['family_details']={k:{'samples':family_counts[k],**scores(c)} for k,c in families.items()}
    result['strata']={k:{'samples':stratum_counts[k],**scores(c)} for k,c in strata.items()}
    mf=2*multi_tp/np.maximum(2*multi_tp+multi_fp+multi_fn,1e-8)
    result['multilabel_f1']={NAMES[i+1]:float(mf[i]) for i in range(16)}
    result['boundary']={'precision':float(boundary_counts[0]/max(boundary_counts[1],1)),
                        'recall':float(boundary_counts[2]/max(boundary_counts[3],1)),'tolerance':'one resampled interval'}
    if instances:result['instances']={'precision_iou50':float(query_counts[0]/max(query_counts[1],1)),
                                     'recall_iou50':float(query_counts[0]/max(query_counts[2],1)),
                                     'scope':model.config.query_scope}
    return result


def train(cache,output,steps=2000,device='auto',seed=42,train_limit=0,val_limit=100,
          eval_every=200,lr=3e-4,accumulate=1,augmentation=True,instances=False,
          init=None,resume=None,width=96,backbone='transformer',freeze_backbone=False,adaptive=False,queries=None,repeat_queries=False,relative_noise=False):
    if init and resume:raise ValueError('Choose either init or resume')
    if min(steps,eval_every,accumulate)<=0 or lr<=0:raise ValueError('Steps, evaluation interval, accumulation and learning rate must be positive')
    cache=Path(cache);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'last.pt').exists() and not resume:raise ValueError('Run already exists; use --resume or a new output')
    torch.set_num_threads(4);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    dev=model_device(device);rng=np.random.default_rng(seed)
    index=json.loads((cache/'index.json').read_text());training=[v for v in index if v['split']=='train'];validation=[v for v in index if v['split']=='val']
    if train_limit:training=training[:train_limit]
    if val_limit:validation=validation[:val_limit]
    if not training or not validation:raise ValueError('Training and validation splits required')
    fingerprint=hashlib.sha256((cache/'index.json').read_bytes()+(cache/'config.json').read_bytes()).hexdigest()
    configuration=ModelConfig(width=width,global_width=width*2,backbone=backbone,
                              adaptive_sampling=adaptive,topology_instances=adaptive,queries=queries or (512 if adaptive else 64),query_scope='repeats' if repeat_queries else 'all')
    model=ContourTransformer(configuration).to(dev)
    if adaptive:model.config.sampling_version=json.loads((cache/'config.json').read_text()).get('sampling','adaptive-turn-v1')
    if init:
        previous=safe_load(init)
        config=ModelConfig(**previous['model_config'])
        if adaptive:
            config.adaptive_sampling=True;config.topology_instances=True;config.queries=queries or 512;config.backbone=backbone
            config.query_scope='repeats' if repeat_queries else 'all'
            config.sampling_version=json.loads((cache/'config.json').read_text()).get('sampling','adaptive-turn-v1')
        model=ContourTransformer(config).to(dev)
        if adaptive:
            state=model.state_dict()
            for k,v in previous['model'].items():
                if k in state and state[k].shape==v.shape:state[k]=v
            n=min(len(previous['model']['queries']),config.queries);state['queries'][:n]=previous['model']['queries'][:n]
            model.load_state_dict(state)
        else:model.load_state_dict(previous['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',enabled=dev.type=='cuda')
    begin=0;best=-1.;best_loss=1e9
    if resume:
        previous=safe_load(resume);model=ContourTransformer(ModelConfig(**previous['model_config'])).to(dev);model.load_state_dict(previous['model'])
        if previous['data_fingerprint']!=fingerprint:raise ValueError('Resume dataset mismatch')
        expected={'steps':steps,'seed':seed,'augmentation':augmentation,'instances':instances,
                  'relative_noise':relative_noise,
                  'freeze_backbone':freeze_backbone,'accumulate':accumulate,'learning_rate':lr,
                  'train_ids':[e['id'] for e in training],'validation_ids':[e['id'] for e in validation]}
        changed=[k for k,v in expected.items() if previous['settings'].get(k,False if k=='relative_noise' else None)!=v]
        if changed:raise ValueError(f'Resume settings mismatch: {changed}; use --init for a new training schedule')
        if previous['step']>=steps:raise ValueError('Checkpoint already completed this schedule; use --init for a new run')
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01);optimizer.load_state_dict(previous['optimizer'])
        scaler.load_state_dict(previous['scaler']);begin=previous['step'];best=previous['best_score']
        rng.bit_generator.state=json.loads(previous['numpy_rng_state']);torch.set_rng_state(previous['torch_rng_state'].cpu())
        if dev.type=='cuda' and previous.get('cuda_rng_state') is not None:torch.cuda.set_rng_state(previous['cuda_rng_state'].cpu())
    if freeze_backbone:
        if not instances:raise ValueError('Backbone freezing requires instance training')
        for name,parameter in model.named_parameters():
            parameter.requires_grad_(name=='queries' or name.startswith('query_') or name.startswith('mask_embed'))
    validate_cache_sampling(model,cache)
    selection_policy=checkpoint_score({'macro_f1':0.,'instances':{'precision_iou50':0.,'recall_iou50':0.}},instances,model.config.query_scope)[1]
    if resume:
        old_policy=previous['settings'].get('checkpoint_selection','mean_semantic_instance_precision_recall' if instances else 'semantic_macro_f1')
        if old_policy!=selection_policy:raise ValueError('Checkpoint selection policy changed; use --init for a new schedule')
    # Cache only the small overfit set; large training uses on-demand compact records.
    small={e['file']:load_cached(cache/e['file']) for e in training} if len(training)<=64 else {}
    histogram=np.zeros(16);weight_total=0.
    for e in training[:1000]:
        a=small.get(e['file']) or load_cached(cache/e['file']);histogram+=(a['semantic']*a['weights'][:,None]).sum(0);weight_total+=a['weights'].sum()
    frequency=histogram/max(weight_total,1e-8)
    class_weights=torch.as_tensor(np.clip(np.sqrt((1-frequency)/np.maximum(frequency,.001)),1,8),device=dev,dtype=torch.float32)
    settings={'cache':str(cache.resolve()),'steps':steps,'seed':seed,'train_samples':len(training),'val_samples':len(validation),
              'augmentation':augmentation,'instances':instances,'freeze_backbone':freeze_backbone,'accumulate':accumulate,'learning_rate':lr,
              'relative_noise':relative_noise,
              'device':str(dev),'label_version':LABEL_VERSION,'model':model.metadata(),'data_fingerprint':fingerprint,
              'tokens':json.loads((cache/'config.json').read_text()).get('tokens',1024),
              'sampling':json.loads((cache/'config.json').read_text()).get('sampling','fixed-v1'),
              'checkpoint_selection':selection_policy,
              'train_ids':[e['id'] for e in training],'validation_ids':[e['id'] for e in validation]}
    (output/'config.json').write_text(json.dumps(settings,indent=2))
    print(f"Training {sum(p.numel() for p in model.parameters()):,} parameters on {dev}; {len(training)} train / {len(validation)} val",flush=True)
    start=time.perf_counter();loss_sum=0.;optimizer.zero_grad(set_to_none=True)
    for step in range(begin,steps):
        model.train();lr_mult=min(1.,(step+1)/max(1,int(.05*steps)))*(.15+.85*.5*(1+math.cos(math.pi*step/steps)))
        for group in optimizer.param_groups:group['lr']=lr*lr_mult
        loss_value=0.
        for micro in range(accumulate):
            entry=training[int(rng.integers(len(training)))];a=small.get(entry['file']) or load_cached(cache/entry['file'])
            if augmentation:a=augment(a,rng,relative_noise=relative_noise)
            x=torch.as_tensor(a['x'],device=dev);t=tensor_targets(a,dev)
            with torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=dev.type=='cuda'):
                out=model(x,a['offsets'],instances=instances)
                loss,_=segmentation_loss(out,t,class_weights)
                if instances:loss=loss+instance_loss(out,t)
            if not torch.isfinite(loss):raise FloatingPointError(f'Non-finite loss at step {step}')
            scaler.scale(loss/accumulate).backward();loss_value+=float(loss.detach())/accumulate
        scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);loss_sum+=loss_value
        if (step+1)%eval_every==0 or step+1==steps:
            metrics=evaluate(model,validation,cache,dev,instances)
            overfit=evaluate(model,training[:32],cache,dev,instances) if train_limit else None
            selection,_=checkpoint_score(overfit if overfit else metrics,instances,model.config.query_scope)
            improved=selection>best;best=max(best,selection)
            report={'step':step+1,'loss':loss_sum/min(eval_every,step+1-begin),'validation':metrics,'overfit':overfit,
                    'elapsed_seconds':time.perf_counter()-start,'peak_cuda_mb':torch.cuda.max_memory_allocated()/1e6 if dev.type=='cuda' else 0.}
            with (output/'history.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
            print(json.dumps({k:report[k] for k in ('step','loss','elapsed_seconds','peak_cuda_mb')}),
                  f"val F1={metrics['macro_f1']:.3f}"+(f" train F1={overfit['macro_f1']:.3f}" if overfit else ''),flush=True)
            checkpoint={'model':model.state_dict(),'model_config':model.metadata(),'optimizer':optimizer.state_dict(),
                        'scaler':scaler.state_dict(),'step':step+1,'best_score':best,'data_fingerprint':fingerprint,
                        'label_version':LABEL_VERSION,'instances_trained':instances,'settings':settings,
                        'torch_rng_state':torch.get_rng_state(),'cuda_rng_state':torch.cuda.get_rng_state() if dev.type=='cuda' else None,
                        'numpy_rng_state':json.dumps(rng.bit_generator.state),'report':report}
            temporary=output/'last.pt.partial';torch.save(checkpoint,temporary);temporary.replace(output/'last.pt')
            if improved:
                temporary=output/'best.pt.partial';torch.save(checkpoint,temporary);temporary.replace(output/'best.pt')
                (output/'best-report.json').write_text(json.dumps(report,indent=2))
            loss_sum=0.
    return output/'best.pt'
