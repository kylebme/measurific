import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


def tensor_targets(sample,device):
    return {k:torch.as_tensor(v,device=device,dtype=torch.long if k=='instance_classes' else torch.float32)
            for k,v in sample.items() if k not in ('offsets','x')}


def segmentation_loss(pred,t,class_weights=None):
    weights=t['weights']/t['weights'].sum()
    target=t['semantic'];positive=class_weights if class_weights is not None else torch.ones(16,device=target.device)*3
    bce=F.binary_cross_entropy_with_logits(pred['semantic'],target,reduction='none',pos_weight=positive)
    sem=(bce*weights[:,None]).sum(0).mean()
    p=pred['semantic'].sigmoid();inter=(p*target*weights[:,None]).sum(0)
    denom=((p+target)*weights[:,None]).sum(0)
    present=(target*weights[:,None]).sum(0)>0
    dice=(1-(2*inter+1e-5)/(denom+1e-5))[present].mean()
    # Primary target supports mixed intervals crossing source feature boundaries.
    ce=-(t['primary']*F.log_softmax(pred['primary'],dim=-1)).sum(-1)
    if class_weights is not None:
        primary_weights=torch.cat([torch.ones(1,device=target.device),class_weights.sqrt()])
        ce=ce*(t['primary']*primary_weights).sum(-1)
    primary=(ce*weights).sum()
    boundary=(F.binary_cross_entropy_with_logits(pred['boundary'],t['boundary'],reduction='none',pos_weight=torch.tensor(8.,device=target.device))*weights).sum()
    defect=(-t['defect']*F.log_softmax(pred['defect'],dim=-1)*torch.tensor([.2,2.,2.],device=target.device)).sum(-1)
    defect=(defect*weights).sum()
    total=sem+dice+primary+.3*boundary+.15*defect
    return total,{'semantic':float(sem.detach()),'dice':float(dice.detach()),'primary':float(primary.detach()),'boundary':float(boundary.detach())}


def instance_loss(pred,t):
    classes=t['instance_classes'];masks=t['instance_masks'];q=pred['query_masks'];logits=pred['query_class']
    weights=t['weights']
    if 'instance_token_indices' in pred:
        indices=pred['instance_token_indices'];masks=masks[:,indices];weights=weights[indices]
        keep=masks.sum(1)>0;classes=classes[keep];masks=masks[keep]
    if pred.get('query_scope')=='repeats':
        keep=(classes==10)|(classes==11);classes=classes[keep];masks=masks[keep]
    if len(classes)>len(q):raise ValueError('Query capacity below target instance count')
    weights=weights/weights.sum()
    if not len(classes):
        return F.cross_entropy(logits,torch.full((len(q),),17,device=q.device,dtype=torch.long))
    with torch.no_grad():
        prob=q.sigmoid();inter=(prob*weights)@masks.T
        denom=(prob*weights).sum(1)[:,None]+(masks*weights).sum(1)[None]
        cost=-(logits.softmax(-1)[:,classes])+2*(1-(2*inter+1e-5)/(denom+1e-5))
        rows,cols=linear_sum_assignment(cost.float().cpu().numpy())
    rows=torch.as_tensor(rows,device=q.device);cols=torch.as_tensor(cols,device=q.device)
    labels=torch.full((len(q),),17,dtype=torch.long,device=q.device);labels[rows]=classes[cols]
    class_weight=torch.ones(18,device=q.device);class_weight[17]=.15
    ce=F.cross_entropy(logits,labels,weight=class_weight)
    selected=q[rows];truth=masks[cols]
    bce=(F.binary_cross_entropy_with_logits(selected,truth,reduction='none')*weights).sum(-1).mean()
    prob=selected.sigmoid();dice=(1-(2*(prob*truth*weights).sum(-1)+1e-5)/((prob+truth).mul(weights).sum(-1)+1e-5)).mean()
    return ce+bce+2*dice
