"""A contour-window transformer; no dense fine-resolution NxN attention."""
from dataclasses import dataclass,asdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .data import INPUT_DIM

@dataclass
class ModelConfig:
    width: int = 96
    global_width: int = 192
    heads: int = 4
    window: int = 64
    local_depth: int = 2
    global_depth: int = 3
    pooled_tokens: int = 128
    queries: int = 64
    backbone: str = 'transformer'
    topology_instances: bool = False
    adaptive_sampling: bool = False
    query_scope: str = 'all'
    sampling_version: str = 'adaptive-turn-v1'
    label_schema: str = 'observed-compounds-v1'
    direct_global_context: bool = False
    dropout: float = 0.
    full_context: bool = False


def primitive_class_count(schema):
    return {'primitives-compounds-v1': 5, 'primitives-compounds-v2': 4}.get(schema)

class LocalBlock(nn.Module):
    def __init__(self,width,heads,window,shift):
        super().__init__();self.window=window;self.shift=shift
        self.norm=nn.LayerNorm(width);self.attn=nn.MultiheadAttention(width,heads,batch_first=True)
        self.ff=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width*3),nn.GELU(),nn.Linear(width*3,width))
    def forward(self,x,offsets):
        gather=[];restore=[];start=0
        for lo,hi in zip(offsets[:-1],offsets[1:]):
            n=hi-lo;pad=(-n)%self.window
            gather.append(lo+(np.arange(n+pad)-self.shift)%n)
            restore.append(start+(np.arange(n)+self.shift)%n);start+=n+pad
        idx=torch.as_tensor(np.concatenate(gather),device=x.device)
        z=x[idx].reshape(-1,self.window,x.shape[-1]);norm=self.norm(z)
        z=z+self.attn(norm,norm,norm,need_weights=False)[0];z=z+self.ff(z)
        return z.reshape(-1,x.shape[-1])[torch.as_tensor(np.concatenate(restore),device=x.device)]

class CircularConv(nn.Module):
    def __init__(self,width,dilation):
        super().__init__();self.norm=nn.LayerNorm(width);self.conv=nn.Conv1d(width,width,5,dilation=dilation)
        self.pad=2*dilation;self.ff=nn.Sequential(nn.GELU(),nn.Linear(width,width))
    def forward(self,x,offsets):
        out=[]
        for lo,hi in zip(offsets[:-1],offsets[1:]):
            r=self.norm(x[lo:hi]);indices=torch.arange(-self.pad,len(r)+self.pad,device=x.device)%len(r)
            z=self.conv(r[indices].T[None])[0].T;out.append(x[lo:hi]+self.ff(z))
        return torch.cat(out)

class ContourTransformer(nn.Module):
    def __init__(self,config=None):
        super().__init__();self.config=config or ModelConfig();c=self.config
        self.embed=nn.Sequential(nn.Linear(INPUT_DIM,c.width),nn.LayerNorm(c.width),nn.GELU(),nn.Linear(c.width,c.width))
        self.local=nn.ModuleList([self.block(i) for i in range(c.local_depth)])
        self.up=nn.Linear(c.width,c.global_width)
        layer=nn.TransformerEncoderLayer(c.global_width,c.heads,c.global_width*3,dropout=c.dropout,batch_first=True,norm_first=True)
        self.global_encoder=nn.TransformerEncoder(layer,c.global_depth,enable_nested_tensor=False)
        self.decode=nn.Sequential(nn.Linear(c.width+c.global_width,c.width),nn.GELU(),nn.LayerNorm(c.width))
        if c.direct_global_context:
            self.context_norm=nn.LayerNorm(c.width)
            self.context_attention=nn.MultiheadAttention(c.width,c.heads,kdim=c.global_width,
                vdim=c.global_width,dropout=c.dropout,batch_first=True)
            # Warm starts initially preserve the old local/interpolated pathway.
            self.context_gate=nn.Parameter(torch.zeros(()))
        self.refine=self.block(c.local_depth)
        primitive_classes=primitive_class_count(c.label_schema);primitive=primitive_classes is not None
        self.semantic=nn.Linear(c.width,primitive_classes-1 if primitive else 16)
        self.primary=nn.Linear(c.width,primitive_classes if primitive else 17)
        if primitive:
            self.compound=nn.Sequential(nn.Linear(c.width+primitive_classes,c.width),nn.GELU(),nn.Linear(c.width,5))
        self.boundary=nn.Linear(c.width,1);self.defect=nn.Linear(c.width,3)
        self.queries=nn.Parameter(torch.randn(c.queries,c.width)*.02)
        decoder=nn.TransformerDecoderLayer(c.width,c.heads,c.width*3,dropout=0.,batch_first=True,norm_first=True)
        self.query_decoder=nn.TransformerDecoder(decoder,2)
        self.query_class=nn.Linear(c.width,5 if primitive else 18)
        self.query_mask=nn.Linear(c.width,c.width)
        self.mask_embed=nn.Linear(c.width,c.width)
    def block(self,i):
        c=self.config
        return LocalBlock(c.width,c.heads,c.window,(i%2)*(c.window//2)) if c.backbone=='transformer' else CircularConv(c.width,2**(i%3))
    def forward(self,x,offsets,instances=False):
        offsets=[int(v) for v in offsets];z=self.embed(x)
        for block in self.local:z=block(z,offsets)
        sizes=[];sources=[];buckets=[];lefts=[];rights=[];fractions=[];start=0
        for lo,hi in zip(offsets[:-1],offsets[1:]):
            n=hi-lo;size=min(n,max(1 if self.config.adaptive_sampling else 4,round(self.config.pooled_tokens*n/len(x))))
            sizes.append(size)
            for j in range(size):
                indices=np.arange(j*n//size,((j+1)*n+size-1)//size)+lo
                sources.append(indices);buckets.append(np.full(len(indices),start+j))
            pos=(np.arange(n,dtype=np.float32)+.5)*size/n-.5;left=np.floor(pos).astype(np.int64)
            lefts.append(start+left%size);rights.append(start+(left+1)%size);fractions.append(pos-left)
            start+=size
        src=torch.as_tensor(np.concatenate(sources),device=x.device);bucket=torch.as_tensor(np.concatenate(buckets),device=x.device)
        pooled=z.new_zeros((start,z.shape[-1])).index_add(0,bucket,z[src])
        pooled=pooled/torch.bincount(bucket,minlength=start).to(z.dtype)[:,None]
        coarse=self.global_encoder(self.up(pooled)[None])[0]
        left=torch.as_tensor(np.concatenate(lefts),device=x.device);right=torch.as_tensor(np.concatenate(rights),device=x.device)
        fraction=torch.as_tensor(np.concatenate(fractions),device=x.device,dtype=x.dtype)[:,None]
        context=coarse[left]*(1-fraction)+coarse[right]*fraction
        fine_memory=self.up(z)+context if self.config.full_context else None
        z=self.decode(torch.cat([z,context],dim=-1))
        if self.config.direct_global_context:
            # Chunk queries: no fine-resolution N x N attention or giant score tensor.
            memory=fine_memory if self.config.full_context else coarse
            def attend(chunk, memory):
                return self.context_attention(self.context_norm(chunk)[None],
                    memory[None],memory[None],need_weights=False)[0][0]
            if self.config.full_context and self.training:
                from torch.utils.checkpoint import checkpoint
                # Recompute scores during backward instead of storing N squared
                # activations. No keys are dropped: every token sees every token.
                attended=torch.cat([checkpoint(attend,chunk,memory,use_reentrant=False,
                    preserve_rng_state=True) for chunk in z.split(128)])
            else:
                attended=torch.cat([attend(chunk,memory) for chunk in z.split(128 if self.config.full_context else 512)])
            z=z+self.context_gate.tanh()*attended
        z=self.refine(z,offsets)
        out={'semantic':self.semantic(z),'primary':self.primary(z),'boundary':self.boundary(z).squeeze(-1),
             'defect':self.defect(z)}
        if primitive_class_count(self.config.label_schema) is not None:
            out['compound']=self.compound(torch.cat([z,out['primary'].softmax(-1)],dim=-1))
        if instances:
            exterior=x[:,5]<.5
            memory=z[exterior] if self.config.topology_instances else z
            q=self.query_decoder(self.queries[None],memory[None])[0]
            out['query_class']=self.query_class(q)
            mask_features=z[exterior] if self.config.topology_instances and self.training else z
            out['query_masks']=self.query_mask(q)@self.mask_embed(mask_features).T/(self.config.width**.5)
            if self.config.topology_instances:
                if self.training:
                    out['instance_token_indices']=torch.where(exterior)[0]
                    out['query_scope']=self.config.query_scope
                    return out
                # Through-opening loops already supply exact instance connectivity.
                # Learned queries cover outer surfaces; loop count has no query ceiling.
                out['query_masks']=out['query_masks'].masked_fill(~exterior[None],-20.)
                ring_ids=torch.as_tensor(np.repeat(np.arange(len(offsets)-1),np.diff(offsets)),device=x.device)
                flags=x[torch.as_tensor(offsets[:-1],device=x.device),5]>.5
                hole_ids=torch.where(flags)[0]
                if len(hole_ids):
                    sums=out['primary'].new_zeros((len(offsets)-1,17)).index_add(0,ring_ids,out['primary'])
                    means=sums/torch.as_tensor(np.diff(offsets),device=x.device,dtype=sums.dtype)[:,None]
                    classes=F.pad(means[hole_ids],(0,1),value=-20.)
                    masks=torch.where(hole_ids[:,None]==ring_ids[None],20.,-20.).to(z.dtype)
                    out['query_class']=torch.cat([out['query_class'],classes])
                    out['query_masks']=torch.cat([out['query_masks'],masks])

        return out
    def metadata(self):return asdict(self.config)
