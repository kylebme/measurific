"""Reference/prediction contact sheet using original observed edges."""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from .train import load_model
from .inference import predict

COLORS=['#777777','#168aad','#ef8354','#8c5ec0','#36a657']


def review(checkpoint,cache,output,split='val',limit=12):
    torch.set_num_threads(4);cache=Path(cache);model,_=load_model(checkpoint)
    rows=[e for e in json.loads((cache/'index.json').read_text()) if e['split']==split]
    wanted=['slotted_bracket','multi_thread_shaft','threaded_shaft','toothed_disk','internal_spline_ring',
            'timing_pulley','cam','d_profile_spacer','labyrinth_plate','comb_plate','mixed_shaft','polygon_broach_plate']
    chosen=[next(e for e in rows if e['family']==f) for f in wanted][:limit]
    fig,axes=plt.subplots(len(chosen),2,figsize=(13,3*len(chosen)),squeeze=False)
    examples=[]
    for row,e in enumerate(chosen):
        p=Path(e['file']);p=p if p.is_absolute() else cache/p
        source=p.parent/'sources'/(e['id']+'.contours.npz')
        with np.load(source) as a:c={k:a[k] for k in a.files}
        result,arrays=predict(model,c,instances=True)
        examples.append({'id':e['id'],'family':e['family'],'source':str(source),'prediction':result})
        q=c['actual_xy_mm'];end=np.arange(len(q))+1
        for lo,hi in zip(c['ring_offsets'][:-1],c['ring_offsets'][1:]):end[hi-1]=lo
        segments=np.stack([q,q[end]],axis=1)
        for col,classes in enumerate([c['edge_base_class'],arrays['classes']]):
            ax=axes[row,col];ax.add_collection(LineCollection(segments,colors=np.array(COLORS)[classes],linewidths=1.2))
            ax.autoscale();ax.set_aspect('equal');ax.margins(.06);ax.axis('off')
            title=f"{e['id']} {e['family']} — {'reference' if col==0 else 'prediction'}"
            ax.set_title(title,fontsize=10)
            if col:
                ax.text(.02,.02,', '.join(f"{v['semantic']}" for v in result['compound_features'][:8]),
                        transform=ax.transAxes,fontsize=7,wrap=True)
    fig.suptitle('Line: blue | Arc: orange | Circle: purple | Arbitrary profile: green',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.99]);destination=Path(output);destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination,dpi=130);plt.close(fig)
    destination.with_suffix('.json').write_text(json.dumps(examples))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('cache');p.add_argument('output')
    p.add_argument('--split',default='val');p.add_argument('--limit',type=int,default=12)
    review(**vars(p.parse_args()))
