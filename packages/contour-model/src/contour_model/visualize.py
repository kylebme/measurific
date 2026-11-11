"""Static contour-label and measurement-selection review sheets."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from .data import records,adapted_labels,NAMES
from .train import load_model
from .inference import predict

PALETTE=np.array([[120,120,120]]+[[int(v) for v in np.random.default_rng(i+17).integers(60,240,3)] for i in range(1,17)],np.uint8)


def draw_contour(image,xy,offsets,labels,box):
    d=ImageDraw.Draw(image);left,top,width,height=box
    scale=min((width-20)/max(np.ptp(xy[:,0]),1e-8),(height-20)/max(np.ptp(xy[:,1]),1e-8))
    q=(xy-(xy.min(0)+xy.max(0))/2)*[scale,-scale]+[left+width/2,top+height/2]
    for lo,hi in zip(offsets[:-1],offsets[1:]):
        for i in range(lo,hi):
            j=lo if i+1==hi else i+1
            d.line([tuple(q[i]),tuple(q[j])],fill=tuple(PALETTE[int(labels[i])]),width=2)


def review(checkpoint,source,destination,split='test',limit=12,tokens=1024):
    torch.set_num_threads(4);model,ckpt=load_model(checkpoint);tiles=[];seen=set()
    for meta,c in records(source):
        if meta['split']!=split or meta['family'] in seen:continue
        seen.add(meta['family'])
        _,truth,_=adapted_labels(meta,c)
        result,prob=predict(model,c,tokens=tokens,instances=ckpt['instances_trained'])
        tile=Image.new('RGB',(640,370),'#182027');d=ImageDraw.Draw(tile)
        d.text((10,8),f"{meta['sample_id']}  {meta['family']}  {split}",fill='white')
        d.text((10,28),'Reference labels',fill='white');d.text((330,28),'Predicted labels',fill='white')
        draw_contour(tile,c['actual_xy_mm'],c['ring_offsets'],truth,(0,48,320,260))
        draw_contour(tile,c['actual_xy_mm'],c['ring_offsets'],prob['classes'],(320,48,320,260))
        nerr=int(np.sum(truth!=prob['classes']));accepted=sum(f['status']=='accepted' for f in result['features'])
        d.text((10,320),f"Edge disagreements {nerr}/{len(truth)} | accepted selections {accepted}",fill='white')
        present=np.unique(truth);x=10
        for cls in present:
            name=NAMES[int(cls)];d.text((x,344),name,fill=tuple(PALETTE[cls]));x+=len(name)*6+14
        tiles.append(tile)
        if len(tiles)>=limit:break
    if not tiles:raise ValueError('No matching review samples')
    sheet=Image.new('RGB',(1920,370*((len(tiles)+2)//3)),'#182027')
    for i,tile in enumerate(tiles):sheet.paste(tile,((i%3)*640,(i//3)*370))
    Path(destination).parent.mkdir(parents=True,exist_ok=True);sheet.save(destination)
    return len(tiles)
