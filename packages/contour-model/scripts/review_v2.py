"""Render fixed complex held-out examples with semantic and instance predictions."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from contour_model.validation import indexed_records,observable_reference_labels
from contour_model.data import adapted_labels
from contour_model.train import load_model
from contour_model.inference import predict
from contour_model.visualize import draw_contour


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('checkpoint');p.add_argument('source');p.add_argument('output');p.add_argument('--split',default='test');a=p.parse_args()
    source=Path(a.source);output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    index=json.loads((source/'index.json').read_text())
    families=['perforated_grid_plate','honeycomb_screen','multi_step_shaft','groove_stack_pin','multi_thread_shaft','comb_plate','heat_sink_section','stepped_taper_arbor','keyhole_panel','internal_spline_ring','u_channel_section','labyrinth_plate']
    chosen={}
    for family in families:
        candidates=[e for e in index if e['split']==a.split and e['family']==family]
        chosen[max(candidates,key=lambda e:(e['instances'],e['rings']))['id']]=family
    torch.set_num_threads(2);model,ck=load_model(a.checkpoint);manifest=[];thumbs=[]
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    except OSError:font=ImageFont.load_default()
    for meta,c in indexed_records(source,a.split,set(chosen)):
        ids,truth,_=observable_reference_labels(meta,c);result,prob=predict(model,c,instances=ck['instances_trained'])
        instance_labels=np.zeros(len(ids),int)
        leaves=[f for f in result['features'] if f['grouping']!='semantic_parent']
        for i,f in enumerate(leaves):
            if f['status']=='accepted':instance_labels[f['raw_edge_indices']]=1+i%16
        image=Image.new('RGB',(1800,680),'#182027');draw=ImageDraw.Draw(image)
        draw.text((15,10),f"{meta['sample_id']}  {meta['family']}  {a.split}",font=font,fill='white')
        for i,(title,labels) in enumerate(zip(['Reference semantics','Predicted semantics','Predicted instances; gray = abstain'],[truth,prob['classes'],instance_labels])):
            draw.text((15+i*600,45),title,font=font,fill='white')
            draw_contour(image,c['actual_xy_mm'],c['ring_offsets'],labels,(i*600,85,600,520))
        reference=len(np.unique(ids));accepted=sum(f['status']=='accepted' for f in leaves)
        draw.text((15,625),f"{len(c['ring_offsets'])-2} openings | {reference} reference features | {accepted} accepted predictions (counts do not imply correct matching)",font=font,fill='white')
        filename=f"{meta['sample_id']}-{meta['family']}.png";image.save(output/filename)
        manifest.append({'id':meta['sample_id'],'family':meta['family'],'file':filename,'reference_features':reference,'accepted_features':accepted})
        thumbs.append(image.resize((900,340),Image.Resampling.LANCZOS))
        print(filename,flush=True)
    sheet=Image.new('RGB',(1800,340*((len(thumbs)+1)//2)),'#182027')
    for i,t in enumerate(thumbs):sheet.paste(t,((i%2)*900,(i//2)*340))
    sheet.save(output/'overview.png');(output/'manifest.json').write_text(json.dumps({'checkpoint':str(Path(a.checkpoint).resolve()),'split':a.split,'selection':'Maximum realized reference instance count per named family','images':manifest},indent=2))

if __name__=='__main__':main()
