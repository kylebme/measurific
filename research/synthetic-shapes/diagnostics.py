"""Create human-review panels from delivered data, not separately regenerated parts."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from inspection_shapes.dataset import iter_samples


def main(output,destination):
    wanted={'threaded_shaft','washer','slotted_bracket'};rows=[]
    palette=np.array([[245,245,245]]+[[int(v) for v in np.random.default_rng(i).integers(60,235,3)] for i in range(1,256)],np.uint8)
    for meta,a in iter_samples(output):
        if meta['family'] not in wanted:continue
        wanted.remove(meta['family'])
        row=Image.new('RGB',(1920,344),'#20252a');d=ImageDraw.Draw(row)
        gray=Image.fromarray(a['image']).convert('RGB')
        region=Image.fromarray(palette[a['region_semantic']])
        overlay=np.repeat(a['image'][:,:,None],3,axis=2)
        labels=a['boundary_semantic'];mask=labels>0;overlay[mask]=palette[labels[mask]]
        overlay[a['anomaly']>0]=[255,30,80]
        contour=a['contours'];px=contour['actual_xy_px'];nom=contour['nominal_xy_px']
        candidates=np.flatnonzero(contour['edge_anomaly_class'])
        if not len(candidates):
            candidates=np.flatnonzero(contour['edge_semantic_bits'] & (1<<10))
        if not len(candidates):candidates=np.arange(len(px)//4)
        index=int(candidates[len(candidates)//2]);cx,cy=px[index]
        box=(int(cx)-48,int(cy)-48,int(cx)+48,int(cy)+48)
        zoom=gray.crop(box).resize((288,288),Image.Resampling.NEAREST)
        zd=ImageDraw.Draw(zoom)
        for lo,hi in zip(contour['ring_offsets'][:-1],contour['ring_offsets'][1:]):
            for points,color in ((nom,(20,215,250)),(px,(255,75,55))):
                coords=(points[lo:hi]-np.array(box[:2]))*3
                # PIL clips segments outside the patch; keep ring closure.
                zd.line([tuple(p) for p in np.vstack([coords,coords[0]])],fill=color,width=1)
        for col,(panel,title) in enumerate([(gray,'Grayscale'),(region,'Filled semantic regions'),
                                           (Image.fromarray(overlay),'Semantic boundary / red defects'),
                                           (zoom,'Zoom: cyan nominal / red actual')]):
            panel.thumbnail((480,288))
            row.paste(panel,(col*480+(480-panel.width)//2,38+(288-panel.height)//2))
            d.text((col*480+10,20),title,fill='white')
        d.text((10,3),f"{meta['sample_id']}  {meta['family']}  severity={meta['imperfections']['severity']}",fill='white')
        rows.append(row)
        if not wanted:break
    sheet=Image.new('RGB',(1920,344*len(rows)))
    for i,row in enumerate(rows):sheet.paste(row,(0,i*344))
    sheet.save(destination)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('destination',type=Path)
    args=p.parse_args();main(args.output,args.destination)
