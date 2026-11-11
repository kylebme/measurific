"""Exhaustive, CPU-only contact sheets of the committed raw contour corpus."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import multiprocessing
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .data import NAMES, remap_path

COLORS = ['#64748b','#334155','#2563eb','#dc2626','#22804b','#be185d',
          '#0891b2','#7c3aed','#a16207','#c2410c','#a21caf','#4d7c0f',
          '#4f46e5','#0f766e','#475569','#9333ea','#b91c1c']
TILE_W,TILE_H,COLS,ROWS = 800,390,4,6
HEADER,FOOTER = 110,160


def endpoint_indices(instance_ids, offsets):
    """Raw vertices at instance transitions, including same-class joins and wrap.

    A whole-ring circle has no segment endpoints; its storage seam is not shown.
    """
    return np.concatenate([lo+np.flatnonzero(instance_ids[lo:hi] != np.roll(instance_ids[lo:hi],1))
                           for lo,hi in zip(offsets[:-1],offsets[1:])])


def page_groups(entries, per_page=24):
    """Balance the final pages so there is no tiny last sheet."""
    if not entries:return []
    pages=math.ceil(len(entries)/per_page);base,extra=divmod(len(entries),pages)
    groups=[];at=0
    for i in range(pages):
        n=base+(i<extra);groups.append(entries[at:at+n]);at+=n
    return groups


def source_paths(root, entry):
    if 'source_contours' in entry:return remap_path(entry['source_metadata']),remap_path(entry['source_contours'])
    directory=remap_path(Path(root)/Path(entry['file'])).parent/'sources'
    return directory/(entry['id']+'.json'),directory/(entry['id']+'.contours.npz')


def font(size, bold=False):
    name='DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    try:return ImageFont.truetype(name,size)
    except OSError:
        return ImageFont.truetype('/System/Library/Fonts/Supplemental/'+('Arial Bold.ttf' if bold else 'Arial.ttf'),size)


def render_page(job):
    root,output,page,total,entries,scale=job
    config=json.loads((Path(root)/'config.json').read_text())
    primitive=config.get('label_version','').startswith('primitives-compounds-')
    names=config['base_names'] if primitive else NAMES
    colors=['#64748b','#168aad','#ef8354','#8c5ec0','#36a657'] if primitive else COLORS
    destination=Path(output)/'sheets'/f'sheet-{page:04d}.png'
    receipt=destination.with_suffix('.json')
    expected=[e['id'] for e in entries]
    if destination.exists() and receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved['sample_ids']==expected and hashlib.sha256(destination.read_bytes()).hexdigest()==saved['sha256']:
            return saved
        raise ValueError(f'Existing sheet does not match: {destination}')
    size=(COLS*TILE_W,HEADER+ROWS*TILE_H+FOOTER)
    im=Image.new('RGB',(size[0]*scale,size[1]*scale),'white');draw=ImageDraw.Draw(im)
    fonts={n:font(n*scale,bold=n in (22,32)) for n in (18,20,22,24,32)}
    def text(x,y,message,n=20,fill='#334155'):
        draw.text((x*scale,y*scale),message,font=fonts[n],fill=fill)
    def line(points,fill,width=1):
        draw.line([(x*scale,y*scale) for x,y in points],fill=fill,width=width*scale)
    text(26,15,'Contour corpus  /  Reference '+('primitive' if primitive else 'semantic')+' labels',32)
    text(26,63,f'Sheet {page:,} of {total:,}  |  Samples {expected[0]}–{expected[-1]}  |  {len(entries)} parts',20)
    text(2040,66,'Original observed contours · Each part scaled to fit',20)
    for tile,entry in enumerate(entries):
        meta_path,contour_path=source_paths(root,entry)
        meta=json.loads(meta_path.read_text())
        if str(meta['sample_id'])!=entry['id'] or meta['family']!=entry['family'] or meta['split']!=entry['split']:
            raise ValueError(f'Metadata/index mismatch: {meta_path}')
        with np.load(contour_path,allow_pickle=False) as arrays:
            xy=arrays['actual_xy_mm'];offsets=arrays['ring_offsets'];ids=arrays['edge_feature_id']
            if primitive:
                labels=arrays['edge_base_class'];ids=arrays['edge_base_instance']
        if not primitive:
            lookup=np.array([0]+[f['semantic_id'] for f in meta['features']]);labels=lookup[ids]
        x=(tile%COLS)*TILE_W;y=HEADER+(tile//COLS)*TILE_H
        line([(x+12,y+2),(x+TILE_W-12,y+2)],'#dbe2ea')
        text(x+22,y+14,f'{tile+1:02d}   {entry["id"]}   {entry["family"]}',22)
        extent=np.ptp(xy,axis=0)
        text(x+22,y+47,f'{entry["split"]}  ·  {len(offsets)-1} rings  ·  {extent[0]:.2f} × {extent[1]:.2f} mm',18,'#64748b')
        factor=min((TILE_W-70)/max(extent[0],1e-9),(TILE_H-110)/max(extent[1],1e-9))
        q=(xy-(xy.min(0)+xy.max(0))/2)*[factor,-factor]+[x+TILE_W/2,y+82+(TILE_H-100)/2]
        q*=scale
        for lo,hi in zip(offsets[:-1],offsets[1:]):
            ring=q[lo:hi];classes=labels[lo:hi];closed=np.vstack([ring,ring[0]])
            cuts=np.r_[0,np.flatnonzero(classes[1:]!=classes[:-1])+1,len(ring)]
            for a,b in zip(cuts[:-1],cuts[1:]):
                draw.line([tuple(v) for v in closed[a:b+1]],fill=colors[int(classes[a])],width=2*scale,joint='curve')
        for vx,vy in q[endpoint_indices(ids,offsets)]:
            r=3*scale
            draw.ellipse((vx-r,vy-r,vx+r,vy+r),fill='white',outline='#111827',width=scale)
    footer=HEADER+ROWS*TILE_H
    line([(20,footer),(size[0]-20,footer)],'#cbd5e1')
    for cls,name in enumerate(names):
        x=26+(cls%6)*525;y=footer+14+(cls//6)*34
        line([(x,y+12),(x+36,y+12)],colors[cls],3)
        text(x+48,y,name,20)
    text(26,footer+86,'White dots = labeled segment endpoints (including same-class joins). Whole-ring circles have no endpoints.',18,'#334155')
    text(26,footer+123,'Colors show primary reference classes. Form errors, edge roughness and defects are retained. No model predictions or idealized geometry.',18,'#64748b')
    if scale!=1:im=im.resize(size,Image.Resampling.LANCZOS)
    temp=destination.with_suffix('.partial.png');im.save(temp,compress_level=3);temp.replace(destination)
    result={'page':page,'file':str(destination.relative_to(output)),'sample_ids':expected,
            'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'width':size[0],'height':size[1]}
    receipt.write_text(json.dumps(result,indent=2))
    return result


def write_browser(output, pages, entries, locations):
    # Embedded metadata works directly from file:// without fetch/CORS or a server.
    lookup={e['id']:[*locations[e['id']],e['family'],e['split']] for e in entries}
    content='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Contour contact sheets</title>
<style>body{font:16px system-ui;margin:0;background:#edf1f5;color:#182334}header{position:sticky;top:0;background:white;padding:16px 24px;border-bottom:1px solid #cbd5e1;z-index:2}h1{font-size:22px;margin:0 0 10px}button,input{font:inherit;padding:7px;margin-right:7px}button{cursor:pointer}a{color:#2563eb}#sheet{display:block;max-width:100%;margin:18px auto;background:white}#message{margin-top:10px}#status{margin-left:10px}main{padding:0 20px}details{margin-top:12px}#review{padding:12px;background:#f1f5f9}#review input{width:360px}</style>
<header><h1>Selected contour training parts · Reference labels and endpoints</h1>
<button id="prev">← Previous</button><label>Sheet <input id="page" type="number" min="1" style="width:90px"></label>
<button id="next">Next →</button><span id="status"></span> · <a id="full" target="_blank">Open full-resolution PNG</a>
<div style="margin-top:10px"><label>Find sample ID <input id="sample" placeholder="e.g. 000020231" style="width:160px"></label><button id="find">Find</button>
<a href="review.csv">Download review worksheet</a> · <a href="manifest.json">Coverage manifest</a></div>
<div id="message">Every selected sample appears exactly once. White dots mark labeled endpoints. Arrow keys change sheets when not typing.</div>
<details><summary>Review notes (saved in this browser; export a backup)</summary><div id="review"><input id="noteId" placeholder="Sample ID"><input id="note" placeholder="Your review note"><button id="save">Save note</button><button id="export">Export notes CSV</button><span id="saved"></span></div></details>
</header><main><img id="sheet" alt="Contour contact sheet"></main><script>
const pages=__PAGES__, samples=__SAMPLES__;let current=1;
const $=id=>document.getElementById(id);
function show(n){current=Math.max(1,Math.min(pages.length,Number(n)||1));const p=pages[current-1];$('page').value=current;$('page').max=pages.length;$('sheet').src=p.file;$('full').href=p.file;$('status').textContent=`of ${pages.length.toLocaleString()} — ${p.sample_ids.length} parts`;$('prev').disabled=current===1;$('next').disabled=current===pages.length;history.replaceState(null,'','#'+current);}
$('prev').onclick=()=>show(current-1);$('next').onclick=()=>show(current+1);$('page').onchange=()=>show($('page').value);
function find(){const id=$('sample').value.trim().padStart(9,'0'),r=samples[id];if(!r){$('message').textContent='Sample not found.';return;}show(r[0]);$('message').textContent=`Sample ${id}: tile ${r[1]}, ${r[2]}, ${r[3]}`;$('noteId').value=id;}
$('find').onclick=find;$('sample').onkeydown=e=>{if(e.key==='Enter')find();};
document.onkeydown=e=>{if(['INPUT','TEXTAREA'].includes(e.target.tagName))return;if(e.key==='ArrowLeft')show(current-1);if(e.key==='ArrowRight')show(current+1);};
const key='inspection-contour-review-'+__HASH__;let notes={};try{notes=JSON.parse(localStorage.getItem(key)||'{}');}catch(e){}
$('save').onclick=()=>{const id=$('noteId').value.trim().padStart(9,'0');if(!samples[id]){$('saved').textContent='Enter a valid sample ID.';return;}notes[id]=$('note').value;try{localStorage.setItem(key,JSON.stringify(notes));$('saved').textContent=`Saved ${Object.keys(notes).length} notes in this browser.`;}catch(e){$('saved').textContent='Browser storage unavailable. Export notes before closing.';}};
$('export').onclick=()=>{const quote=s=>'"'+String(s).replaceAll('"','""')+'"';const rows=[['sample_id','sheet','tile','family','split','note'],...Object.entries(notes).map(([id,n])=>[id,...samples[id],n])];const blob=new Blob([rows.map(r=>r.map(quote).join(',')).join('\\n')],{type:'text/csv'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='contour-review-notes.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
show(location.hash.slice(1));</script></html>'''
    content=content.replace('__PAGES__',json.dumps(pages)).replace('__SAMPLES__',json.dumps(lookup)).replace('__HASH__',json.dumps(hashlib.sha256(json.dumps(lookup).encode()).hexdigest()[:16]))
    (output/'index.html').write_text(content)


def build(source,output,workers=4,limit=None,antialias=2):
    source=Path(source).resolve();output=Path(output).resolve()
    entries=json.loads((source/'index.json').read_text())
    entries=sorted(entries,key=lambda e:e['id'])
    if limit is not None:entries=entries[:limit]
    if not entries or len({e['id'] for e in entries})!=len(entries):raise ValueError('Expected nonempty unique sample IDs')
    config={'source':str(source),'index_sha256':hashlib.sha256((source/'index.json').read_bytes()).hexdigest(),
            'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'limit':limit,'antialias':antialias,
            'per_page':24,'samples':len(entries),'labels':'primary reference semantics; observed raw contours'}
    output.mkdir(parents=True,exist_ok=True);(output/'sheets').mkdir(exist_ok=True)
    conf=output/'config.json'
    if conf.exists() and json.loads(conf.read_text())!=config:raise ValueError('Output configuration mismatch; use a new directory')
    conf.write_text(json.dumps(config,indent=2));groups=page_groups(entries)
    jobs=[(str(source),str(output),i+1,len(groups),group,antialias) for i,group in enumerate(groups)]
    results=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for result in pool.map(render_page,jobs,chunksize=1):
            results.append(result)
            if len(results)%25==0 or len(results)==len(jobs):print(f'Rendered {len(results)}/{len(jobs)} sheets',flush=True)
    covered=[ident for r in results for ident in r['sample_ids']]
    if covered!=[e['id'] for e in entries]:raise RuntimeError('Coverage mismatch')
    locations={ident:(r['page'],tile+1) for r in results for tile,ident in enumerate(r['sample_ids'])}
    with (output/'review.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['sample_id','sheet','tile','family','split','source_contours','status','notes'])
        for e in entries:writer.writerow([e['id'],*locations[e['id']],e['family'],e['split'],str(source_paths(source,e)[1]),'',''])
    manifest={'samples':len(entries),'sheets':len(results),'coverage':'every selected sample exactly once','pages':results}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2));write_browser(output,results,entries,locations)
    (output/'README.txt').write_text(f'{len(entries):,} observed contours on {len(results):,} PNG contact sheets.\nOpen index.html for sample lookup, previous/next navigation, full-resolution images and optional review notes.\nReference semantic colors; not model predictions. Each part is scaled independently.\nAll indexed samples occur exactly once; last pages are balanced to avoid a nearly empty sheet.\nreview.csv maps every sample to its sheet/tile and source. Browser notes must be exported to preserve a portable backup.\nmanifest.json records coverage and SHA256 for every PNG. Rendering can resume with the same command/configuration.\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('output')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int);p.add_argument('--antialias',type=int,choices=(1,2),default=2)
    a=p.parse_args()
    if a.workers<1 or (a.limit is not None and a.limit<1):p.error('workers and limit must be positive')
    build(**vars(a))

if __name__=='__main__':main()
