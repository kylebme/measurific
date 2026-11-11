"""Bounded-memory, atomic, resumable tar shards; no pickle or per-pixel JSON."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
import hashlib
import json
import os
import multiprocessing
import tarfile
import time
import numpy as np
from PIL import Image, ImageDraw
from .geometry import generate, Part, Ring
from .render import random_camera, render, feature_bits
from .taxonomy import PARTS, taxonomy


def json_bytes(obj):return (json.dumps(obj,sort_keys=True,separators=(",",":"))+"\n").encode()
def npz_bytes(**arrays):
    out=BytesIO();np.savez_compressed(out,**arrays);return out.getvalue()
def png_bytes(array):
    out=BytesIO();Image.fromarray(array).save(out,format="PNG",compress_level=2);return out.getvalue()

def split_for(seed, design_id):
    value=int.from_bytes(hashlib.blake2b(f"split:{seed}:{design_id}".encode(),digest_size=8).digest(),"big")%100
    return "train" if value<80 else "val" if value<90 else "test"


def sample(job):
    index,config=job
    views=config["views_per_design"];design=index//views;view=index%views
    family=list(PARTS)[design%len(PARTS)]
    seeds=np.random.SeedSequence([config["seed"],design]).generate_state(2)
    part=generate(family,int(seeds[0]))
    appearance_seed=int(np.random.SeedSequence([config["seed"],design,view,1]).generate_state(1)[0])
    camera=random_camera(part,np.random.default_rng(appearance_seed),config["width"],config["height"],config["supersample"])
    image_arrays={};pixels=[camera.project(r.actual) for r in part.rings];timing={}
    if not config["contours_only"]:
        image_arrays,pixels,timing=render(part,camera,appearance_seed)
    offsets=np.r_[0,np.cumsum([len(r.nominal) for r in part.rings])].astype(np.int32)
    edge_feature=np.concatenate([r.feature for r in part.rings])
    contours=dict(nominal_xy_mm=np.concatenate([r.nominal for r in part.rings]),
        actual_xy_mm=np.concatenate([r.actual for r in part.rings]),
        actual_xy_px=np.concatenate(pixels),nominal_xy_px=np.concatenate([camera.project(r.nominal) for r in part.rings]),
        ring_offsets=offsets,ring_is_hole=np.array([r.hole for r in part.rings]),
        edge_feature_id=edge_feature,edge_primitive_id=np.concatenate([r.primitive for r in part.rings]),
        edge_semantic_bits=feature_bits(part)[edge_feature],
        edge_anomaly_class=np.concatenate([r.defect for r in part.rings]),
        edge_anomaly_instance=np.concatenate([r.anomaly_instance for r in part.rings]))
    # Full hierarchy in metadata + primary edge ID is an exact compact multi-label encoding.
    meta=dict(schema_version=1,sample_id=f"{index:09d}",design_id=design,view_id=view,
              split=split_for(config["seed"],design),family=family,part_taxonomy=PARTS[family],
              units="mm",geometry_seed=int(seeds[0]),appearance_seed=appearance_seed,
              camera=asdict(camera),features=part.features,primitives=part.primitives,
              relationships=part.relationships,patterns=part.patterns,imperfections=part.imperfections,
              anomalies=part.anomalies,backend="cpu",
              annotation_policy="Nominal semantic identity on actual edges; material-exterior normals; region plus 3px nearest-edge band; ancestor bits ORed; distance ties resolved by scipy EDT",
              projection="x right, y down; integer pixel centers; mm x right, y up; radial+tangential forward distortion")
    files={"json":json_bytes(meta),"contours.npz":npz_bytes(**contours)}
    for name,arr in image_arrays.items():
        if name=="semantic_bits":files["semantic_bits.npz"]=npz_bytes(bits=arr)
        else:files[f"{name}.png"]=png_bytes(arr)
    feature_counts={}
    for f in part.features:feature_counts[f["semantic"]]=feature_counts.get(f["semantic"],0)+1
    stats=dict(index=index,design_id=design,family=family,split=meta["split"],feature_counts=feature_counts,
               anomaly_count=len(part.anomalies),severity=part.imperfections["severity"],
               attempts=part.imperfections["rejection_attempts"],timing=timing)
    return files,stats


def add_tar(tar,name,data):
    info=tarfile.TarInfo(name);info.size=len(data);info.mtime=0;info.mode=0o644
    tar.addfile(info,BytesIO(data))


def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()


def write_dataset(output, count=1200, seed=20260905, workers=4, shard_size=100,
                  width=1920,height=1080,supersample=2,views_per_design=1,
                  contours_only=False):
    if count<1 or workers<1 or shard_size<1 or views_per_design<1 or min(width,height)<64 or supersample<1:
        raise ValueError("Invalid count, workers, shard size, views, image size, or supersampling")
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    source_hash=hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        source_hash.update(source.name.encode());source_hash.update(source.read_bytes())
    config=dict(schema_version=1,generator_sha256=source_hash.hexdigest(),seed=seed,width=width,height=height,supersample=supersample,
                views_per_design=views_per_design,contours_only=contours_only,backend="cpu",shard_size=shard_size)
    if (output/"manifest.json").exists():
        old_manifest=json.loads((output/"manifest.json").read_text())
        if any(s["end"]>count for s in old_manifest["shards"]):
            raise ValueError("Cannot shrink a dataset; requested count is below existing sample count")
    config_path=output/"config.json"
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        raise ValueError("Existing dataset configuration differs; use a new output directory")
    # Exclusive lock prevents simultaneous writers, including runs with different configs.
    lock=output/".writer.lock"
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        os.write(fd,str(os.getpid()).encode());os.close(fd)
        config_path.write_bytes(json_bytes(config));(output/"taxonomy.json").write_bytes(json_bytes(taxonomy()))
        manifest=[];start=time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
            for shard_start in range(0,count,shard_size):
                end=min(count,shard_start+shard_size)
                stem=f"shard-{shard_start:09d}-{end-1:09d}"
                dest=output/(stem+".tar");receipt=output/(stem+".json")
                # Never overlap ranges when extending a previously partial final shard.
                for old in output.glob("shard-*.tar"):
                    a,b=map(int,old.stem.split("-")[1:])
                    if old!=dest and a<end and b>=shard_start:
                        raise ValueError("Extension overlaps a partial shard; extend at a full shard boundary")
                if dest.exists() and receipt.exists():
                    entry=json.loads(receipt.read_text())
                    if digest(dest)!=entry["sha256"]:raise ValueError(f"Corrupt shard: {dest}")
                    manifest.append(entry);print(f"Verified existing {dest.name}",flush=True);continue
                temporary=output/(stem+".tar.partial")
                stats=[]
                with tarfile.open(temporary,"w") as tar:
                    # Submit only 2*workers samples at a time: bounded on Python 3.11+.
                    for batch in range(shard_start,end,2*workers):
                        jobs=[(i,config) for i in range(batch,min(end,batch+2*workers))]
                        for files,stat in pool.map(sample,jobs,chunksize=1):
                            prefix=f'{stat["split"]}/{stat["index"]:09d}'
                            for suffix,data in files.items():add_tar(tar,f"{prefix}.{suffix}",data)
                            stats.append(stat)
                os.replace(temporary,dest)
                entry=dict(file=dest.name,start=shard_start,end=end,count=end-shard_start,
                           bytes=dest.stat().st_size,sha256=digest(dest),samples=stats)
                receipt_tmp=receipt.with_suffix(".json.partial");receipt_tmp.write_bytes(json_bytes(entry));os.replace(receipt_tmp,receipt)
                manifest.append(entry)
                manifest_tmp=output/"manifest.json.partial"
                manifest_tmp.write_bytes(json_bytes(dict(config=config,shards=manifest)))
                os.replace(manifest_tmp,output/"manifest.json")
                elapsed=time.perf_counter()-start
                print(f"Committed {dest.name}: {end}/{count}, {elapsed:.1f}s elapsed",flush=True)
        # Includes verified shards after an entirely resumed run.
        manifest_tmp=output/"manifest.json.partial"
        manifest_tmp.write_bytes(json_bytes(dict(config=config,shards=manifest)))
        os.replace(manifest_tmp,output/"manifest.json")
        return manifest
    finally:
        lock.unlink(missing_ok=True)


def iter_samples(output,split=None):
    """Sequential training reader: yields (metadata, arrays), bounded to one sample.

PNG/NPZ are decoded without pickle. Iterate once per epoch or shuffle shards
externally; split filtering occurs before decoding large arrays.
"""
    output=Path(output)
    manifest=json.loads((output/"manifest.json").read_text())
    for shard in manifest["shards"]:
        with tarfile.open(output/shard["file"],"r|") as tar:
            current=None;meta=None;arrays={}
            for member in tar:
                prefix,suffix=member.name.split(".",1)
                if split and prefix.split("/")[0]!=split:continue
                if prefix!=current:
                    if meta is not None:yield meta,arrays
                    current=prefix;meta=None;arrays={}
                data=tar.extractfile(member).read()
                if suffix=="json":meta=json.loads(data)
                elif suffix.endswith("npz"):
                    with np.load(BytesIO(data),allow_pickle=False) as a:
                        arrays[suffix[:-4]]={k:a[k] for k in a.files}
                else:arrays[suffix[:-4]]=np.array(Image.open(BytesIO(data)))
            if meta is not None:yield meta,arrays


def preview(output, destination, limit=24):
    tilew,tileh=480,318;tiles=[]
    palette=np.array([[0,0,0]]+[[int(x) for x in np.random.default_rng(i).integers(65,255,3)] for i in range(1,256)],np.uint8)
    for meta,a in iter_samples(output):
        if "image" not in a:continue
        im=np.repeat(a["image"][:,:,None],3,axis=2)
        labels=a["boundary_semantic"]
        im[labels>0]=palette[labels[labels>0]]
        anomaly=a["anomaly"]>0;im[anomaly]=[255,30,70]
        tile=Image.new("RGB",(tilew,tileh),"#20252a")
        img=Image.fromarray(im);img.thumbnail((tilew,tileh-48));tile.paste(img,((tilew-img.width)//2,28))
        d=ImageDraw.Draw(tile);d.text((10,8),f'{meta["sample_id"]} {meta["family"]}',fill="white")
        d.text((10,tileh-17),f'{meta["split"]} | severity {meta["imperfections"]["severity"]} | red: defects',fill="white")
        tiles.append(tile)
        if len(tiles)>=limit:break
    if not tiles:raise ValueError("No rendered samples to preview")
    cols=4;rows=(len(tiles)+cols-1)//cols
    sheet=Image.new("RGB",(cols*tilew,rows*tileh),"#20252a")
    for i,t in enumerate(tiles):sheet.paste(t,((i%cols)*tilew,(i//cols)*tileh))
    sheet.save(destination)


def part_from_record(metadata, contours):
    """Reconstruct the stored geometry for later CPU rendering without resampling.

Use this with a contour-only record or the delivered image dataset. Returned
arrays are independent copies, safe for later geometric experimentation.
"""
    from copy import deepcopy
    p=Part(metadata['family'],features=deepcopy(metadata['features']),
           primitives=deepcopy(metadata['primitives']),relationships=deepcopy(metadata['relationships']),
           patterns=deepcopy(metadata['patterns']),imperfections=deepcopy(metadata['imperfections']),
           anomalies=deepcopy(metadata['anomalies']))
    for i,(lo,hi) in enumerate(zip(contours['ring_offsets'][:-1],contours['ring_offsets'][1:])):
        p.rings.append(Ring(nominal=contours['nominal_xy_mm'][lo:hi].copy(),
                            feature=contours['edge_feature_id'][lo:hi].copy(),
                            primitive=contours['edge_primitive_id'][lo:hi].copy(),
                            hole=bool(contours['ring_is_hole'][i]),
                            actual=contours['actual_xy_mm'][lo:hi].copy(),
                            defect=contours['edge_anomaly_class'][lo:hi].copy(),
                            anomaly_instance=contours['edge_anomaly_instance'][lo:hi].copy()))
    return p
