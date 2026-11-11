"""Rebuild training representations from immutable indexed raw observations."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,hashlib,json,multiprocessing,os
import numpy as np
from .data import targets,LABEL_VERSION
from .sampling import prepare_adaptive,LOCAL_VERSION


def one(job):
    source,output,entry=job;source=Path(source);output=Path(output);ident=entry['id']
    directory=output/'shards'/f'{int(ident)//500:05d}';target=directory/(ident+'.npz');receipt=directory/(ident+'.record.json')
    if target.exists() and receipt.exists():return json.loads(receipt.read_text())
    root=source/Path(entry['file']).parent/'sources';meta_path=root/(ident+'.json');raw_path=root/(ident+'.contours.npz')
    meta=json.loads(meta_path.read_text())
    with np.load(raw_path,allow_pickle=False) as a:c={k:a[k] for k in a.files}
    prepared=prepare_adaptive(c,version=LOCAL_VERSION);t=targets(meta,c,prepared)
    directory.mkdir(parents=True,exist_ok=True);temp=directory/(ident+'.partial.npz')
    np.savez_compressed(temp,x=prepared.x,offsets=prepared.offsets,weights=prepared.weights,**t);temp.replace(target)
    row={**entry,'file':str(target.relative_to(output)),'tokens':len(prepared.x),'source_contours':str(raw_path),'source_metadata':str(meta_path)}
    receipt.write_text(json.dumps(row));return row


def build(source,output,workers=4,limit=None):
    source=Path(source).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    from . import data,sampling
    cfg={'source':str(source),'source_index_sha256':hashlib.sha256((source/'index.json').read_bytes()).hexdigest(),
         'source_hash':hashlib.sha256(Path(data.__file__).read_bytes()+Path(sampling.__file__).read_bytes()+Path(__file__).read_bytes()).hexdigest(),
         'sampling':LOCAL_VERSION,'tokens':'adaptive','label_version':LABEL_VERSION,'input_dim':12,'limit':limit}
    p=output/'config.json'
    if p.exists() and json.loads(p.read_text())!=cfg:raise ValueError('Cache config mismatch')
    fd=os.open(output/'.cache.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        p.write_text(json.dumps(cfg,indent=2));entries=json.loads((source/'index.json').read_text())
        if limit:entries=entries[:limit]
        rows=[]
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for row in pool.map(one,((str(source),str(output),e) for e in entries),chunksize=2):
                rows.append(row)
                if len(rows)%250==0:print(f'Prepared {len(rows)}/{len(entries)}',flush=True)
        temp=output/'index.json.partial';temp.write_text(json.dumps(rows,indent=2));temp.replace(output/'index.json')
        return rows
    finally:os.close(fd);(output/'.cache.lock').unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int)
    build(**vars(p.parse_args()))
