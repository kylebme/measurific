"""Rebuild primitive targets from retained analytic provenance; observations unchanged."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
import multiprocessing
import os
import numpy as np
from inspection_shapes import primitive_labels
from inspection_shapes import repair_primitives
from . import corpus_primitives
from .sampling import prepare_adaptive, VERSION


def write_one(job):
    entry,source,destination=job
    source=Path(source);destination=Path(destination)
    original=source/entry['file'];raw_path=original.parent/'sources'/(entry['id']+'.contours.npz')
    metadata=json.loads(raw_path.with_name(entry['id']+'.json').read_text())
    with np.load(raw_path) as a:raw={k:a[k] for k in a.files}
    repaired=repair_primitives.repair(metadata,raw)
    prepared=prepare_adaptive(raw)
    targets,annotations,groups=corpus_primitives.targets(metadata,raw,prepared)
    raw.update(annotations);metadata['compound_instances']=groups
    metadata.update(label_version=primitive_labels.LABEL_VERSION,
                    base_names=primitive_labels.BASE_NAMES,
                    compound_names=primitive_labels.COMPOUND_NAMES,
                    freeform_policy='complete ring -> circle; near-straight partial span -> line; otherwise arc')
    path=destination/entry['file'];path.parent.mkdir(parents=True,exist_ok=True)
    out=path.parent/'sources';out.mkdir(exist_ok=True)
    np.savez_compressed(out/raw_path.name,**raw)
    (out/(entry['id']+'.json')).write_text(json.dumps(metadata))
    np.savez_compressed(path,x=prepared.x,offsets=prepared.offsets,weights=prepared.weights,**targets)
    return {**entry,'repaired_rings':repaired}


def build(source,output,workers=6):
    source=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists():raise ValueError('Use a new recache directory')
    config=json.loads((source/'config.json').read_text())
    config.update(source='primitive-three-base-relabel-v2',parent_source=str(source.resolve()),
        parent_source_hash=config['source_hash'],sampling=VERSION,
        label_version=primitive_labels.LABEL_VERSION,
        base_names=primitive_labels.BASE_NAMES,
        compound_names=primitive_labels.COMPOUND_NAMES,
        freeform_policy='complete ring -> circle; max chord deviation <= 0.2% -> line; otherwise arc',
        source_hash=hashlib.sha256(Path(__file__).read_bytes()+Path(primitive_labels.__file__).read_bytes()+Path(repair_primitives.__file__).read_bytes()+Path(corpus_primitives.__file__).read_bytes()).hexdigest())
    lock=out/'.writer.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        (out/'config.json').write_text(json.dumps(config,indent=2))
        entries=json.loads((source/'index.json').read_text());rows=[]
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for row in pool.map(write_one,((e,str(source),str(out)) for e in entries),chunksize=1):
                rows.append(row)
                if len(rows)%250==0:print(f'Recached {len(rows)}/{len(entries)}',flush=True)
        (out/'index.json').write_text(json.dumps(rows,indent=2))
    finally:os.close(fd);lock.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');p.add_argument('--workers',type=int,default=6)
    build(**vars(p.parse_args()))
