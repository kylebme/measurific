"""Primitive/compound corpus. Raw observations and caches share design splits."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
import multiprocessing
import os
import numpy as np
from inspection_shapes.primitive_labels import labels, LABEL_VERSION, BASE_NAMES, COMPOUND_NAMES
from .corpus_v2 import create_record
from .sampling import prepare_adaptive, VERSION


def targets(metadata, contours, prepared, annotation_override=None):
    annotations, groups = labels(metadata, contours) if annotation_override is None else annotation_override
    W = prepared.overlap
    primary = np.asarray(W@np.eye(len(BASE_NAMES), dtype=np.float32)[annotations['edge_base_class']], np.float32)
    compound = np.asarray(W@np.eye(5, dtype=np.float32)[annotations['edge_compound_class']], np.float32)
    boundary = np.zeros(len(prepared.x), np.float32)
    ids = annotations['edge_base_instance']
    for lo, hi in zip(contours['ring_offsets'][:-1], contours['ring_offsets'][1:]):
        for edge in np.flatnonzero(ids[lo:hi] != np.roll(ids[lo:hi], 1))+lo:
            hit = W[:, edge].nonzero()[0]
            if len(hit): boundary[hit[0]] = 1
    for lo, hi in zip(prepared.offsets[:-1], prepared.offsets[1:]):
        b = boundary[lo:hi].copy()
        boundary[lo:hi] = np.maximum(b, .5*np.maximum(np.roll(b, 1), np.roll(b, -1)))
    # Closed slot loops have unlimited topology-provided instances. Learned queries
    # associate repeated surfaces, including internal tooth profiles.
    repeat = [g for g in groups if g['class_id'] >= 2]
    masks = np.asarray([W@(annotations['edge_compound_instance'] == g['id']).astype(np.float32)
                        for g in repeat], np.float32).reshape(len(repeat), len(prepared.x))
    result = dict(primary=primary, semantic=primary[:, 1:], compound=compound, boundary=boundary,
                  defect=np.asarray(W@np.eye(3, dtype=np.float32)[contours['edge_anomaly_class']], np.float32),
                  instance_masks=masks, instance_classes=np.array([g['class_id'] for g in repeat], np.int64))
    return result, annotations, groups


def write_one(job):
    index, seed, destination = job
    root = Path(destination); ident = f'{index:09d}'
    folder = root/'shards'/f'{index//500:05d}'; folder.mkdir(parents=True, exist_ok=True)
    receipt = folder/(ident+'.record.json')
    if receipt.exists(): return json.loads(receipt.read_text())
    metadata, contours = create_record(index, seed)
    prepared = prepare_adaptive(contours)
    t, annotations, groups = targets(metadata, contours, prepared)
    contours.update(annotations)
    metadata.update(label_version=LABEL_VERSION, base_names=BASE_NAMES,
                    compound_names=COMPOUND_NAMES, compound_instances=groups)
    source = folder/'sources'; source.mkdir(exist_ok=True)
    np.savez_compressed(source/(ident+'.contours.npz'), **contours)
    (source/(ident+'.json')).write_text(json.dumps(metadata))
    filename = folder/(ident+'.npz')
    np.savez_compressed(filename, x=prepared.x, offsets=prepared.offsets, weights=prepared.weights, **t)
    row = dict(id=ident, file=str(filename.relative_to(root)), family=metadata['family'],
               split=metadata['split'], design_id=index, tokens=len(prepared.x),
               rings=len(prepared.offsets)-1, instances=len(groups),
               stress=metadata['stress'], complexity=metadata['complexity'])
    receipt.write_text(json.dumps(row))
    return row


def build(output, count=12000, seed=19237, workers=6):
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    from inspection_shapes import geometry, expanded, primitive_labels, teeth
    from . import corpus, corpus_v2, sampling
    modules = [geometry, expanded, primitive_labels, teeth, corpus, corpus_v2, sampling]
    fingerprint = hashlib.sha256(b''.join(Path(m.__file__).read_bytes() for m in modules)+Path(__file__).read_bytes()).hexdigest()
    config = dict(source='primitive-corpus-v1', seed=seed, source_hash=fingerprint,
                  label_version=LABEL_VERSION, sampling=VERSION, tokens='adaptive',
                  base_names=BASE_NAMES, compound_names=COMPOUND_NAMES)
    config_path = root/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Corpus configuration changed; use a fresh directory')
    lock = root/'.writer.lock'; fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        config_path.write_text(json.dumps(config, indent=2))
        rows=[]
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for i, row in enumerate(pool.map(write_one, ((i,seed,str(root)) for i in range(count)), chunksize=1)):
                rows.append(row)
                if (i+1)%100 == 0: print(f'Generated {i+1}/{count}', flush=True)
        (root/'index.json').write_text(json.dumps(rows, indent=2))
    finally:
        os.close(fd); lock.unlink()
    return rows


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--count',type=int,default=12000)
    p.add_argument('--seed',type=int,default=19237);p.add_argument('--workers',type=int,default=6)
    build(**vars(p.parse_args()))
