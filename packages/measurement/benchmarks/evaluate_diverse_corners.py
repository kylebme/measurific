#!/usr/bin/env python3
"""Frozen diverse silhouette benchmark; image estimators never receive truth."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from evaluate_extraction_degradation import (write_json, paths, score_extraction, preview_tile)
from inspection_shapes.geometry import generate
from inspection_shapes.render import random_camera, Camera, render
from inspection_shapes.taxonomy import PARTS
from inspection_measurement import Contours, contours_from_image


def generate_dataset(output):
    if output.exists():
        raise FileExistsError('Use a fresh dataset directory')
    (output/'images').mkdir(parents=True)
    (output/'truth').mkdir()
    cases = []
    for fi, family in enumerate(PARTS):
        for design in range(3):
            seed = 731903+fi*1009+design*101
            severity = [0., .5, 1.5][design]
            part = generate(family, seed, severity=severity)
            camera = random_camera(part, np.random.default_rng(seed+11), 768, 576, 4)
            # Preserve independently sampled pose, scale, distortion and illumination.
            for condition in ['focused', 'blurred']:
                c = Camera(**asdict(camera))
                if condition == 'focused':
                    c.sigma_px = .4
                    c.read_noise_dn, c.electrons_per_dn, c.prnu = .3, 150., .0005
                else:
                    c.sigma_px = [1.6, 1.2, 2.1][design]
                    c.read_noise_dn, c.electrons_per_dn, c.prnu = 1.5, 40., .003
                    c.bright, c.dark = [235., 195., 180.][design], [12., 28., 42.][design]
                stem = f'{family}-d{design}-{condition}'
                arrays, projected, _ = render(part, c, seed+31)
                image = arrays['image']
                polarity = 'light-on-dark' if (fi+design) % 3 == 1 else 'dark-on-light'
                if polarity == 'light-on-dark': image = 255-image
                Image.fromarray(image).save(output/'images'/f'{stem}.png')
                np.savez_compressed(output/'truth'/f'{stem}.npz',
                    actual_xy_px=np.concatenate(projected),
                    ring_offsets=np.r_[0, np.cumsum([len(q) for q in projected])],
                    ring_is_hole=np.array([r.hole for r in part.rings]))
                cases.append(dict(id=stem, family=family, design=design,
                    split='development' if design == 0 else 'heldout',
                    geometry_seed=seed, severity=severity, condition=condition,
                    polarity=polarity, camera=asdict(c), image=f'images/{stem}.png',
                    truth=f'truth/{stem}.npz', image_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
                print(stem, flush=True)
    write_json(output/'manifest.json', dict(protocol_version=1, cases=cases,
        description='12 families x 3 independent designs x 2 optics/noise conditions. Development design 0; designs 1/2 held out. Variable pose, scale, distortion, illumination, contrast, polarity and form error. Continuous-area renderer at 4x sampling.'))


def evaluate_dataset(source, output, split, method):
    output.mkdir(parents=True, exist_ok=True)
    code_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                   for name in ['experimental_corner_fit.py', 'experimental_optical_fit.py',
                                'evaluate_diverse_corners.py', 'evaluate_extraction_degradation.py']}
    run_manifest = dict(source=str(source.resolve()), split=split, method=method, code_sha256=code_hashes,
        complete=False, metric='Bidirectional distance to continuous contours; <=0.1px samples bound missed maxima by 0.05px. Failures remain in the denominator.')
    write_json(output/'manifest.json', run_manifest)
    cases = json.loads((source/'manifest.json').read_text())['cases']
    records = []
    for case in cases:
        if split != 'all' and case['split'] != split: continue
        image = cv2.imread(str(source/case['image']), 0)
        if hashlib.sha256(image.tobytes()).hexdigest() != case['image_sha256']:
            raise ValueError('Image differs from frozen benchmark manifest')
        with np.load(source/case['truth']) as truth:
            truth_paths = paths(truth['actual_xy_px'], truth['ring_offsets'])
            truth_holes = truth['ring_is_hole']
        r = {key: case[key] for key in ['id', 'family', 'split', 'condition', 'polarity']}
        t = time.perf_counter()
        try:
            original, info = contours_from_image(image, polarity=case['polarity'])
            baseline = score_extraction(truth_paths, truth_holes, original, info)
            write_json(output/(case['id']+'.original.json'), original.to_dict())
            baseline.pop('extraction_diagnostics', None)
            r['baseline'] = baseline
            if method == 'local':
                from experimental_corner_fit import recover_pae_corners
                actual, info = recover_pae_corners(image, original, info, polarity=case['polarity'])
            elif method == 'legacy':
                from experimental_optical_fit import extract_for_benchmark
                actual, info = extract_for_benchmark(image, recover_corners=True, polarity=case['polarity'])
            else:
                actual = original
            result = score_extraction(truth_paths, truth_holes, actual, info)
            result.pop('extraction_diagnostics', None)
            r.update(result=result, status=result['status'], recovery=info.get('corner_recovery'))
            write_json(output/(case['id']+'.contours.json'), actual.to_dict())
            preview_tile(image, truth_paths, actual, result, case['id'], size=(500, 375)).save(output/(case['id']+'.png'))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            r.update(status='error', error=str(error))
        r['seconds'] = time.perf_counter()-t
        records.append(r)
        detail = r.get('error')
        if detail is None:
            detail = (r['baseline']['hausdorff_px'], r['result']['hausdorff_px'])
        print(case['id'], r['status'], detail, flush=True)
        write_json(output/'results.json', records)
    write_json(output/'manifest.json', dict(run_manifest, complete=True, cases=len(records)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    gen = sub.add_parser('generate'); gen.add_argument('--output', type=Path, required=True)
    ev = sub.add_parser('evaluate')
    ev.add_argument('--input', type=Path, required=True); ev.add_argument('--output', type=Path, required=True)
    ev.add_argument('--split', choices=['development', 'heldout', 'all'], default='development')
    ev.add_argument('--method', choices=['pae', 'local', 'legacy'], default='local')
    args = parser.parse_args()
    if args.action == 'generate': generate_dataset(args.output)
    else: evaluate_dataset(args.input, args.output, args.split, args.method)


if __name__ == '__main__': main()
