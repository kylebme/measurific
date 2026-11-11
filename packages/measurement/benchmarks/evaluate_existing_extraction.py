#!/usr/bin/env python3
"""Re-score an existing, immutable image/truth sweep with a selected extractor."""
import argparse
import json
from pathlib import Path
import time

import cv2

from evaluate_extraction_degradation import (np, paths, score_extraction, write_json,
                                           preview_tile, report_markdown, make_contact_sheet)
from inspection_measurement import contours_from_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=['default', 'experimental-optical', 'experimental-corners'], default='default')
    parser.add_argument('--pattern', default='*')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results, tiles = [], []
    manifest = json.loads((args.input/'manifest.json').read_text())
    families = [f['family'] for f in manifest['families']]
    records = json.loads((args.input/'results.json').read_text())
    for record in records:
        path = args.input/record['image']
        if not path.match(args.pattern): continue
        image = cv2.imread(str(path), 0)
        truth = np.load(args.input/record['truth'])
        truth_paths = paths(truth['actual_xy_px'], truth['ring_offsets'])
        c = None
        start = time.perf_counter()
        try:
            if args.method.startswith('experimental-'):
                from experimental_optical_fit import extract_for_benchmark
                c, info = extract_for_benchmark(image, recover_corners=args.method == 'experimental-corners')
            else:
                c, info = contours_from_image(image)
            extraction_seconds = time.perf_counter()-start
            r = score_extraction(truth_paths, truth['ring_is_hole'], c, info)
            # Extracted observations only; no sidecars beside benchmark images.
            write_json(args.output/(path.stem+'.contours.json'), c.to_dict())
        except Exception as error:
            r = {'status': 'extraction_error', 'error': f'{type(error).__name__}: {error}'}
            extraction_seconds = time.perf_counter()-start
        r.update({k: record[k] for k in ['family', 'image', 'truth', 'blur_sigma_px', 'noise']})
        r['extraction_seconds'] = extraction_seconds
        results.append(r)
        print(path.name, r['status'], r.get('error', round(r.get('hausdorff_px', 0), 4)), flush=True)
        write_json(args.output/'results.json', results)
        condition = list(['clean', 'low', 'nominal']).index(r['noise'])*4+list([.35, .8, 1.3, 1.8]).index(r['blur_sigma_px'])
        tiles.append((r['family'], condition, preview_tile(image, truth_paths, c, r, path.stem)))
    write_json(args.output/'manifest.json', {'source': str(args.input.resolve()), 'method': args.method,
               'metric': 'Distances to continuous segments, <=.1px arclength sampling in both directions.',
               'sampled_maximum_underestimate_bound_px': .05})
    if tiles: make_contact_sheet(tiles, args.output/'contact-sheet.png', families)
    (args.output/'README.md').write_text(report_markdown(results, families,
                                        source=args.input.resolve(), method=args.method))

if __name__ == '__main__': main()
