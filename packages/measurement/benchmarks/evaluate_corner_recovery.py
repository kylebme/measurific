#!/usr/bin/env python3
"""Offline corner experiment using saved optical fits; never reads GUI sidecars."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from evaluate_extraction_degradation import (write_json, paths, score_extraction,
                                           report_markdown, preview_tile, make_contact_sheet)
from experimental_corner_fit import repair_corners
from inspection_measurement import Contours


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--optical-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cache_manifest = json.loads((args.optical_cache/'manifest.json').read_text())
    if Path(cache_manifest['source']).resolve() != args.input.resolve():
        parser.error('Cached optical fits refer to a different image dataset')
    records = json.loads((args.optical_cache/'results.json').read_text())
    families = list(dict.fromkeys(r['family'] for r in records))
    args.output.mkdir(parents=True, exist_ok=True)
    results, tiles = [], []
    for record in records:
        path = args.input/record['image']
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        truth = np.load(args.input/record['truth'])
        truth_paths = paths(truth['actual_xy_px'], truth['ring_offsets'])
        contours = None
        try:
            original = Contours.from_dict(json.loads(
                (args.optical_cache/(path.stem+'.contours.json')).read_text()))
            diagnostics = dict(record['extraction_diagnostics'])
            sigma, noise = diagnostics['blur_sigma_px'], diagnostics['noise_sigma_dn']
            rings, counts = zip(*(repair_corners(original.path(i), image, sigma, noise)
                                  for i in range(len(original.closed))))
            contours = Contours(np.concatenate(rings), np.r_[0, np.cumsum([len(q) for q in rings])])
            diagnostics.update(experimental=True, corner_corrections=sum(counts))
            # Reject any change of nesting or intersections between rings.
            from shapely.geometry import Polygon
            old, new = ([Polygon(q.path(i)) for i in range(len(q.closed))]
                        for q in (original, contours))
            for i in range(len(new)):
                for j in range(i):
                    if (new[i].boundary.intersects(new[j].boundary) or
                            new[i].contains(new[j]) != old[i].contains(old[j]) or
                            new[j].contains(new[i]) != old[j].contains(old[i])):
                        raise ValueError('Experimental corner correction changed topology')
            result = score_extraction(truth_paths, truth['ring_is_hole'], contours, diagnostics)
            write_json(args.output/(path.stem+'.contours.json'), contours.to_dict())
        except (ValueError, KeyError, FileNotFoundError) as error:
            result = dict(status='extraction_error', error=str(error))
        result.update({k: record[k] for k in ['family', 'image', 'truth', 'noise', 'blur_sigma_px']})
        results.append(result)
        print(path.name, result['status'], result.get('hausdorff_px', result.get('error')), flush=True)
        condition = ['clean', 'low', 'nominal'].index(result['noise'])*4+[.35, .8, 1.3, 1.8].index(result['blur_sigma_px'])
        tiles.append((result['family'], condition, preview_tile(image, truth_paths, contours, result, path.stem)))
        write_json(args.output/'results.json', results)
    write_json(args.output/'manifest.json', dict(source=str(args.input.resolve()),
               optical_cache=str(args.optical_cache.resolve()), method='experimental-corners',
               sampled_maximum_underestimate_bound_px=.05))
    make_contact_sheet(tiles, args.output/'contact-sheet.png', families)
    (args.output/'README.md').write_text(report_markdown(results, families,
                                        source=args.input.resolve(), method='experimental-corners'))


if __name__ == '__main__':
    main()
