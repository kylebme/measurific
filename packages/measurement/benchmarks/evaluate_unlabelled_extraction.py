#!/usr/bin/env python3
"""Exercise default extraction on images without ground truth or sidecars."""
import argparse
import time
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

from evaluate_extraction_degradation import write_json
from inspection_measurement import contours_from_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=['default', 'experimental-local-corners', 'experimental-optical', 'experimental-corners'], default='default')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records, tiles = [], []
    images = sorted(p for p in args.input.iterdir()
                    if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'})
    if not images:
        parser.error('No supported images in input directory')
    for path in images:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        contours = None
        start = time.perf_counter()
        try:
            if args.method == 'default':
                contours, diagnostics = contours_from_image(image)
            elif args.method == 'experimental-local-corners':
                from experimental_corner_fit import recover_pae_corners
                contours, diagnostics = contours_from_image(image)
                contours, diagnostics = recover_pae_corners(image, contours, diagnostics)
            else:
                from experimental_optical_fit import extract_for_benchmark
                contours, diagnostics = extract_for_benchmark(image, recover_corners=args.method == 'experimental-corners')
            record = {'image': path.name, 'status': 'ok', 'diagnostics': diagnostics}
            write_json(args.output/(path.stem+'.contours.json'), contours.to_dict())
        except (ValueError, TypeError) as error:
            record = {'image': path.name, 'status': 'error', 'error': str(error)}
        record['seconds'] = time.perf_counter()-start
        records.append(record)
        print(path.name, record['status'], record.get('error', ''), flush=True)
        write_json(args.output/'results.json', records)
        tile = Image.new('RGB', (400, 330), '#141b22')
        draw = ImageDraw.Draw(tile)
        draw.text((5, 5), path.name+' '+record['status'], fill='white')
        if image is not None:
            preview = Image.fromarray(image).convert('RGB')
            preview.thumbnail((400, 300))
            tile.paste(preview, (0, 30))
            scale = preview.width/image.shape[1]
            if contours is not None:
                for i in range(len(contours.closed)):
                    q = [(x*scale, y*scale+30) for x, y in contours.path(i)]
                    draw.line(q+[q[0]], fill='#ff4fc8', width=1)
        tiles.append(tile)
    sheet = Image.new('RGB', (1600, 330*((len(tiles)+3)//4)), '#141b22')
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 4)*400, (i//4)*330))
    sheet.save(args.output/'contact-sheet.png')
    write_json(args.output/'manifest.json', {'source': str(args.input.resolve()),
               'method': args.method, 'ground_truth_available': False})
    passed = sum(r['status'] == 'ok' for r in records)
    (args.output/'README.md').write_text(
        '# Unlabelled silhouette extraction\n\n'
        f'{passed}/{len(records)} images produced valid closed contours using `{args.method}`. '
        'This checks extraction and topology validity; without ground truth '
        'it does not establish coordinate accuracy or completeness of recovered features.\n\n'
        'Magenta overlays are in `contact-sheet.png`. Contour files are stored here, '
        'separately from input images, so GUI loading still exercises extraction.\n')


if __name__ == '__main__':
    main()
