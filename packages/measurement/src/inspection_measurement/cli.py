"""Programmatic inputs also have a small file-based command-line wrapper."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from . import Contours, inspect_image


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('request', type=Path)
    parser.add_argument('output', type=Path, help='Output directory')
    parser.add_argument('--contours', type=Path, help='Saved native contour JSON (otherwise extract with the edge model)')
    parser.add_argument('--polarity', choices=['dark-on-light', 'light-on-dark'], default='dark-on-light')
    parser.add_argument('--min-object-area', type=float, help='Drop parts/specks below this area (px^2, default 50)')
    parser.add_argument('--min-hole-area', type=float, help='Drop holes below this area (px^2, default 8)')
    parser.add_argument('--point-spacing', type=float, default=1.1, help='Contour vertex spacing (px); 0 keeps every lattice crossing')
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text())
    contours = Contours.from_dict(json.loads(args.contours.read_text())) if args.contours else None
    image = np.asarray(Image.open(args.image).convert('L'))
    report, annotated, contours = inspect_image(image, request, contours=contours,
        extraction={'polarity': args.polarity, 'min_object_area': args.min_object_area,
                    'min_hole_area': args.min_hole_area, 'point_spacing': args.point_spacing})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (args.output/'contours.json').write_text(json.dumps(contours.to_dict(), indent=2)+'\n')
    annotated.save(args.output/'annotated.png')
    print(json.dumps(report['summary']))
    return 1 if report['summary']['failed_or_unavailable'] or report['summary']['tolerance_failures'] else 0
