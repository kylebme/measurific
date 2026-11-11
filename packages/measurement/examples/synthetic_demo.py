"""Render a washer with synthetic-shapes, extract edge-model contours, measure and annotate.

Run from repository root using the PYTHONPATH shown in README.md.
"""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from inspection_shapes.geometry import generate
from inspection_shapes.render import Camera, render
from inspection_measurement import contours_from_image, inspect_image


def main(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    part = generate('washer', 561, severity=0)
    extent = np.ptp(part.rings[0].actual, axis=0).max()
    camera = Camera(width=640, height=480, supersample=4,
                    pixels_per_mm=340/extent, translation_x=320, translation_y=240,
                    sigma_px=.8, read_noise_dn=.25, prnu=0, electrons_per_dn=1000)
    images, _, _ = render(part, camera, seed=19)
    gray = images['image']
    contours, diagnostics = contours_from_image(gray)
    outer = next(r['path'] for r in diagnostics['rings'] if not r['is_hole'])
    inner = next(r['path'] for r in diagnostics['rings'] if r['is_hole'])
    request = {'schema_version': 1, 'units': 'px', 'coordinate_hash': contours.coordinate_hash,
        'features': [
            {'id': 'outer', 'type': 'circle', 'semantic': 'external_circle', 'selection': {'path': outer}},
            {'id': 'bore', 'type': 'circle', 'semantic': 'hole', 'selection': {'path': inner}}],
        'measurements': [
            {'id': 'OD least-squares diameter', 'operator': 'least_squares_diameter', 'feature_refs': ['outer'],
             'nominal_value': 340, 'tolerance': {'lower': 339, 'upper': 341}},
            {'id': 'Bore least-squares diameter', 'operator': 'least_squares_diameter', 'feature_refs': ['bore']},
            {'id': 'Bore-to-OD center offset', 'operator': 'center_offset', 'feature_refs': ['outer', 'bore']},
            {'id': 'Minimum wall separation', 'operator': 'minimum_separation', 'feature_refs': ['outer', 'bore']},
            {'id': 'OD radial range about LS center', 'operator': 'circularity_ls', 'feature_refs': ['outer']}]}
    report, annotated, _ = inspect_image(gray, request, contours=contours)
    report['extraction'] = diagnostics
    Image.fromarray(gray).save(output/'input.png')
    annotated.save(output/'annotated.png')
    for name, document in [('request', request), ('results', report), ('contours', contours.to_dict())]:
        (output/f'{name}.json').write_text(json.dumps(document, indent=2, allow_nan=False)+'\n')
    expected = []
    for ring in part.rings:
        p = camera.project(ring.actual)
        expected.append(float(2*np.linalg.norm(p-p.mean(0), axis=1).mean()))
    (output/'reference.json').write_text(json.dumps({'analytic_projected_diameters_px': expected,
        'note': 'Renderer rasterization introduces bias; this is an integration demo, not a subpixel accuracy certification.'}, indent=2)+'\n')
    print(json.dumps(report['summary']))
    for m in report['measurements']:
        print(m['id'], m['value'], m.get('reason', ''))
    return report


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'packages/measurement/outputs/washer')
