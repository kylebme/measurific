"""Create an imported-mesh example using Trimesh, then render with the CLI.

Run from the project root: python examples/make_mesh_job.py
"""
from pathlib import Path
import json
import numpy as np
import trimesh

folder = Path(__file__).resolve().parent/'generated_mesh'
folder.mkdir(exist_ok=True)
trimesh.creation.annulus(r_min=0.45, r_max=1.0, height=0.3, sections=256).export(folder/'washer.stl')
# A chamfered box as the convex hull of analytically constructed vertices.
# Imported curves retain their tessellation error; increase sections for a
# tighter error budget. The washer's radial sagitta is r*(1-cos(pi/sections)).
vertices = []
for z, half in [(0.0, 0.8), (0.15, 1.0), (0.65, 1.0), (0.8, 0.8)]:
    vertices.extend([[x, y, z] for x in [-half, half] for y in [-half, half]])
trimesh.convex.convex_hull(np.asarray(vertices)).export(folder/'chamfered_block.stl')
job = {
    'config': {
        'camera': {'width': 160, 'height': 96, 'magnification': 0.1},
        'illumination': {'kind': 'diffuse', 'samples': 32},
        'sampling': {'supersample': 8},
        'optics': {'psf': 'gaussian', 'object_na': 0.02, 'support_px': 4},
    },
    'shapes': [
        {'kind': 'mesh', 'path': 'washer.stl', 'name': 'washer', 'translation_mm': [-1.3, 0, 0], 'backend': 'auto'},
        {'kind': 'mesh', 'path': 'chamfered_block.stl', 'name': 'chamfered_block', 'translation_mm': [1.3, 0, 0], 'rotation_deg': [8, 0, 10], 'backend': 'auto'},
    ],
    'dataset': {'count': 1, 'random_phase': True, 'exposures_ms': [1]},
}
(folder/'job.json').write_text(json.dumps(job, indent=2)+'\n')
print(folder/'job.json')
