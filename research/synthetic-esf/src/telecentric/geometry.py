"""Double precision analytic solids and a Trimesh adapter.

Camera coordinates: x right, y down, z along parallel rays. Rays pass through
the fixed object-space focus plane; solids are opaque and non-reflecting.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
from .config import positive


def slab(o, d, lower, upper):
    """Intersection intervals for infinite lines with an axis-aligned box."""
    lo = np.full(len(o), -np.inf)
    hi = np.full(len(o), np.inf)
    for i in range(o.shape[1]):
        if abs(d[i]) < 1e-15:
            hi = np.where((o[:, i] >= lower[i]) & (o[:, i] <= upper[i]), hi, -np.inf)
        else:
            a, b = (lower[i] - o[:, i]) / d[i], (upper[i] - o[:, i]) / d[i]
            lo = np.maximum(lo, np.minimum(a, b))
            hi = np.minimum(hi, np.maximum(a, b))
    return lo, hi


def quadratic_interval(o, d, radius):
    a = float(d @ d)
    c = np.einsum('ij,ij->i', o, o) - radius**2
    if a < 1e-30:
        return np.full(len(o), -np.inf), np.where(c <= 0, np.inf, -np.inf)
    b = o @ d
    disc = b*b - a*c
    root = np.sqrt(np.maximum(disc, 0))
    # Cancellation is unimportant for the occupancy predicate; coordinates are
    # translated to each primitive's center before solving.
    return (-b-root)/a, np.where(disc >= 0, (-b+root)/a, -np.inf)


@dataclass
class Primitive:
    kind: str
    center_mm: tuple = (0.0, 0.0, 0.0)
    radius_mm: float = 0.5
    size_mm: tuple = (1.0, 1.0, 0.1)
    length_mm: float = 2.0
    axis: str = 'y'
    angle_deg: float = 0.0
    name: str = 'part'

    def __post_init__(self):
        if self.kind not in {'sphere', 'cylinder', 'box'}:
            raise ValueError('Primitive kind must be sphere, cylinder, or box')
        if self.axis not in {'y', 'z'}:
            raise ValueError('Cylinder axis must be y or z; angle_deg rotates about z')
        if len(self.center_mm) != 3 or len(self.size_mm) != 3:
            raise ValueError('center_mm and size_mm must have three entries')
        if not np.all(np.isfinite([*self.center_mm, self.angle_deg])):
            raise ValueError('Primitive pose must be finite')
        for key, val in [('radius_mm', self.radius_mm), ('length_mm', self.length_mm), *[('size_mm', x) for x in self.size_mm]]:
            positive(key, val)
        a = np.deg2rad(self.angle_deg)
        self.rotation = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])

    @property
    def z_bounds(self):
        half = self.radius_mm if self.kind == 'sphere' or (self.kind == 'cylinder' and self.axis == 'y') else (self.size_mm[2]/2 if self.kind == 'box' else self.length_mm/2)
        return self.center_mm[2] - half, self.center_mm[2] + half

    def blocked(self, origins, direction):
        o = (origins - self.center_mm) @ self.rotation
        d = direction @ self.rotation
        if self.kind == 'sphere':
            lo, hi = quadratic_interval(o, d, self.radius_mm)
        elif self.kind == 'box':
            half = np.asarray(self.size_mm)/2
            lo, hi = slab(o, d, -half, half)
        else:
            axial = 1 if self.axis == 'y' else 2
            radial = [i for i in range(3) if i != axial]
            lo, hi = quadratic_interval(o[:, radial], d[radial], self.radius_mm)
            al, ah = slab(o[:, [axial]], d[[axial]], [-self.length_mm/2], [self.length_mm/2])
            lo, hi = np.maximum(lo, al), np.minimum(hi, ah)
        return (lo <= hi) & (hi != -np.inf)

    def contour(self, count=512):
        """Exact points on the analytic orthographic boundary, not a raster trace."""
        circle = self.kind == 'sphere' or (self.kind == 'cylinder' and self.axis == 'z')
        if circle:
            a = np.arange(count) * (2*np.pi/count)
            normals = np.column_stack((np.cos(a), np.sin(a)))
            points = normals * self.radius_mm
            curvature = np.full(count, 1/self.radius_mm)
        else:
            w, h = self.size_mm[:2] if self.kind == 'box' else (2*self.radius_mm, self.length_mm)
            n = max(2, count//4)
            t = (np.arange(n)+0.5)/n  # corners have undefined normal/curvature
            points = np.concatenate([np.column_stack((-w/2+w*t, np.full(n, -h/2))), np.column_stack((np.full(n, w/2), -h/2+h*t)), np.column_stack((w/2-w*t, np.full(n, h/2))), np.column_stack((np.full(n, -w/2), h/2-h*t))])
            normals = np.repeat([[0, -1], [1, 0], [0, 1], [-1, 0]], n, axis=0)
            curvature = np.zeros(4*n)
        r = self.rotation[:2, :2]
        return dict(xy_mm=points @ r.T + self.center_mm[:2], normal_xy=normals @ r.T, curvature_per_mm=curvature)

    def metadata(self):
        from dataclasses import asdict
        return {**asdict(self), 'ground_truth': 'analytic orthographic boundary', 'z_bounds_mm': self.z_bounds}


class Mesh:
    def __init__(self, path, scale_to_mm=1.0, translation_mm=(0, 0, 0), rotation_deg=(0, 0, 0), name='mesh', backend='auto'):
        import trimesh
        positive('scale_to_mm', scale_to_mm)
        if backend not in {'auto', 'triangle', 'embree'}:
            raise ValueError('backend must be auto, triangle, or embree')
        if len(translation_mm) != 3 or len(rotation_deg) != 3 or not np.all(np.isfinite([*translation_mm, *rotation_deg])):
            raise ValueError('Mesh pose must contain finite 3-vectors')
        self.path = str(Path(path).resolve())
        self.mesh = trimesh.load(self.path, force='mesh', process=False)
        if not isinstance(self.mesh, trimesh.Trimesh) or not len(self.mesh.faces):
            raise ValueError('Expected a nonempty triangle mesh')
        if not np.all(np.isfinite(self.mesh.vertices)):
            raise ValueError('Mesh vertices must be finite')
        self.mesh.apply_scale(scale_to_mm)
        transform = trimesh.transformations.euler_matrix(*np.deg2rad(rotation_deg), axes='sxyz')
        transform[:3, 3] = translation_mm
        self.mesh.apply_transform(transform)
        self.backend = 'triangle'
        if backend != 'triangle':
            try:
                from trimesh.ray.ray_pyembree import RayMeshIntersector
                self.intersector = RayMeshIntersector(self.mesh)
                self.backend = 'embree'
            except ImportError:
                if backend == 'embree':
                    raise
        if self.backend == 'triangle':
            from trimesh.ray.ray_triangle import RayMeshIntersector
            self.intersector = RayMeshIntersector(self.mesh)
        self.name = name
        self._meta = dict(kind='mesh', name=name, path=self.path, scale_to_mm=scale_to_mm, translation_mm=translation_mm, rotation_deg=rotation_deg, backend=self.backend, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), watertight=bool(self.mesh.is_watertight), ground_truth='projected input mesh; CAD tessellation error is not included')
        self._projection = None

    @property
    def z_bounds(self):
        return self.mesh.bounds[:, 2]

    def blocked(self, origins, direction):
        # Move origins before the entire mesh; analytic primitives use infinite
        # lines, whereas Trimesh accepts only forward half-rays.
        if not len(origins):
            return np.zeros(0, dtype=bool)
        start_z = self.z_bounds[0] - max(float(np.ptp(self.mesh.bounds, axis=0).max()), 1.0)
        t = (start_z-origins[:, 2])/direction[2]
        starts = origins + t[:, None]*direction
        dirs = np.broadcast_to(direction, starts.shape).copy()
        return self.intersector.intersects_any(starts, dirs)

    def contour(self, count=512):
        import shapely
        from shapely.geometry import Polygon
        if self._projection is None:
            triangles = [Polygon(t[:, :2]) for t in self.mesh.triangles]
            self._projection = shapely.union_all([p for p in triangles if p.area > 0])
        geometry = self._projection
        polys = [geometry] if geometry.geom_type == 'Polygon' else list(geometry.geoms)
        rings = [r for p in polys for r in [p.exterior, *p.interiors]]
        total = sum(r.length for r in rings)
        points, normals, ring_ids = [], [], []
        # Each straight segment is sampled directly: true mesh vertices and
        # hole boundaries do not depend on a rendered threshold.
        for rid, ring in enumerate(rings):
            coords = np.asarray(ring.coords)
            for a, b in zip(coords[:-1], coords[1:]):
                delta = b-a
                length = np.linalg.norm(delta)
                if length == 0:
                    continue
                n = max(1, int(np.ceil(count*length/total)))
                points.extend(a + ((np.arange(n)+0.5)/n)[:, None]*delta)
                normal = np.array([delta[1], -delta[0]])/length
                midpoint = (a+b)/2
                if shapely.contains_xy(geometry, *(midpoint + normal*max(length*1e-7, 1e-10))):
                    normal = -normal
                normals.extend(np.tile(normal, (n, 1)))
                ring_ids.extend([rid]*n)
        return dict(xy_mm=np.asarray(points).reshape(-1, 2), normal_xy=np.asarray(normals).reshape(-1, 2), curvature_per_mm=np.zeros(len(points)), ring_id=np.asarray(ring_ids))

    def metadata(self):
        return {**self._meta, 'z_bounds_mm': self.z_bounds.tolist(), 'triangles': len(self.mesh.faces)}


class Scene:
    def __init__(self, shapes):
        self.shapes = list(shapes)

    def blocked(self, origins, direction):
        result = np.zeros(len(origins), dtype=bool)
        for shape in self.shapes:
            remaining = ~result
            if not remaining.any():
                break
            result[remaining] |= shape.blocked(origins[remaining], direction)
        return result

    def labels(self, camera, count=512):
        result = []
        for i, shape in enumerate(self.shapes):
            label = shape.contour(count)
            xy = label['xy_mm']
            # Probe just outside this contour. Coincident outer boundaries are
            # visible, whereas a shared seam with solid on both sides is not.
            outside = xy + label['normal_xy'] * (camera.object_pitch_mm*1e-7)
            origins = np.column_stack((outside, np.full(len(xy), camera.focus_z_mm)))
            visible = np.ones(len(xy), dtype=bool)
            for j, other in enumerate(self.shapes):
                if j != i:
                    visible &= ~other.blocked(origins, np.array([0., 0., 1.]))
            label['visible_in_union'] = visible
            label['ideal_xy_px'] = camera.world_to_ideal_pixel(xy)
            label['xy_px'] = camera.world_to_pixel(xy)
            label['distorted_xy_px'] = label['xy_px']
            label['normal_world_xy'] = label['normal_xy']
            label['normal_xy'] = camera.image_normals(label['ideal_xy_px'], label['normal_world_xy'])
            label['in_frame'] = np.all((label['xy_px'] >= 0) & (label['xy_px'] <= [camera.width, camera.height]), axis=1)
            result.append({'shape': shape.metadata(), **{k: v.tolist() for k, v in label.items()}})
        return result


def scene_from_list(specs, base_dir=Path('.')):
    shapes = []
    for spec in specs:
        spec = dict(spec)
        kind = spec.pop('kind')
        if kind == 'mesh':
            spec['path'] = str(base_dir / spec['path'])
            shapes.append(Mesh(**spec))
        else:
            shapes.append(Primitive(kind=kind, **spec))
    return Scene(shapes)
