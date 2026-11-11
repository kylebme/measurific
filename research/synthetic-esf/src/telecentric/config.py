"""All distances are in mm, except explicit *_um and *_px fields."""
from dataclasses import asdict, dataclass, field
import math


def positive(name, value, zero=False):
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")


def integer(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class Camera:
    width: int = 1920
    height: int = 1080
    pixel_pitch_um: float = 3.45
    magnification: float = 0.5
    focus_z_mm: float = 0.0
    center_xy_mm: tuple[float, float] = (0.0, 0.0)
    distortion_radial: tuple[float, float, float] = (0.0, 0.0, 0.0)
    distortion_tangential: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        integer('width', self.width)
        integer('height', self.height)
        positive('pixel_pitch_um', self.pixel_pitch_um)
        positive('magnification', self.magnification)
        if len(self.center_xy_mm) != 2 or not all(map(math.isfinite, (*self.center_xy_mm, self.focus_z_mm))):
            raise ValueError('Camera coordinates must be finite')
        for name, count in [('distortion_radial', 3), ('distortion_tangential', 2)]:
            values = getattr(self, name)
            if len(values) != count or not all(map(math.isfinite, values)):
                raise ValueError(f'{name} must have {count} finite coefficients')
        if self.has_distortion:
            import numpy as np
            x, y = np.meshgrid(np.linspace(-0.1*self.width, 1.1*self.width, 25), np.linspace(-0.1*self.height, 1.1*self.height, 25))
            jac = self.distortion_jacobian(np.stack((x, y), axis=-1))
            if np.any(np.linalg.det(jac) <= 0.05):
                raise ValueError('Distortion is folded or nearly singular over the field/halo')

    @property
    def object_pitch_mm(self):
        return self.pixel_pitch_um * 1e-3 / self.magnification

    def pixel_to_world(self, xy):
        import numpy as np
        return (self.pixel_to_ideal(xy) - [self.width / 2, self.height / 2]) * self.object_pitch_mm + self.center_xy_mm

    def world_to_pixel(self, xy):
        return self.ideal_to_pixel(self.world_to_ideal_pixel(xy))

    def world_to_ideal_pixel(self, xy):
        import numpy as np
        return (np.asarray(xy) - self.center_xy_mm) / self.object_pitch_mm + [self.width / 2, self.height / 2]

    @property
    def has_distortion(self):
        return any(self.distortion_radial) or any(self.distortion_tangential)

    def _distort(self, xy, jacobian=False):
        import numpy as np
        xy = np.asarray(xy, dtype=float)
        scale = max(self.width, self.height)/2
        x, y = np.moveaxis((xy-[self.width/2, self.height/2])/scale, -1, 0)
        k1, k2, k3 = self.distortion_radial
        p1, p2 = self.distortion_tangential
        r2 = x*x+y*y
        radial = 1+r2*(k1+r2*(k2+r2*k3))
        if not jacobian:
            out = np.stack((x*radial+2*p1*x*y+p2*(r2+2*x*x), y*radial+p1*(r2+2*y*y)+2*p2*x*y), axis=-1)
            return out*scale+[self.width/2, self.height/2]
        dr = 2*(k1+2*k2*r2+3*k3*r2*r2)
        cross = x*y*dr+2*p1*x+2*p2*y
        return np.stack((radial+x*x*dr+2*p1*y+6*p2*x, cross, cross, radial+y*y*dr+6*p1*y+2*p2*x), axis=-1).reshape(x.shape+(2, 2))

    def ideal_to_pixel(self, xy):
        import numpy as np
        return self._distort(xy) if self.has_distortion else np.asarray(xy, dtype=float)

    def distortion_jacobian(self, xy):
        return self._distort(xy, jacobian=True)

    def pixel_to_ideal(self, xy):
        import numpy as np
        target = np.asarray(xy, dtype=float)
        if not self.has_distortion or target.size == 0:
            return target
        estimate = target.copy()
        for _ in range(20):
            residual = self.ideal_to_pixel(estimate)-target
            if np.max(np.abs(residual)) < 1e-10:
                return estimate
            jac = self.distortion_jacobian(estimate)
            a, b, c, d = jac[..., 0, 0], jac[..., 0, 1], jac[..., 1, 0], jac[..., 1, 1]
            det = a*d-b*c
            if np.any(det <= 0.05) or not np.isfinite(jac).all():
                raise ValueError('Distortion inverse encountered a folded/singular map')
            dx = (d*residual[..., 0]-b*residual[..., 1])/det
            dy = (-c*residual[..., 0]+a*residual[..., 1])/det
            estimate -= np.stack((dx, dy), axis=-1)
        raise ValueError('Distortion inversion failed to converge to 1e-10 pixels')

    def image_normals(self, ideal_xy, world_normals):
        import numpy as np
        jac = self.distortion_jacobian(ideal_xy)
        normals = np.asarray(world_normals)
        # J^{-T} n, normalized; determinant is positive and cancels.
        nx = jac[..., 1, 1]*normals[..., 0]-jac[..., 1, 0]*normals[..., 1]
        ny = -jac[..., 0, 1]*normals[..., 0]+jac[..., 0, 0]*normals[..., 1]
        out = np.stack((nx, ny), axis=-1)
        return out/np.linalg.norm(out, axis=-1, keepdims=True)


@dataclass(frozen=True)
class Illumination:
    kind: str = 'cone'  # collimated, cone, gaussian, diffuse
    half_angle_deg: float = 0.25
    gaussian_sigma_deg: float = 0.2
    samples: int = 32
    seed: int = 17

    def __post_init__(self):
        if self.kind not in {'collimated', 'cone', 'gaussian', 'diffuse'}:
            raise ValueError('Unknown illumination kind')
        if not math.isfinite(self.half_angle_deg) or not 0 <= self.half_angle_deg <= 90:
            raise ValueError('half_angle_deg must be in [0, 90]')
        positive('gaussian_sigma_deg', self.gaussian_sigma_deg)
        integer('samples', self.samples)
        integer('seed', self.seed, 0)
        if self.samples & (self.samples - 1):
            raise ValueError('samples must be a power of two for Sobol quadrature')


@dataclass(frozen=True)
class Optics:
    object_na: float = 0.015
    wavelength_um: float = 0.63
    psf: str = 'gaussian'  # none, gaussian, mixture, airy, measured
    sigma_px: float = 0.6
    sigma_y_px: float | None = None
    angle_deg: float = 0.0
    tail_weight: float = 0.08
    tail_sigma_px: float = 1.5
    tail_offset_px: tuple[float, float] = (0.7, 0.0)
    support_px: float = 5.0
    measured_path: str | None = None
    measured_pitch_um: float | None = None  # OBJECT space, pre-pixel PSF
    field_grid: tuple[int, int] = (1, 1)  # columns, rows; nodes in row-major order
    field_nodes: tuple[dict, ...] = ()  # local PSF overrides at full-field grid nodes

    def __post_init__(self):
        if not math.isfinite(self.object_na) or not 0 < self.object_na < 1:
            raise ValueError('object_na must be in (0, 1), in air')
        positive('wavelength_um', self.wavelength_um)
        if self.psf not in {'none', 'gaussian', 'mixture', 'airy', 'measured'}:
            raise ValueError('Unknown PSF')
        positive('sigma_px', self.sigma_px)
        if self.sigma_y_px is not None:
            positive('sigma_y_px', self.sigma_y_px)
        if not math.isfinite(self.angle_deg):
            raise ValueError('angle_deg must be finite')
        positive('tail_sigma_px', self.tail_sigma_px)
        positive('support_px', self.support_px)
        if not 0 <= self.tail_weight <= 1:
            raise ValueError('tail_weight must be in [0, 1]')
        if len(self.tail_offset_px) != 2 or not all(map(math.isfinite, self.tail_offset_px)):
            raise ValueError('tail_offset_px must contain two finite values')
        if self.psf == 'measured':
            if not self.measured_path or self.measured_pitch_um is None:
                raise ValueError('Measured PSF requires path and object-space sample pitch')
            positive('measured_pitch_um', self.measured_pitch_um)
        if len(self.field_grid) != 2:
            raise ValueError('field_grid is [columns, rows]')
        for size in self.field_grid:
            integer('field_grid', size)
        if self.field_nodes:
            if len(self.field_nodes) != self.field_grid[0]*self.field_grid[1]:
                raise ValueError('field_nodes count must equal field_grid product')
            allowed = {'psf', 'sigma_px', 'sigma_y_px', 'angle_deg', 'tail_weight', 'tail_sigma_px', 'tail_offset_px', 'measured_path', 'measured_pitch_um'}
            base = asdict(self)
            base.update(field_grid=(1, 1), field_nodes=())
            for node in self.field_nodes:
                if not isinstance(node, dict) or set(node)-allowed:
                    raise ValueError('Invalid field node PSF override')
                Optics(**{**base, **node})
        elif tuple(self.field_grid) != (1, 1):
            raise ValueError('field_grid requires field_nodes')


@dataclass(frozen=True)
class Sensor:
    bright_e_per_ms: float = 10000.0
    dark_e_per_ms: float = 0.05
    read_noise_e: float = 2.0
    full_well_e: float = 30000.0
    electrons_per_adu: float = 8.0
    black_level_adu: float = 64.0
    bits: int = 12
    prnu_std: float = 0.005
    dsnu_e_per_ms: float = 0.01
    fixed_pattern_seed: int = 123
    noise: bool = True

    def __post_init__(self):
        for name in ('bright_e_per_ms', 'full_well_e', 'electrons_per_adu'):
            positive(name, getattr(self, name))
        for name in ('dark_e_per_ms', 'read_noise_e', 'black_level_adu', 'prnu_std', 'dsnu_e_per_ms'):
            positive(name, getattr(self, name), zero=True)
        integer('bits', self.bits)
        integer('fixed_pattern_seed', self.fixed_pattern_seed, 0)
        if self.bits > 16 or self.black_level_adu > 2**self.bits - 1:
            raise ValueError('Sensor requires <=16 bits and a representable black level')


@dataclass(frozen=True)
class Sampling:
    supersample: int = 16
    tile_px: int = 32
    ray_chunk: int = 65536

    def __post_init__(self):
        for key in ('supersample', 'tile_px', 'ray_chunk'):
            integer(key, getattr(self, key))


@dataclass(frozen=True)
class RenderConfig:
    camera: Camera = field(default_factory=Camera)
    illumination: Illumination = field(default_factory=Illumination)
    optics: Optics = field(default_factory=Optics)
    sensor: Sensor = field(default_factory=Sensor)
    sampling: Sampling = field(default_factory=Sampling)
    background: float = 1.0
    foreground: float = 0.0
    gradient_xy: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        positive('background', self.background)
        positive('foreground', self.foreground, zero=True)
        if self.foreground >= self.background:
            raise ValueError('foreground must be below background')
        if len(self.gradient_xy) != 2 or not all(map(math.isfinite, self.gradient_xy)):
            raise ValueError('gradient_xy must have two finite entries')
        if sum(map(abs, self.gradient_xy)) >= 2:
            raise ValueError('Gradient must remain positive over the sensor')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for name, factory in [('camera', Camera), ('illumination', Illumination), ('optics', Optics), ('sensor', Sensor), ('sampling', Sampling)]:
            if name in value:
                value[name] = factory(**value[name])
        return cls(**value)
