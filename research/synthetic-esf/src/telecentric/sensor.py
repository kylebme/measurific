"""Linear monochrome sensor: persistent PRNU/DSNU, shot/read noise, ADC."""
import numpy as np
from .config import positive


class SensorModel:
    def __init__(self, config, shape):
        self.config = config
        rng = np.random.default_rng(config.fixed_pattern_seed)
        self.prnu = np.maximum(0, 1 + rng.normal(0, config.prnu_std, shape))
        self.dark_rate = np.maximum(0, config.dark_e_per_ms + rng.normal(0, config.dsnu_e_per_ms, shape))

    def capture(self, irradiance, exposure_ms=1.0, seed=0, roi=None):
        positive('exposure_ms', exposure_ms)
        c = self.config
        prnu, dark = self.prnu, self.dark_rate
        if roi is not None:
            x, y, w, h = roi
            prnu, dark = prnu[y:y+h, x:x+w], dark[y:y+h, x:x+w]
        if irradiance.shape != prnu.shape:
            raise ValueError('Irradiance dimensions must match sensor or ROI')
        if not np.isfinite(irradiance).all() or np.any(irradiance < 0):
            raise ValueError('Irradiance must be finite and nonnegative')
        mean_e = exposure_ms*(irradiance*c.bright_e_per_ms*prnu + dark)
        if c.noise:
            rng = np.random.default_rng(seed)
            # Do not clamp the Poisson mean at full well: saturation happens
            # after charge is accumulated, before electronic read noise.
            charge = rng.poisson(mean_e).astype(np.float64)
        else:
            charge = mean_e.copy()
        well_saturated = charge >= c.full_well_e
        charge = np.minimum(charge, c.full_well_e)
        if c.noise:
            charge += rng.normal(0, c.read_noise_e, charge.shape)
        analog_adu = charge/c.electrons_per_adu + c.black_level_adu
        maximum = 2**c.bits - 1
        raw = np.rint(np.clip(analog_adu, 0, maximum)).astype(np.uint16)
        return dict(raw=raw, expected_e=mean_e.astype(np.float32), saturated=well_saturated | (analog_adu >= maximum), clipped_low=analog_adu <= 0)
