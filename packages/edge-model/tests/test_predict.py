"""The shipped inference model runs without the dataset/cache and recovers a known
analytic edge: a dark disk on a backlit field, rendered by exact area coverage (like a
real sensor), compared to the true circle.

Known limitation (RESEARCH.md): under symmetric Gaussian blur of a 2-D disk the model
places edges +0.07-0.17 px outward (sigma 0.6-1.5), likely a learned prior from the
renders' defocus of 3-D parts. The sharp case is asserted; the blur case is only
reported."""

import os
import unittest

import cv2
import numpy as np

from edgesr.predict import DEFAULT_MODEL, load


def disk_image(size=192, cx=96.3, cy=95.7, r=40.25, bright=52000, dark=3000, ss=8, seed=0,
               sigma=0.0):
    g = (np.arange(size * ss) + 0.5) / ss
    inside = (g[None, :] - cx) ** 2 + (g[:, None] - cy) ** 2 < r * r
    cov = inside.reshape(size, ss, size, ss).mean((1, 3))
    img = bright + (dark - bright) * cov
    if sigma:
        img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma)
    img += np.random.default_rng(seed).normal(0, 150, img.shape).astype(np.float32)
    return np.clip(np.round(img), 0, 65535).astype(np.uint16), (cx, cy, r)


@unittest.skipUnless(os.path.exists(DEFAULT_MODEL), "model file not present")
class PredictTest(unittest.TestCase):
    def test_recovers_disk(self):
        img, (cx, cy, r) = disk_image()
        model = load(threads=4)
        out = model(img)
        p = out["points"]
        self.assertGreater(len(p), 1000)
        radial = np.hypot(p[:, 0] - cx, p[:, 1] - cy) - r
        self.assertLess(np.abs(radial).max(), 0.2)          # no spurious contours
        self.assertLess(np.abs(radial).mean(), 0.05)        # sub-pixel localization
        self.assertLess(abs(np.median(radial)), 0.05)       # no bias

    def test_report_blur_bias(self):
        model = load(threads=4)
        for sigma in (0.6, 1.5):
            img, (cx, cy, r) = disk_image(sigma=sigma)
            p = model(img)["points"]
            radial = np.hypot(p[:, 0] - cx, p[:, 1] - cy) - r
            print(f"\n  Gaussian blur sigma {sigma}: median radial bias {np.median(radial):+.3f} px "
                  f"(known limitation)")
            self.assertLess(np.abs(radial).max(), 0.5)          # still no spurious contours

    def test_8bit_and_max_value(self):
        img16, _ = disk_image()
        img8 = np.round(img16 / 257).astype(np.uint8)
        model = load(threads=4)
        n16, n8 = len(model(img16)["points"]), len(model(img8)["points"])
        self.assertLess(abs(n16 - n8) / n16, 0.05)
        # 12-bit data stored in uint16 must be given its full scale
        img12 = np.round(img16 / 16).astype(np.uint16)
        self.assertLess(abs(len(model(img12, max_value=4095)["points"]) - n16) / n16, 0.05)


if __name__ == "__main__":
    unittest.main()
