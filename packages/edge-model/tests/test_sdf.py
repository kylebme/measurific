"""SDFNet pieces that need no dataset: contour extraction, small-region filter, model shapes."""

import unittest

import numpy as np
import torch

from edgesr.sdf_infer import contour_points, filter_small
from edgesr.sdfnet import SDFNet


def disk_sdf(size=64, cx=31.3, cy=30.8, r=12.25, T=8.0):
    g = np.arange(size) + 0.5
    d = r - np.hypot(g[None, :] - cx, g[:, None] - cy)        # + inside
    return np.clip(d, -T, T).astype(np.float32), (cx, cy, r)


class ContourTest(unittest.TestCase):
    def test_disk_points_lie_on_the_circle(self):
        sdf, (cx, cy, r) = disk_sdf()
        p = contour_points(sdf)
        rad = np.hypot(p[:, 0] - cx, p[:, 1] - cy)
        # bilinear interpolation of a curved field: error ~1 / (8 r) px (0.010 at r = 12)
        self.assertLess(np.abs(rad - r).max(), 0.02)
        # densified: about 2 x 16 lattice crossings per px of circumference (|cos| + |sin|)
        self.assertGreater(len(p), 16 * 2 * np.pi * r)

    def test_straight_edge_is_exact(self):
        g = np.arange(32) + 0.5
        x0 = 15.37
        sdf = np.clip(x0 - g[None, :] + 0 * g[:, None], -8, 8).astype(np.float32)
        p = contour_points(sdf)
        self.assertLess(np.abs(p[:, 0] - x0).max(), 1e-5)

    def test_small_regions_are_removed(self):
        sdf, _ = disk_sdf(r=12.25)
        blob, _ = disk_sdf(cx=55.0, cy=55.0, r=2.0)
        both = np.maximum(sdf, blob)
        out = filter_small(both, (50.0, 8.0))
        self.assertTrue((out[50:60, 50:60] < 0).all())
        self.assertTrue((out[28:34, 28:34] > 0).all())


class ModelTest(unittest.TestCase):
    def test_shapes_and_truncation(self):
        m = SDFNet({"widths": [8, 16, 16, 16, 16], "trunc_px": 8.0}).eval()
        with torch.no_grad():
            o = m(torch.randn(2, 4, 64, 96))
        self.assertEqual(tuple(o["sdf"].shape), (2, 64, 96))
        self.assertEqual(tuple(o["dir"].shape), (2, 2, 64, 96))
        self.assertEqual(tuple(o["sdf4"].shape), (2, 16, 24))
        self.assertLessEqual(float(o["sdf"].abs().max()), 8.0)


if __name__ == "__main__":
    unittest.main()
