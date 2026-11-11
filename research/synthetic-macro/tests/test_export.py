"""Run with a venv that has OpenCV:  ../../packages/edge-model/.venv/bin/python -m unittest tests.test_export"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ExportTests(unittest.TestCase):
    def test_bridged_hole_rasterizes_empty(self):
        from macro.export import mask_polygons, merge_components, rasterize_evenodd
        yy, xx = np.mgrid[0:64, 0:64]
        r = np.hypot(xx - 31.5, yy - 31.5)
        mask = (r < 25) & (r > 10)
        poly = merge_components(mask_polygons(mask))
        back = rasterize_evenodd(poly, mask.shape).astype(bool)
        self.assertGreater((back & mask).sum() / (back | mask).sum(), 0.99)


if __name__ == "__main__":
    unittest.main()
