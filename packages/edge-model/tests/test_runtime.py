"""Torch-free ONNX runtime (edgesr.runtime): parity with the torch predictor and ordered
closed rings. No dataset or cache needed."""

import importlib.util
import os
import subprocess
import sys
import unittest

import cv2
import numpy as np

from edgesr.runtime import DEFAULT_MODEL

HAVE_ORT = importlib.util.find_spec("onnxruntime") is not None
HAVE_TORCH = importlib.util.find_spec("torch") is not None


def disk_image(size=192, cx=96.3, cy=95.7, r=40.25, ss=8):
    g = (np.arange(size * ss) + 0.5) / ss
    cover = ((g[None, :] - cx) ** 2 + (g[:, None] - cy) ** 2 < r * r).reshape(size, ss, size, ss).mean((1, 3))
    img = 52000 - 49000 * cover + np.random.default_rng(0).normal(0, 150, cover.shape)
    return np.clip(np.round(img), 0, 65535).astype(np.uint16), (cx, cy, r)


def shoelace(q):
    return np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]) / 2


@unittest.skipUnless(HAVE_ORT and os.path.isdir(DEFAULT_MODEL), "onnxruntime or exported model missing")
class RuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from edgesr.runtime import EdgeModel
        cls.model = EdgeModel(threads=4)

    def test_import_does_not_need_torch(self):
        code = "import sys, edgesr.runtime; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True)

    @unittest.skipUnless(HAVE_TORCH, "torch not installed")
    def test_points_match_torch_predictor(self):
        from scipy.spatial import cKDTree

        from edgesr.predict import load
        img, _ = disk_image()
        expected = load(threads=4)(img)["points"]
        actual = self.model(img)["points"]
        self.assertLessEqual(abs(len(actual) - len(expected)), 2)
        self.assertLess(np.quantile(cKDTree(actual).query(expected)[0], 0.999), 1e-3)

    def test_disk_ring_is_ordered_closed_and_accurate(self):
        from shapely import Polygon
        img, (cx, cy, r) = disk_image()
        out = self.model.rings(img)
        self.assertEqual((out["is_hole"], out["depth"], out["parent"]), ([False], [0], [-1]))
        q = out["rings"][0]
        steps = np.linalg.norm(np.diff(np.vstack([q, q[:1]]), axis=0), axis=1)
        self.assertLess(steps.max(), 0.1)                       # consecutive lattice crossings
        self.assertTrue(Polygon(q).is_valid)
        self.assertLess(shoelace(q), 0)                         # object outline orientation
        radial = np.hypot(q[:, 0] - cx, q[:, 1] - cy) - r
        self.assertLess(np.abs(radial).max(), 0.2)
        self.assertLess(abs(radial.mean()), 0.05)

    def test_part_cut_by_the_frame_closes_along_it(self):
        img = np.full((180, 200), 235, np.uint8)
        img[40:150] = 12                                        # band across the full width
        cv2.circle(img, (100, 95), 22, 235, -1)
        out = self.model.rings(img)
        self.assertEqual(out["is_hole"], [False, True])
        self.assertEqual(out["parent"], [-1, 0])
        outer, hole = out["rings"]
        self.assertLess(outer[:, 0].min(), 0.1)
        self.assertGreater(outer[:, 0].max(), 199.9)
        self.assertGreater(shoelace(hole), 0)
        radial = np.hypot(hole[:, 0] - 100.5, hole[:, 1] - 95.5) - 22.5
        self.assertLess(np.abs(radial).max(), 1.0)              # a rasterized circle

    def test_saturated_backlight_is_scaled_into_the_training_range(self):
        # Clipped at full scale (as in many JPEGs), touching the bottom frame. Unscaled,
        # the model hallucinated edges beside the saturated backlight (4 rings).
        img = np.full((160, 200), 300., np.float32)
        img[70:, 40:160] = 15
        cv2.circle(img, (100, 60), 30, 15, -1)
        img = cv2.GaussianBlur(img, (0, 0), 1.0) + np.random.default_rng(1).normal(0, 1.5, img.shape)
        out = self.model.rings(np.clip(np.round(img), 0, 255).astype(np.uint8))
        self.assertAlmostEqual(out["gain"], 0.95, places=3)
        self.assertEqual(out["is_hole"], [False])
        # a backlight inside the training range is left alone
        self.assertEqual(self.model.rings(disk_image()[0])["gain"], 1.0)

    def test_roi_and_minimum_areas(self):
        img = np.full((200, 200), 230, np.uint8)
        img[60:140, 60:140] = 15
        img[20:24, 20:24] = 15                                  # 16 px^2 speck: dropped
        img[20:32, 160:172] = 15                                # 144 px^2: kept
        img[96:100, 96:100] = 230                               # 16 px^2 hole: kept
        out = self.model.rings(img)
        self.assertEqual(out["is_hole"], [False, False, True])
        self.assertGreater(out["dropped_rings"], 0)
        # a circular ROI excludes the small square and cuts through the large one
        yy, xx = np.mgrid[:200, :200]
        valid = (xx + 0.5 - 100) ** 2 + (yy + 0.5 - 100) ** 2 <= 50 ** 2
        out = self.model.rings(img, valid_mask=valid)
        self.assertEqual(out["is_hole"], [False, True])
        outer = out["rings"][0]
        self.assertLess(np.hypot(outer[:, 0] - 100, outer[:, 1] - 100).max(), 51)


def square(x0, y0, x1, y1, hole=False):
    q = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]], float)     # object: negative area
    return q[::-1] if hole else q


class RingSelectionTest(unittest.TestCase):
    """Seed gating and tracer closure, without the model."""

    def test_seed_decides_which_rings_exist(self):
        from edgesr.runtime import nest_rings
        seed = np.zeros((100, 100), bool)
        seed[10:60, 10:60] = True                       # a washer-like part ...
        seed[20:50, 20:50] = False                      # ... with a large bore
        rings = [square(10, 10, 60, 60), square(20, 20, 50, 50, hole=True),
                 square(70, 70, 90, 90),                # no seed support: e.g. a faint fiber
                 square(75, 75, 85, 85, hole=True)]     # inside the unsupported ring
        out = nest_rings(rings, [False, True, False, True], (50.0, 8.0), seed=seed)
        self.assertEqual(out["is_hole"], [False, True])
        self.assertEqual(out["parent"], [-1, 0])
        self.assertEqual((out["unsupported_rings"], out["dropped_rings"]), (1, 2))
        # the bore is judged against seed background, the part only on its own material
        np.testing.assert_array_equal(out["rings"][0], rings[0])

    def test_boundary_at_the_lattice_edge_closes(self):
        from edgesr.runtime import LOGIT, trace_rings
        cell, scale = 16, 16
        canvas = np.full((1, cell * scale, cell * scale), -LOGIT, np.float32)
        canvas[0, :64, :64] = LOGIT                    # object in the frame's top-left corner
        occ = {"canvas": canvas, "cy": np.array([0]), "cx": np.array([0]),
               "mask": np.zeros((32, 32), np.uint8)}
        out = trace_rings(occ, cell, scale)
        self.assertEqual(out["is_hole"], [False])
        q = out["rings"][0]
        self.assertLess(q.min(), 0.1)

    def test_seed_boundary_outside_the_band_closes(self):
        # The part's top edge lies in the band; its sides run down to the lattice border
        # through inactive cells (as when restrict_occupancy cuts a seed that reached past
        # the frame into the apron). The seed decides there, and the ring closes.
        from edgesr.runtime import LOGIT, trace_rings
        cell, scale = 16, 4
        mask = np.zeros((64, 64), bool)
        mask[24:, 16:48] = True
        cx = np.arange(4)
        cy = np.ones(4, int)
        canvas = np.stack([np.where(np.kron(mask[16:32, c * cell:(c + 1) * cell], np.ones((scale, scale), bool)),
                                    LOGIT, -LOGIT) for c in cx]).astype(np.float32)
        out = trace_rings({"canvas": canvas, "cy": cy, "cx": cx, "mask": mask}, cell, scale)
        self.assertEqual(out["is_hole"], [False])
        q = out["rings"][0]
        self.assertAlmostEqual(-shoelace(q), 32 * 40, delta=1.0)

    def test_orientation_of_a_degenerate_ring(self):
        # A lone sample just below 0 inside an object: its hole ring collapses onto the
        # sample (area ~1e-14 px^2); orientation must still say "hole".
        from edgesr.runtime import LOGIT, nest_rings, trace_rings
        cell, scale = 16, 16
        canvas = np.full((1, cell * scale, cell * scale), -LOGIT, np.float32)
        canvas[0, 10:200, 10:200] = LOGIT
        canvas[0, 100, 100] = -1e-6
        occ = {"canvas": canvas, "cy": np.array([1]), "cx": np.array([1]),
               "mask": np.zeros((48, 48), bool)}
        out = trace_rings(occ, cell, scale)
        self.assertEqual(sorted(out["is_hole"]), [False, True])
        self.assertLess(abs(shoelace(out["rings"][out["is_hole"].index(True)])), 1e-6)
        nested = nest_rings(out["rings"], out["is_hole"], (0.0, 0.0))
        self.assertEqual(nested["parent"], [-1, 0])


if __name__ == "__main__":
    unittest.main()
