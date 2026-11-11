"""Run with the synthetic-shapes venv:  ../synthetic-shapes/.venv/bin/python -m unittest discover -s tests"""
import collections
import math
import os
import sys
import unittest

import numpy as np
from shapely.geometry import Point, Polygon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from macro import SIZE  # noqa: E402
from macro.sample import _boundary_samples, _half_profile, _plain, sample_macro_scene  # noqa: E402


class SampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenes = [sample_macro_scene(11, i) for i in range(40)]

    def test_deterministic(self):
        m1, a1 = sample_macro_scene(11, 5)
        m2, a2 = self.scenes[5]
        self.assertEqual(m1, m2)
        self.assertEqual(sorted(a1), sorted(a2))
        for k in a1:
            np.testing.assert_array_equal(a1[k], a2[k])

    def test_counts_and_splits(self):
        counts = [len(m["parts"]) for m, _ in self.scenes]
        self.assertTrue(all(1 <= c <= 10 for c in counts))
        self.assertGreater(len(set(counts)), 5)
        self.assertEqual({m["split"] for m, _ in self.scenes} <= {"train", "val", "test"}, True)

    def test_parts_inside_frame_unless_cut(self):
        for m, arrays in self.scenes:
            for p in m["parts"]:
                if p["cut_by_frame"] or p["kind"] != "extrude":
                    continue
                ring = arrays[p["rings"][0]]
                c, s = math.cos(p["pose"]["rot_rad"]), math.sin(p["pose"]["rot_rad"])
                xy = ring @ np.array([[c, s], [-s, c]]) + [p["pose"]["x"], p["pose"]["y"]]
                self.assertLessEqual(np.abs(xy).max(), SIZE / 2 + 0.5, (m["scene_index"], p["family"]))

    def test_debris_is_rare(self):
        loose = sum(len(m["loose_debris"]) for m, _ in self.scenes)
        parts = sum(len(m["parts"]) for m, _ in self.scenes)
        attached = sum(len(p["debris"]) for m, _ in self.scenes for p in m["parts"])
        self.assertLess(attached / parts, 0.2)
        self.assertLess(loose / len(self.scenes), 1.5)


class GeometryTests(unittest.TestCase):
    def test_boundary_normals_point_away_from_material(self):
        ring = Polygon(Point(0, 0).buffer(50).exterior, [Point(0, 0).buffer(20).exterior.coords[::-1]])
        for x, y, nx, ny in _boundary_samples(ring, 120):
            wx, wy = x - SIZE / 2, SIZE / 2 - y           # back to world, normal y flipped
            probe = Point(wx + 2 * nx, wy - 2 * ny)
            self.assertFalse(ring.contains(probe))

    def test_wires(self):
        # --wires 0 keeps the original scenes exactly
        m0, _ = sample_macro_scene(11, 7)
        m1, _ = sample_macro_scene(11, 7, 0.0)
        self.assertEqual(m0["scene_id"], m1["scene_id"])
        self.assertEqual([p["family"] for p in m0["parts"]], [p["family"] for p in m1["parts"]])
        wires = []
        for i in range(12):
            m, arrays = sample_macro_scene(11, 1000 + i, 1.0)
            ws = [p for p in m["parts"] if p["group"] == "wire"]
            if ws:
                self.assertEqual(m["render"]["mask_supersample"], 8)
            for p in ws:
                prof = arrays[p["profile"]]
                if p["family"] != "wire_head":         # a head is wider than the wire
                    self.assertAlmostEqual(2 * prof[:, 1].max() / p["wire_diameter_px"], 1.0, delta=0.02)
                self.assertGreater(np.ptp(prof[:, 0]), 0.3 * SIZE * 0.7 ** 3 - 1)   # shortened <= 3x
                self.assertTrue(0.25 <= p["wire_diameter_px"] <= 4.0)
            wires += ws
        self.assertGreater(len(wires), 6)

    def test_half_profile_of_turned_ring(self):
        rings, holes, kind = _plain("plain_rod", np.random.default_rng(3))
        self.assertEqual(kind, "revolve")
        top = _half_profile(rings[0])
        self.assertTrue(np.all(np.diff(top[:, 0]) >= -1e-9))
        self.assertTrue(np.all(top[:, 1] > 0))
        self.assertAlmostEqual(top[0, 0], rings[0][:, 0].min(), places=6)
        self.assertAlmostEqual(top[-1, 0], rings[0][:, 0].max(), places=6)


if __name__ == "__main__":
    unittest.main()
