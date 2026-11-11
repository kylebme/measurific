"""Analytic registration tests for the joint image/mask sampler.

A synthetic scene is defined by an analytic shape. Its 16x mask is the shape
evaluated at hi-res pixel centers and its source image is the exact 16x16 area
coverage of that mask, so image and mask are registered by construction. Any
convention error in the sampler shows up as a mismatch.
"""

import math
import unittest

import numpy as np

from edgesr import SCALE
from edgesr import sampler as S


class AnalyticStore:
    H = W = 256

    def __init__(self, inside):
        self.inside = inside
        self.scenes = [{"part_index": 0}]
        ys = (np.arange(self.H * SCALE) + 0.5) / SCALE
        full = inside(ys[None, :], ys[:, None]).astype(np.uint8)
        self.full_mask = full
        cov = full.reshape(self.H, SCALE, self.W, SCALE).mean((1, 3))
        self.images = [cov]

    def mask_window(self, part, x0, y0, w, h):
        out = np.zeros((h, w), np.uint8)
        n = self.H * SCALE
        cx0, cy0 = max(x0, 0), max(y0, 0)
        cx1, cy1 = min(x0 + w, n), min(y0 + h, n)
        out[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = self.full_mask[cy0:cy1, cx0:cx1]
        return out


def blob(x, y):
    """Asymmetric shape: a disk with a notch plus an off-center square."""
    d = (x - 131.3) ** 2 + (y - 124.6) ** 2 < 9.7 ** 2
    notch = (np.abs(y - 124.6) < 1.3) & (x > 131.3)
    sq = (np.abs(x - 118.2) < 3.1) & (np.abs(y - 138.9) < 2.2)
    return (d & ~notch) | sq


def disk(x, y):
    return (x - 128.37) ** 2 + (y - 127.71) ** 2 < 8.4 ** 2


class RegistrationTest(unittest.TestCase):
    def test_lattice_patches_are_exact(self):
        store = AnalyticStore(blob)
        rng = np.random.default_rng(0)
        for _ in range(16):
            center = (float(rng.integers(120, 136)), float(rng.integers(120, 136)))
            A = S.patch_matrix(flip=bool(rng.integers(2)), quarter_turns=int(rng.integers(4)))
            img, mask = S.sample_patch(store, 0, center, A)
            cov = mask.reshape(S.PATCH, SCALE, S.PATCH, SCALE).mean((1, 3))
            np.testing.assert_allclose(img, cov, atol=1e-6)

    def test_lattice_identity_is_a_crop(self):
        store = AnalyticStore(blob)
        img, mask = S.sample_patch(store, 0, (130.0, 125.0), S.patch_matrix())
        np.testing.assert_array_equal(img, store.images[0][101:149, 106:154].astype(np.float32))
        np.testing.assert_array_equal(mask, store.full_mask[101 * 16:149 * 16, 106 * 16:154 * 16])

    def test_continuous_transforms_track_disk_center(self):
        store = AnalyticStore(disk)
        dc = np.array([128.37, 127.71])
        rng = np.random.default_rng(1)
        # References are the rasterized centroids, which differ from the analytic
        # center by ~1e-3 px of discretization error.
        ys, xs = np.nonzero(store.full_mask)
        ref_m = np.array([(xs + 0.5).mean(), (ys + 0.5).mean()]) / SCALE
        cov = store.images[0]
        g = np.arange(store.W) + 0.5
        ref_i = np.array([(cov * g[None, :]).sum(), (cov * g[:, None]).sum()]) / cov.sum()
        err_m, err_i = [], []
        for _ in range(200):
            c = dc + rng.uniform(-6, 6, 2)
            A = S.patch_matrix(angle=rng.uniform(0, 2 * math.pi), flip=bool(rng.integers(2)))
            img, mask = S.sample_patch(store, 0, tuple(c), A)
            expected = A.T @ (dc - c) + S.PATCH / 2      # disk center in patch coords
            ys, xs = np.nonzero(mask)
            mc = np.array([(xs + 0.5).mean(), (ys + 0.5).mean()]) / SCALE
            g = (np.arange(S.PATCH) + 0.5)
            ic = np.array([(img * g[None, :]).sum(), (img * g[:, None]).sum()]) / img.sum()
            # Map errors back to the source frame, where a convention offset is constant.
            err_m.append(A @ (mc - expected) - (ref_m - dc))
            err_i.append(A @ (ic - expected) - (ref_i - dc))
        err_m, err_i = np.array(err_m), np.array(err_i)
        # Nearest-neighbor mask resampling is bounded by half a hi-res pixel (1/32 px);
        # near-axis-aligned angles approach that bound, but it is unbiased.
        self.assertLess(np.abs(err_m).max(), 1 / 32)
        self.assertLess(np.abs(err_i).max(), 0.01)
        # A half-hi-res-pixel convention error would bias these by ~1/32 px.
        self.assertLess(np.abs(err_m.mean(0)).max(), 0.0005)
        self.assertLess(np.abs(err_i.mean(0)).max(), 0.0005)

    def test_detects_convention_error(self):
        """Sanity check of the test itself: a 1/32 px shifted mask lookup must fail."""
        store = AnalyticStore(disk)
        dc = np.array([128.37, 127.71])
        rng = np.random.default_rng(3)
        errs = []
        for _ in range(48):
            c = dc + rng.uniform(-6, 6, 2)
            A = S.patch_matrix(angle=rng.uniform(0, 2 * math.pi), flip=bool(rng.integers(2)))
            mask = S.sample_mask(store, 0, (c[0] + 1 / 32, c[1] + 1 / 32), A)
            ys, xs = np.nonzero(mask)
            mc = np.array([(xs + 0.5).mean(), (ys + 0.5).mean()]) / SCALE
            errs.append(A @ (mc - (A.T @ (dc - c) + S.PATCH / 2)))
        self.assertGreater(np.abs(np.mean(errs, 0)).min(), 0.02)

    def test_fast_mask_warp_matches_reference(self):
        """cv2 nearest warp == per-pixel floor(16 s), up to rare fixed-point ties."""
        store = AnalyticStore(blob)
        rng = np.random.default_rng(4)
        differing = 0
        for k in range(60):
            c = (rng.uniform(118, 140), rng.uniform(118, 140))
            if k % 3 == 0:
                A = S.patch_matrix(flip=bool(rng.integers(2)), quarter_turns=int(rng.integers(4)))
                c = (float(round(c[0])), float(round(c[1])))
            else:
                A = S.patch_matrix(angle=rng.uniform(0, 2 * math.pi), flip=bool(rng.integers(2)))
            f = S.sample_mask(store, 0, c, A)
            r = S.sample_mask_reference(store, 0, c, A)
            if k % 3 == 0:
                np.testing.assert_array_equal(f, r)
            differing += int((f != r).sum())
        self.assertLess(differing, 20)

    def test_context_extends_input_around_same_center(self):
        store = AnalyticStore(blob)
        rng = np.random.default_rng(5)
        for k in range(12):
            c = (rng.uniform(120, 136), rng.uniform(120, 136))
            A = S.patch_matrix(angle=rng.uniform(0, 2 * math.pi), flip=bool(rng.integers(2))) \
                if k % 2 else S.patch_matrix(quarter_turns=int(rng.integers(4)))
            base = S.sample_image(store, 0, c, A)
            ext = S.sample_image(store, 0, c, A, context=8)
            self.assertEqual(ext.shape, (64, 64))
            np.testing.assert_allclose(ext[8:-8, 8:-8], base, atol=1e-6)

    def test_mask_window_matches_block_store(self):
        """The block-sparse cache must reproduce the tiled mask exactly."""
        try:
            store = S.SourceStore()
            store.images
        except FileNotFoundError:
            self.skipTest("cache not built")
        import json
        import os
        from PIL import Image
        rng = np.random.default_rng(2)
        part_dir = {}
        for s in store.scenes:
            part_dir.setdefault(s["part_index"], s["dir"])
        for part in rng.choice(sorted(part_dir), 3, replace=False):
            with open(os.path.join(part_dir[part], "mask", "index.json")) as f:
                tiles = [t for t in json.load(f)["tiles"] if t["foreground_pixels"] > 0]
            t = tiles[rng.integers(len(tiles))]
            with Image.open(os.path.join(part_dir[part], "mask", t["file"])) as im:
                ref = np.array(im.convert("1"), np.uint8)
            x0, y0 = int(rng.integers(0, 1500)), int(rng.integers(0, 1500))
            got = store.mask_window(int(part), t["x"] + x0, t["y"] + y0, 517, 389)
            np.testing.assert_array_equal(got, ref[y0:y0 + 389, x0:x0 + 517])


class CacheLookupTest(unittest.TestCase):
    def setUp(self):
        try:
            self.store = S.SourceStore()
            self.store.boundary_points(0)
        except FileNotFoundError:
            self.skipTest("cache / boundary points not built")

    def test_mask_points_match_window(self):
        rng = np.random.default_rng(7)
        for part in rng.choice(300, 4, replace=False):
            bp = self.store.boundary_points(int(part))
            cx, cy = (bp[rng.integers(len(bp))] * 16).astype(int)
            x0, y0 = cx - 300, cy - 300
            win = self.store.mask_window(int(part), x0, y0, 600, 600)
            mx = rng.integers(x0, x0 + 600, 5000)
            my = rng.integers(y0, y0 + 600, 5000)
            np.testing.assert_array_equal(self.store.mask_points(int(part), mx, my),
                                          win[my - y0, mx - x0])

    def test_boundary_points_match_direct_extraction(self):
        from edgesr.metrics import contour_points
        rng = np.random.default_rng(8)
        for part in rng.choice(300, 3, replace=False):
            bp = self.store.boundary_points(int(part))
            cx, cy = (bp[rng.integers(len(bp))] * 16).astype(int)
            x0, y0 = cx - 400, cy - 400
            win = self.store.mask_window(int(part), x0, y0, 800, 800)
            direct = contour_points(win.astype(np.float32)) + np.array([x0, y0]) / 16
            lo, hi = np.array([x0, y0]) / 16 + 2, np.array([x0, y0]) / 16 + 48
            sel = lambda p: p[np.all((p >= lo) & (p <= hi), 1)]
            a = np.unique(np.round(sel(direct) * 64).astype(np.int64), axis=0)
            b = np.unique(np.round(sel(bp) * 64).astype(np.int64), axis=0)
            np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()


class CropPointTest(unittest.TestCase):
    """Point-supervision GT for CFI crops (edgesr/crops.py)."""

    def setUp(self):
        try:
            self.store = S.SourceStore()
            self.store.boundary_points(0)
        except FileNotFoundError:
            self.skipTest("cache / boundary points not built")
        from edgesr import crops as C
        self.C = C
        self.el = S.eligible_scenes(self.store, self.store.scene_indices("train"),
                                    S.DEFAULT_SAMPLER["weights"])

    def _crop(self, seed, **cfg):
        C = self.C
        rng = np.random.default_rng(seed)
        geo = C.crop_geometry(self.store, np.random.default_rng(seed), self.el, S.DEFAULT_SAMPLER,
                              192, cfg.get("p_anywhere", 0.0))
        it = C.sample_crop(self.store, rng, self.el, S.DEFAULT_SAMPLER, S.DEFAULT_AUG,
                           {**C.DEFAULT_CROP, **cfg})
        return geo, it

    def test_occupancy_and_edge_field_are_consistent(self):
        C = self.C
        flips = []
        for seed in range(8):
            (scene, center, A, _), it = self._crop(seed)
            part = self.store.scenes[scene]["part_index"]
            # occupancy GT == mask lookup at the mapped point
            src = C.crop_to_source(it["q"].astype(np.float64), center, A, 192)
            occ = self.store.mask_points(part, np.floor(src[:, 0] * 16), np.floor(src[:, 1] * 16))
            np.testing.assert_array_equal(occ, it["occ"].astype(np.uint8))
            # stepping along the GT direction by the GT distance crosses the boundary
            sel = (it["dist"] > 0.1) & (it["dist"] < 1.0)
            q, d, di = it["q"][sel].astype(np.float64), it["dist"][sel], it["dir"][sel]
            b = C.crop_to_source(q + (d - 0.05)[:, None] * di, center, A, 192)
            a = C.crop_to_source(q + (d + 0.05)[:, None] * di, center, A, 192)
            ob = self.store.mask_points(part, *np.floor(b * 16).T)
            oa = self.store.mask_points(part, *np.floor(a * 16).T)
            flips.append((ob != oa).mean())
        self.assertGreater(np.mean(flips), 0.995)

    def test_crop_geometry_round_trip_and_frame_overhang(self):
        C = self.C
        for seed in range(6):
            (scene, center, A, cat), it = self._crop(100 + seed, p_anywhere=1.0)
            P = np.random.default_rng(seed).uniform(0, 192, (50, 2))
            np.testing.assert_allclose(C.source_to_crop(C.crop_to_source(P, center, A, 192),
                                                        center, A, 192), P, atol=1e-9)
            self.assertEqual(it["x"].shape, (192, 192))
            self.assertTrue(np.isfinite(it["x"]).all())
