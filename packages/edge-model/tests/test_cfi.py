"""CFI invariants: sparse window inference == dense evaluation; point decoder == lattice
decoder. These are what make the model seam-free by construction."""

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from edgesr.cfi import CFI, SCALE
from edgesr.cfi_infer import infer


def _randomize(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.2)
    return model


class CFITest(unittest.TestCase):
    def _dense(self, model, img, bl):
        x = torch.from_numpy(img)[None, None]
        b = torch.tensor([bl])
        xin, ctx, occ4, band4 = model.context(x, b)
        ctx_up = F.interpolate(ctx, scale_factor=4, mode="bilinear", align_corners=False)
        return model.fine_features(xin, ctx_up)

    def test_sparse_equals_dense(self):
        torch.set_num_threads(4)
        rng = np.random.default_rng(0)
        img = rng.uniform(0, 1, (128, 128)).astype(np.float32)
        band = np.zeros((128, 128), bool)
        band[40:90, 50:52] = True
        band[60:62, 30:100] = True
        for dec in ("implicit", "block"):
            for dil in ([1, 1, 2, 3], [1, 2, 4, 8]):
                model = _randomize(CFI({"decoder": dec, "fine_dilations": dil})).eval()
                with torch.no_grad():
                    feat = self._dense(model, img, 0.7)[0]
                    ys, xs = np.nonzero(band)
                    ys_t, xs_t = torch.from_numpy(ys), torch.from_numpy(xs)
                    dense = model.decode_pixels(feat, torch.zeros_like(ys_t), ys_t, xs_t)[..., 0]
                    out = infer(model, img, 0.7, band, cell=16, contours=False, frame_pad=0)
                n, py, px, cy, cx = out["band_index"]
                sy, sx = (cy[n] * 16 + py).numpy(), (cx[n] * 16 + px).numpy()
                order = {(a, b): i for i, (a, b) in enumerate(zip(ys, xs))}
                idx = np.array([order[(a, b)] for a, b in zip(sy, sx)])
                np.testing.assert_allclose(out["logits"].numpy(), dense[idx].numpy(),
                                           rtol=1e-4, atol=1e-4)

    def test_seed_mode_sparse_equals_dense(self):
        from edgesr.seed import seed_full
        torch.set_num_threads(4)
        rng = np.random.default_rng(2)
        img = np.full((128, 128), 0.8, np.float32)
        img[40:90, 30:100] = 0.1                                  # a dark part
        img += rng.normal(0, 0.01, img.shape).astype(np.float32)
        band = np.zeros((128, 128), bool)
        band[38:42, 30:100] = True
        band[40:90, 98:102] = True
        model = _randomize(CFI({"context": "seed", "decoder": "block"})).eval()
        with torch.no_grad():
            sd = torch.from_numpy(seed_full(img)["inputs"])[None]
            feat = model.fine(sd)[0]
            ys, xs = np.nonzero(band)
            ys_t, xs_t = torch.from_numpy(ys), torch.from_numpy(xs)
            dense = model.decode_pixels(feat, torch.zeros_like(ys_t), ys_t, xs_t)[..., 0]
            out = infer(model, img, 0.8, band, cell=16, contours=True, frame_pad=0)
        n, py, px, cy, cx = out["band_index"]
        sy, sx = (cy[n] * 16 + py).numpy(), (cx[n] * 16 + px).numpy()
        order = {(a, b): i for i, (a, b) in enumerate(zip(ys, xs))}
        idx = np.array([order[(a, b)] for a, b in zip(sy, sx)])
        np.testing.assert_allclose(out["logits"].numpy(), dense[idx].numpy(), rtol=1e-4, atol=1e-4)

    def test_min_area_filter(self):
        from edgesr.cfi_infer import filter_small_regions, filter_small_regions_exact
        S16 = 16 * 16                      # hi-res canvas side for a 16 px cell
        ins = np.zeros((3, S16, S16), bool)   # cells (0,0), (0,1), (1,0)
        cy, cx = np.array([0, 0, 1]), np.array([0, 1, 0])
        # speck 2x2 px = 4 px^2 fully inside cell 0 -> removed
        ins[0, 32:64, 32:64] = True
        # blob 3 px wide x 4 px tall straddling the cell 0 / cell 1 border: 12 px^2 total,
        # 6 px^2 in each cell -> kept only if merged across the border
        ins[0, 128:192, 208:256] = True
        ins[1, 128:192, 0:48] = True
        # region touching cell 1's right edge (inactive neighbour): 1 px^2 but open -> kept
        ins[1, 100:116, 240:256] = True
        # cell 2 is solid object with a 1 px^2 hole -> hole filled
        ins[2] = True
        ins[2, 100:116, 100:116] = False
        for fn in (filter_small_regions_exact, filter_small_regions):
            out = np.asarray(fn(ins, cy, cx, 16, 8.0))
            self._check_filtered(out)
        # a sub-pixel speck inside one pixel block is removed by the fast filter
        ins2 = np.ones((1, S16, S16), bool)
        ins2[0, 36:40, 36:40] = False
        self.assertTrue(np.asarray(filter_small_regions(ins2, np.array([0]), np.array([0]), 16, 8.0)).all())

    def _check_filtered(self, out):
        self.assertFalse(out[0, 32:64, 32:64].any())
        self.assertTrue(out[0, 128:192, 208:256].all() and out[1, 128:192, 0:48].all())
        self.assertTrue(out[1, 100:116, 240:256].all())
        self.assertTrue(out[2].all())

    def test_point_decoder_matches_lattice(self):
        rng = np.random.default_rng(1)
        for dec in ("implicit", "block"):
            model = _randomize(CFI({"decoder": dec}), 3).eval()
            feat = torch.randn(1, model.cfg["fine_width"], 12, 12)
            ys, xs = torch.tensor([4, 6]), torch.tensor([5, 7])
            with torch.no_grad():
                lat = model.decode_pixels(feat, torch.zeros_like(ys), ys, xs)   # (2,256,k)
                t = (torch.arange(SCALE) + 0.5) / SCALE
                vv, uu = torch.meshgrid(t, t, indexing="ij")
                for i in range(2):
                    qx = xs[i] + uu.reshape(-1)
                    qy = ys[i] + vv.reshape(-1)
                    pts = model.decode_points(feat[0], qx.double().float(), qy.float())
                    np.testing.assert_allclose(pts.numpy(), lat[i].numpy(), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
