"""Coarse model pieces that need no dataset: targets, instance split, boundary metric,
model shapes."""

import unittest

import numpy as np

from edgesr.bench_coarse import gt_points, pred_points
from edgesr.coarse import (CoarseNet, instances, instances_center, part_centers, predict_maps,
                           targets)


def two_squares():
    inst = np.zeros((64, 64), np.uint8)
    inst[16:48, 8:32] = 1
    inst[16:48, 32:56] = 2          # touching at x = 32
    return inst


class CoarseTest(unittest.TestCase):
    def test_targets(self):
        part, contact, sdf = targets(two_squares(), contact_px=2, sdf_clip=6.0)
        self.assertEqual(part.sum(), 32 * 48)
        cols = np.nonzero(contact.any(0))[0]
        self.assertEqual((cols.min(), cols.max()), (30, 33))     # 2 px each side of the seam
        self.assertTrue((sdf[part > 0] > 0).all() and (sdf[part == 0] < 0).all())
        self.assertAlmostEqual(float(sdf[32, 20]), 6.0)         # clipped deep inside
        self.assertAlmostEqual(float(sdf[16, 20]), 0.5)          # boundary pixel

    def test_instances_split_touching(self):
        part, contact, _ = targets(two_squares())
        lab = instances(part, contact)
        self.assertEqual(lab.max(), 2)
        self.assertEqual(len(np.unique(lab[16:48, 8:30])), 1)
        self.assertNotEqual(lab[30, 20], lab[30, 44])
        self.assertTrue(((lab > 0) == (part > 0)).all())         # every part pixel assigned
        self.assertEqual(instances(part, np.zeros_like(part)).max(), 1)   # no contact: merged

    def test_center_head_targets_and_grouping(self):
        inst = two_squares()
        inst[16:48, 40:44] = 0                      # part 2 now has two disconnected pieces
        c = part_centers(inst)
        self.assertAlmostEqual(c[1][0], 19.5)
        part, _, _, heat, offx, offy = targets(inst, centers=c, sigma=5.0, offset_scale=64.0)
        self.assertAlmostEqual(float(heat[int(round(c[1][1])), int(round(c[1][0]))]), 1.0, places=1)
        # exact targets decode to the GT instances, reattaching the split piece
        lab = instances_center(part, heat, np.stack([offx, offy]) * 64.0)
        self.assertEqual(lab.max(), 2)
        self.assertEqual(lab[20, 34], lab[20, 50])
        self.assertNotEqual(lab[20, 10], lab[20, 34])

    def test_boundary_points(self):
        # disk rendered by exact coverage; majority-label GT and a coverage logit
        S, ss, cx, cy, r = 96, 16, 48.3, 47.6, 20.25
        g = (np.arange(S * ss) + 0.5) / ss
        cov = ((g[None] - cx) ** 2 + (g[:, None] - cy) ** 2 < r * r).reshape(S, ss, S, ss).mean((1, 3))
        gp, lab = gt_points((cov > 0.5).astype(np.uint8))
        self.assertTrue((lab == 1).all())
        self.assertLess(np.abs(np.hypot(gp[:, 0] - cx, gp[:, 1] - cy) - r).mean(), 0.25)
        pp = pred_points((cov - 0.5) * 8)
        self.assertLess(np.abs(np.hypot(pp[:, 0] - cx, pp[:, 1] - cy) - r).max(), 0.15)
        # dense sampling: 0.25 px grid, up to ~0.35 px apart on diagonals
        self.assertLess(np.diff(np.sort(np.arctan2(pp[:, 1] - cy, pp[:, 0] - cx))).max() * r, 0.4)

    def test_inference_api_aspect_and_roi(self):
        import os
        import tempfile

        import torch

        from edgesr.coarse_infer import FORMAT, CoarseModel, inscribed_circle
        net = CoarseNet({"block": "dense", "in_mode": "raw", "aux_inputs": True})
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pt")
            torch.save({"format": FORMAT, "model_config": net.cfg, "state_dict": net.state_dict()}, path)
            cm = CoarseModel(path)
        img = np.full((300, 500), 50000, np.uint16)
        img[100:200, 150:300] = 3000
        roi = inscribed_circle(img.shape)
        out = cm(img, valid_mask=roi)
        self.assertEqual(out["part_logit"].shape, (512, 512))
        lg = out["part_logit"]
        for c in (lg[:40, :40], lg[:40, -40:], lg[-40:, :40]):      # outside the inscribed circle
            self.assertTrue((c <= -8).all())
        self.assertEqual(cm.full_res_logit(out, img.shape).shape, (300, 500))
        self.assertEqual(cm.band(out, img.shape).shape, (300, 500))

    def test_model_any_size(self):
        m = CoarseNet().eval()
        out = predict_maps(m, np.full((97, 130), 0.8, np.float32))
        self.assertEqual(out["part"].shape, (97, 130))
        self.assertEqual(out["sdf"].shape, (97, 130))


if __name__ == "__main__":
    unittest.main()
