"""Coarse model: macro-scale (1/4 resolution) part segmentation for the fine edge model.

Input: the full sensor reading resampled to 512 x 512 (model_reading; a 2048^2 frame is
downsampled 4x, other aspect ratios are squashed). Training covers squashed frames
(aspect 5:4 to 1.9:1 either way) and circular regions of interest (an ellipse after
squashing; outside is zero or a dark unlit backlight). Any size works (padded to a
multiple of 32). Output, at the input resolution:
  part     logit: any part (union of visible part masks; attached debris included,
           loose debris excluded)
  contact  logit: part pixels within `contact_px` of a different part (touching or
           stacked parts), used to split the part mask into instances
  sdf      signed distance to the part boundary in macro px (+ inside), clipped to
           +-sdf_clip; used for a sub-pixel boundary and the fine model's band
  center   (instance_head "center", Panoptic-DeepLab style) heatmap of part centers
           (Gaussian, sigma center_sigma px, at each part's mass center) and, for part
           pixels, the offset to their part's center / offset_scale. Instances group part
           pixels by the nearest detected center, so occluded or thinly connected pieces
           of one part stay one instance and touching parts split without a continuous
           contact band.

Coordinates: macro pixel j covers source pixels [4j, 4j + 4) (area downsampling), so its
center is at 4j + 2, consistent with bilinear upsampling (align_corners=False).

Input normalization mirrors the classical seed (seed.py / adapter.py): a quadratic
backlight field is fitted to the 12 x 12 tile p90 levels of the macro reading; the
network sees reading / field and the raw reading (for saturation and exposure).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .coarse_np import *  # noqa: F401,F403  (re-exported: the torch-free helpers)
from .coarse_np import coarse_inputs, in_channels, maps_from_output, n_outputs
from .coarse_np import DEFAULT_COARSE  # noqa: F401


class ConvBN(nn.Sequential):
    def __init__(self, cin, cout, k=3, stride=1, dilation=1, groups=1, act=True):
        pad = dilation * (k // 2)
        layers = [nn.Conv2d(cin, cout, k, stride, pad, dilation, groups, bias=False),
                  nn.BatchNorm2d(cout)]
        if act:
            layers.append(nn.SiLU())
        super().__init__(*layers)


class IRBlock(nn.Module):
    """Inverted residual: dw3x3 -> BN -> pw expand -> SiLU -> pw -> BN(zero init)."""

    def __init__(self, c, dilation=1, expand=2):
        super().__init__()
        self.dw = ConvBN(c, c, 3, dilation=dilation, groups=c, act=False)
        self.pw1 = nn.Conv2d(c, c * expand, 1)
        self.pw2 = ConvBN(c * expand, c, 1, act=False)
        nn.init.zeros_(self.pw2[1].weight)

    def forward(self, x):
        return x + self.pw2(F.silu(self.pw1(self.dw(x))))


class DenseBlock(nn.Module):
    def __init__(self, c, dilation=1, expand=None):
        super().__init__()
        self.c1 = ConvBN(c, c, 3, dilation=dilation)
        self.c2 = ConvBN(c, c, 3, act=False)
        nn.init.zeros_(self.c2[1].weight)

    def forward(self, x):
        return x + self.c2(self.c1(x))


class CoarseNet(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg = {**DEFAULT_COARSE, **(cfg or {})}
        W, B = cfg["widths"], cfg["blocks"]
        Block = IRBlock if cfg["block"] == "ds" else DenseBlock
        self.stem = ConvBN(in_channels(cfg), W[0], 3)
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        nlev = len(W)
        for i in range(nlev):
            if i > 0:
                self.down.append(ConvBN(W[i - 1], W[i], 3, stride=2))
            dil = cfg["dilations"] if i >= nlev - 2 else [1]
            self.enc.append(nn.Sequential(*[Block(W[i], dil[j % len(dil)], cfg["expand"])
                                            for j in range(B[i])]))
        self.lat = nn.ModuleList([nn.Conv2d(W[i + 1], W[i], 1) for i in range(nlev - 1)])
        self.dec = nn.ModuleList([nn.Sequential(*[Block(W[i], 1, cfg["expand"])
                                                  for _ in range(cfg["dec_blocks"][i])])
                                  for i in range(nlev - 1)])
        self.head = nn.Conv2d(W[0], n_outputs(cfg), 1)
        self.multiple = 2 ** (nlev - 1)

    def forward(self, x):
        feats = []
        f = self.stem(x)
        for i, enc in enumerate(self.enc):
            if i > 0:
                f = self.down[i - 1](f)
            f = enc(f)
            feats.append(f)
        g = feats[-1]
        for i in range(len(feats) - 2, -1, -1):
            up = F.interpolate(g, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            g = self.dec[i](feats[i] + self.lat[i](up))
        # (B, 3|6, H, W): part logit, contact logit, sdf / clip [, center heat, offset x, y]
        return self.head(g)


def pad_to(x, m):
    """Edge-replicate pad (B, C, H, W) to a multiple of m; returns padded, (h, w)."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    return x, (h, w)


@torch.no_grad()
def predict_maps(model, reading, device="cpu", valid=None, aspect=1.0):
    """Macro reading (H, W) in [0, 1] -> dict of float32 maps (H, W): part and contact
    probabilities, part logit, sdf (macro px) and, with a center head, center heat and
    offsets (px). valid: optional (H, W) bool region of interest; aspect: width / height
    of the original frame (both used only by models with aux_inputs)."""
    cfg = model.cfg
    if valid is not None:
        reading = np.where(valid, reading, 0.0).astype(np.float32)
    x = torch.from_numpy(coarse_inputs(reading, cfg["in_mode"], cfg.get("aux_inputs", False),
                                       valid, aspect))[None].to(device)
    x, (h, w) = pad_to(x, model.multiple)
    out = model(x)[0, :, :h, :w].float().cpu().numpy()
    m = maps_from_output(out, cfg)
    if valid is not None:                   # nothing is a part outside the region of interest
        m["part_logit"] = np.where(valid, m["part_logit"], np.minimum(m["part_logit"], -8.0))
        m["part"] = 1 / (1 + np.exp(-m["part_logit"]))
    return m
