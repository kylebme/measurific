"""Coarse–Fine–Implicit (CFI) full-image model (PLAN_v2.md).

All parts are translation-equivariant convolutions, so evaluating the fine branch on
gathered windows (margin >= its receptive field) equals dense evaluation exactly.

Coordinates: source pixel i covers [i, i+1). 1/4-res cell j covers source pixels
[4j, 4j+4), i.e. its center is at 4j + 2 (pixel_unshuffle stem), so bilinear
upsampling with align_corners=False is consistent.

  coarse(x, b)            -> ctx (B, Cc, H/4, W/4), occ4 (B,1,H/4,W/4), band4 (B,1,H/4,W/4)
  fine(x, b, ctx_up)      -> feat (B, Cf, H, W)   (valid >= R px from any crop edge)
  decode_*                -> per hi-res sub-pixel logits: occupancy (+ edge field)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ChannelNorm

SCALE = 16

DEFAULT_CFI = {
    "coarse_widths": [24, 48, 64],     # at 1/4, 1/8, 1/16
    "coarse_blocks": [0, 2, 2],        # blocks per level (encoder); decoder uses 1/level
    "ctx_channels": 16,
    "block": "dense",                  # "dense" (3x3+3x3 residual) | "ds" (depthwise-sep.)
    "norm": "ln",                      # pre-norm channel LayerNorm in blocks | None
    "fine_width": 24,
    "fine_dilations": [1, 1, 2, 3],    # R = 1 + sum(dilations) (+1 per dense 2nd conv)
    "fine_expand": 2,
    "decoder": "block",                # "block" | "implicit"
    "dec_hidden": 32,
    "dec_fourier": [1, 2, 4, 8],       # periodic (1 px) sub-pixel encoding frequencies
    "edge_head": True,                 # distance + direction to the nearest edge
    "context": "coarse",               # "coarse" (learned 1/4-res branch) | "seed" (classical
                                       # threshold seed inputs; no coarse network, PLAN_v2 B)
}


class DenseBlock(nn.Module):
    """x + conv3x3(silu(conv3x3_dilated(norm(x)))). RF radius = dilation + 1."""

    def __init__(self, c, dilation=1, norm="ln"):
        super().__init__()
        self.n = ChannelNorm(c) if norm else nn.Identity()
        self.c1 = nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)
        self.radius = dilation + 1

    def forward(self, x):
        return x + self.c2(F.silu(self.c1(self.n(x))))


def make_block(kind, c, dilation=1, norm="ln", expand=2):
    return DenseBlock(c, dilation, norm) if kind == "dense" else DSBlock(c, dilation, expand, norm)


class DSBlock(nn.Module):
    def __init__(self, c, dilation=1, expand=2, norm="ln"):
        super().__init__()
        self.n = ChannelNorm(c) if norm else nn.Identity()
        self.radius = dilation
        self.dw = nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation, groups=c)
        self.pw1 = nn.Conv2d(c, c * expand, 1)
        self.pw2 = nn.Conv2d(c * expand, c, 1)
        nn.init.zeros_(self.pw2.weight)
        nn.init.zeros_(self.pw2.bias)

    def forward(self, x):
        return x + self.pw2(F.silu(self.pw1(self.dw(self.n(x)))))


def _prep(x, b):
    """x: (B,1,H,W) reading in [0,1]; b: (B,) backlight -> 2-channel normalized input."""
    b = b.to(x.dtype).view(-1, 1, 1, 1).clamp_min(1e-3)
    return torch.cat([(x / b - 0.5) * 2.0, ((b - 0.65) * 4.0).expand_as(x)], 1)


class Coarse(nn.Module):
    """1/4-res context: pixel_unshuffle(4) stem (1x1), light work at 1/4, most at 1/8, 1/16."""

    def __init__(self, widths, blocks, ctx_channels, kind="dense", norm="ln"):
        super().__init__()
        w0, w1, w2 = widths
        self.stem = nn.Conv2d(2 * 16, w0, 1)
        self.enc0 = nn.Sequential(*[make_block(kind, w0, 1, norm) for _ in range(blocks[0])])
        self.down1 = nn.Conv2d(w0 * 4, w1, 1)
        self.enc1 = nn.Sequential(*[make_block(kind, w1, 1, norm) for _ in range(blocks[1])])
        self.down2 = nn.Conv2d(w1 * 4, w2, 1)
        dil = [1, 2, 4, 1, 2, 4]
        self.enc2 = nn.Sequential(*[make_block(kind, w2, dil[i], norm) for i in range(blocks[2])])
        self.up2 = nn.Conv2d(w2, w1, 1)
        self.dec1 = make_block(kind, w1, 1, norm)
        self.up1 = nn.Conv2d(w1, w0, 1)
        self.dec0 = make_block(kind, w0, 1, norm)
        self.out_norm = ChannelNorm(w0) if norm else nn.Identity()
        self.ctx = nn.Conv2d(w0, ctx_channels, 1)
        self.heads = nn.Conv2d(w0, 2, 1)        # coarse occupancy, band

    def forward(self, xin):
        h, w = xin.shape[-2:]
        assert h % 16 == 0 and w % 16 == 0, "pad images to a multiple of 16"
        f0 = self.enc0(F.silu(self.stem(F.pixel_unshuffle(xin, 4))))
        f1 = self.enc1(self.down1(F.pixel_unshuffle(f0, 2)))
        f2 = self.enc2(self.down2(F.pixel_unshuffle(f1, 2)))
        g1 = self.dec1(f1 + self.up2(F.interpolate(f2, scale_factor=2, mode="bilinear",
                                                   align_corners=False)))
        g0 = self.dec0(f0 + self.up1(F.interpolate(g1, scale_factor=2, mode="bilinear",
                                                   align_corners=False)))
        g0 = self.out_norm(g0)
        hd = self.heads(g0)
        return self.ctx(g0), hd[:, :1], hd[:, 1:]


class Fine(nn.Module):
    """Source-res local geometry. Outputs features and a per-pixel band logit."""

    def __init__(self, in_ch, width, dilations, expand, kind="dense", norm="ln"):
        super().__init__()
        self.inp = nn.Conv2d(in_ch, width, 3, padding=1)
        blocks = [make_block(kind, width, d, norm, expand) for d in dilations]
        self.blocks = nn.Sequential(*blocks)
        self.norm = ChannelNorm(width) if norm else nn.Identity()
        self.band = nn.Conv2d(width, 1, 1)
        self.radius = 1 + sum(b.radius for b in blocks)

    def forward(self, x):
        f = self.norm(self.blocks(self.inp(x)))
        return f, self.band(f)


def fourier_subpixel(u, v, freqs):
    """Periodic (period 1 px) encoding of sub-pixel position; continuous across pixels."""
    out = []
    for k in freqs:
        a, b = 2 * math.pi * k * u, 2 * math.pi * k * v
        out += [torch.sin(a), torch.cos(a), torch.sin(b), torch.cos(b)]
    return torch.stack(out, -1)


class CFI(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        cfg = {**DEFAULT_CFI, **(cfg or {})}
        self.cfg = cfg
        self.seed_mode = cfg["context"] == "seed"
        if not self.seed_mode:
            self.coarse = Coarse(cfg["coarse_widths"], cfg["coarse_blocks"], cfg["ctx_channels"],
                                 cfg["block"], cfg["norm"])
        fin = 3 if self.seed_mode else 2 + cfg["ctx_channels"]
        self.fine = Fine(fin, cfg["fine_width"], cfg["fine_dilations"],
                         cfg["fine_expand"], cfg["block"], cfg["norm"])
        self.k_out = 4 if cfg["edge_head"] else 1      # occ, dist, dir_x, dir_y
        wf, hdn = cfg["fine_width"], cfg["dec_hidden"]
        if cfg["decoder"] == "block":
            self.block = nn.Conv2d(wf, SCALE * SCALE * self.k_out, 3, padding=1)
        else:
            nf = 4 * len(cfg["dec_fourier"])
            self.proj = nn.Conv2d(wf, hdn, 1)            # first MLP layer, feature part
            self.pe = nn.Linear(nf, hdn, bias=False)    # first MLP layer, position part
            self.l2 = nn.Linear(hdn, hdn)
            self.l3 = nn.Linear(hdn, self.k_out)

    @property
    def radius(self):
        """Receptive-field radius (source px) of the per-pixel decoder inputs."""
        return self.fine.radius + 1

    def context(self, x, b):
        xin = _prep(x, b)
        return (xin,) + self.coarse(xin)

    def fine_features(self, xin, ctx_up):
        """-> (features, per-pixel band logit)."""
        return self.fine(torch.cat([xin, ctx_up], 1))

    # -- decoding a full 16x16 sub-pixel lattice for selected pixels --------------
    def _lattice(self, device, dtype):
        t = (torch.arange(SCALE, device=device, dtype=dtype) + 0.5) / SCALE   # in (0,1)
        vv, uu = torch.meshgrid(t, t, indexing="ij")
        return uu.reshape(-1), vv.reshape(-1)                                # (256,)

    def decode_pixels(self, feat, n, ys, xs, P=None, occupancy_only=False):
        """Logits (M, 256, k_out) for the 16x16 sub-pixels of pixels (n, ys, xs) of
        feat (N, C, H, W); neighbours must exist (pixels >= 1 from the feat border).
        For the implicit decoder, P = self.project(feat) may be passed precomputed."""
        offs = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        if self.cfg["decoder"] == "block":
            nb = torch.stack([feat[n, :, ys + dy, xs + dx] for dy, dx in offs], -1)
            nb = nb.reshape(len(ys), -1)                            # (M, C*9), c-major
            w = self.block.weight.reshape(self.block.out_channels, -1)
            bias = self.block.bias
            k = self.k_out
            if occupancy_only:        # pixel_shuffle channel order: k*256 + sy*16 + sx
                w, bias, k = w[:SCALE * SCALE], bias[:SCALE * SCALE], 1
            out = nb @ w.t() + bias
            return out.view(len(ys), k, SCALE * SCALE).transpose(1, 2)
        if P is None:
            P = self.project(feat)
        nb = torch.stack([P[n, :, ys + dy, xs + dx] for dy, dx in offs], 1)  # (M, 9, hdn)
        wq, pe = self._interp_tables(feat.device, feat.dtype)               # (256,9),(256,hdn)
        h = torch.matmul(wq, nb) + pe                                       # (M, 256, hdn)
        return self.l3(F.silu(self.l2(F.silu(h))))

    def project(self, feat):
        return self.proj(feat)

    def _interp_tables(self, device, dtype):
        """Bilinear weights of the 256 sub-pixel positions over the 3x3 neighbourhood
        (pixel centers at i + 0.5), and the position term of the first MLP layer."""
        u, v = self._lattice(device, dtype)
        wq = torch.zeros(SCALE * SCALE, 9, device=device, dtype=dtype)
        # position relative to the center pixel's center, in [-0.5, 0.5)
        du, dv = u - 0.5, v - 0.5
        for (wy, oy) in ((1 - dv.abs(), 0), (dv.abs(), torch.sign(dv))):
            for (wx, ox) in ((1 - du.abs(), 0), (du.abs(), torch.sign(du))):
                oyi = (oy if torch.is_tensor(oy) else torch.zeros_like(dv)).long()
                oxi = (ox if torch.is_tensor(ox) else torch.zeros_like(du)).long()
                idx = (oyi + 1) * 3 + (oxi + 1)
                wq.scatter_add_(1, idx[:, None], (wy * wx)[:, None])
        pe = self.pe(fourier_subpixel(u, v, self.cfg["dec_fourier"]))
        return wq, pe

    def decode_points(self, feat, qx, qy, bidx=None):
        """Logits (N, k_out) at continuous points (qx, qy) (feat pixel coords, pixel
        centers at i + 0.5) of feat (B, C, H, W); bidx selects the batch item."""
        if feat.dim() == 3:
            feat = feat[None]
        if bidx is None:
            bidx = torch.zeros(len(qx), dtype=torch.long, device=feat.device)
        offs = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        if self.cfg["decoder"] == "block":
            px, py = qx.floor().long(), qy.floor().long()
            sx = ((qx - px) * SCALE).floor().long().clamp(0, SCALE - 1)
            sy = ((qy - py) * SCALE).floor().long().clamp(0, SCALE - 1)
            nb = torch.stack([feat[bidx, :, py + dy, px + dx] for dy, dx in offs], -1)
            nb = nb.reshape(len(qx), -1)                                  # (N, C*9)
            w = self.block.weight.reshape(self.k_out, SCALE * SCALE, -1)
            b = self.block.bias.view(self.k_out, SCALE * SCALE)
            sub = sy * SCALE + sx
            wsel = w[:, sub].permute(1, 0, 2)                              # (N, k, C*9)
            return torch.einsum("nkc,nc->nk", wsel, nb) + b[:, sub].t()
        P = self.proj(feat)
        u, v = qx - 0.5, qy - 0.5
        x0, y0 = u.floor().long(), v.floor().long()
        fx, fy = (u - x0).unsqueeze(1), (v - y0).unsqueeze(1)

        def g(yy, xx):
            return P[bidx, :, yy, xx]
        h = (g(y0, x0) * (1 - fx) * (1 - fy) + g(y0, x0 + 1) * fx * (1 - fy) +
             g(y0 + 1, x0) * (1 - fx) * fy + g(y0 + 1, x0 + 1) * fx * fy)
        h = h + self.pe(fourier_subpixel(qx - qx.floor(), qy - qy.floor(), self.cfg["dec_fourier"]))
        return self.l3(F.silu(self.l2(F.silu(h))))

    def dense_features(self, x, b, seed=None):
        """Training / whole-crop path: coarse outputs and full-res fine features.
        In seed mode `seed` holds the (B, 3, H, W) classical seed inputs and there are
        no coarse outputs (None)."""
        if self.seed_mode:
            feat, pband = self.fine(seed)
            return feat, pband, None, None
        xin, ctx, occ4, band4 = self.context(x, b)
        ctx_up = F.interpolate(ctx, scale_factor=4, mode="bilinear", align_corners=False)
        feat, pband = self.fine(torch.cat([xin, ctx_up], 1))
        return feat, pband, occ4, band4
