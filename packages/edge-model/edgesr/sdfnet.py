"""SDFNet: U-Net regressing a truncated signed distance to the part boundary (PLAN_v3).

Full-resolution encoder/decoder stages keep subpixel precision; the deep levels (stride up
to 16, optional self-attention there) give ~250 px of context for depth-of-field ramps,
plateaus and lighting. Heads (fp32, full resolution):
  sdf      T * tanh(raw / T), crop px, + inside the object
  log_b    Laplace scale of the sdf error (uncertainty; trained on the detached error)
  dir      direction to the nearest edge (2, unnormalized)
and a 1/4-resolution TSDF head (deep supervision; later the band proposal of a cascade).
`target: "occ"` replaces the sdf head by an occupancy logit (ablation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ChannelNorm

DEFAULT_SDFNET = {
    "kind": "sdf",
    "in_ch": 4,
    "widths": [32, 64, 128, 192, 256],
    "enc_blocks": [2, 2, 2, 2, 2],
    "dec_blocks": [1, 1, 1, 1],
    "attention": False,       # one self-attention block at the deepest level
    "trunc_px": 8.0,
    "target": "sdf",          # "sdf" | "occ" (ablation: same backbone, occupancy head)
    "groups": 8,
    "norm": "ln",             # "ln": per-pixel channel LayerNorm (local); "gn": GroupNorm, whose
                              # whole-image statistics couple every output to every input pixel
}


class ChannelLN(nn.Module):
    """LayerNorm over channels at each pixel, reduced along dim 1 directly (no permute
    copies; same math as model.ChannelNorm)."""

    def __init__(self, c, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        v = (x - u).pow(2).mean(1, keepdim=True)
        return (x - u) * torch.rsqrt(v + self.eps) * self.weight[:, None, None] + self.bias[:, None, None]


def make_norm(kind, c, groups=8):
    if kind == "ln":
        return ChannelLN(c)
    if kind == "ln_permute":
        return ChannelNorm(c)
    return nn.GroupNorm(min(groups, c), c)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, groups=8, norm="ln"):
        super().__init__()
        self.n1 = make_norm(norm, cin, groups)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n2 = make_norm(norm, cout, groups)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        return self.skip(x) + self.c2(F.silu(self.n2(h)))


class Attention(nn.Module):
    """Global self-attention (breaks locality: every output sees the whole input)."""

    def __init__(self, c, heads=4, groups=8):
        super().__init__()
        self.n = nn.GroupNorm(min(groups, c), c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.out = nn.Linear(c, c)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        t = self.n(x).flatten(2).transpose(1, 2)
        a = self.attn(t, t, t, need_weights=False)[0]
        return x + self.out(a).transpose(1, 2).reshape(B, C, H, W)


class SDFNet(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg = {**DEFAULT_SDFNET, **(cfg or {})}
        w, g, nk = cfg["widths"], cfg["groups"], cfg["norm"]
        self.multiple = 2 ** (len(w) - 1)
        self.stem = nn.Conv2d(cfg["in_ch"], w[0], 3, padding=1)
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for i, c in enumerate(w):
            self.enc.append(nn.Sequential(*[ResBlock(c, c, g, nk) for _ in range(cfg["enc_blocks"][i])]))
            if i + 1 < len(w):
                self.down.append(nn.Conv2d(c, w[i + 1], 3, stride=2, padding=1))
        self.attn = Attention(w[-1], groups=g) if cfg["attention"] else nn.Identity()
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(len(w) - 2, -1, -1):
            self.up.append(nn.Conv2d(w[i + 1], w[i], 1))
            self.dec.append(nn.Sequential(ResBlock(2 * w[i], w[i], g, nk),
                                          *[ResBlock(w[i], w[i], g, nk) for _ in range(cfg["dec_blocks"][i] - 1)]))
        self.head = nn.Sequential(make_norm(nk, w[0], g), nn.SiLU(), nn.Conv2d(w[0], w[0], 3, padding=1),
                                  nn.SiLU(), nn.Conv2d(w[0], 4, 1))
        self.head4 = nn.Conv2d(w[2], 1, 1) if len(w) > 2 else None
        self.T = float(cfg["trunc_px"])

    def forward(self, x):
        """x: (B, in_ch, H, W), H and W multiples of `multiple`. Returns dict of fp32 maps."""
        h = self.stem(x)
        skips = []
        for i, blk in enumerate(self.enc):
            h = blk(h)
            if i + 1 < len(self.enc):
                skips.append(h)
                h = self.down[i](h)
        h = self.attn(h)
        f4 = h if h.shape[-1] * 4 == x.shape[-1] else None     # 3-level nets: the bottleneck
        for k, (up, dec) in enumerate(zip(self.up, self.dec)):
            s = skips[-1 - k]
            h = F.interpolate(up(h), size=s.shape[-2:], mode="bilinear", align_corners=False)
            h = dec(torch.cat([h, s], 1))
            if s.shape[-1] * 4 == x.shape[-1]:
                f4 = h
        o = self.head(h).float()
        raw = o[:, 0]
        out = {"log_b": o[:, 1].clamp(-6, 4), "dir": o[:, 2:4]}
        if self.cfg["target"] == "occ":
            out["occ"] = raw
            out["sdf"] = raw              # level 0 = boundary, + inside (for contouring)
        else:
            out["sdf"] = self.T * torch.tanh(raw / self.T)
        if f4 is not None and self.head4 is not None:
            out["sdf4"] = (self.T / 4) * torch.tanh(self.head4(f4).float()[:, 0] / (self.T / 4))
        return out


def load_sdfnet(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ck["config"]["model"] if "config" in ck else ck["model_config"]
    m = SDFNet(cfg)
    m.load_state_dict(ck["model"] if "model" in ck else ck["state_dict"])
    return m
