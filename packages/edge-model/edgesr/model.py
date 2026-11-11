"""Compact CNN: native-resolution residual trunk + progressive 2x upsampling."""

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_MODEL = {
    "width": 64,                     # trunk channels at 48x48
    "blocks": 8,                     # residual blocks at 48x48
    "dec_channels": [48, 32, 16, 8], # channels after each 2x stage (96, 192, 384, 768)
    "dec_blocks": 1,                 # residual blocks per decoder stage (int or per-stage list)
    "upsample": "pixelshuffle",      # "pixelshuffle" | "resize"
    "act": "silu",
    "input_skip": False,             # concat the bilinear-upsampled input at each stage
    "dilations": None,               # per trunk block dilation (None: all 1)
    "backlight_input": False,        # condition on a supplied backlight brightness value
    "context_px": 0,                 # extra input context per side around the 48 px target
    "context_margin": 3,             # of that, input px kept through the decoder (then cropped)
    "norm": None,                    # None | "group": pre-norm res blocks, norm after upsample
    "norm_dec_stages": 4,            # normalize only the first N (lowest-res) decoder stages
    "norm_trunk_out": None,          # None | "channel" | "group": norm on the trunk output
}


def _norm(kind, c):
    if kind is None:
        return nn.Identity()
    if kind == "group":
        return nn.GroupNorm(4 if c % 4 == 0 else 1, c)
    if kind == "channel":
        return ChannelNorm(c)
    raise ValueError(kind)


class ChannelNorm(nn.Module):
    """LayerNorm over channels at each pixel (no spatial statistics)."""

    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def _act(name):
    return {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}[name]()


class ResBlock(nn.Module):
    def __init__(self, c, act, dilation=1, norm=None):
        super().__init__()
        self.n = _norm(norm, c)
        self.c1 = nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.act = _act(act)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x):
        return x + self.c2(self.act(self.c1(self.n(x))))


def _icnr_(weight, scale=2):
    """ICNR init: every sub-pixel phase starts identical (no checkerboard)."""
    out_c, in_c, kh, kw = weight.shape
    sub = torch.empty(out_c // scale ** 2, in_c, kh, kw)
    nn.init.kaiming_normal_(sub, nonlinearity="relu")
    weight.data.copy_(sub.repeat_interleave(scale ** 2, dim=0))


class Up(nn.Module):
    def __init__(self, cin, cout, mode, extra_in, blocks, act, norm=None):
        super().__init__()
        self.n = _norm(norm, cout)
        self.mode = mode
        if mode == "pixelshuffle":
            self.conv = nn.Conv2d(cin, cout * 4, 3, padding=1)
            _icnr_(self.conv.weight)
        elif mode == "resize":
            self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        else:
            raise ValueError(mode)
        self.merge = nn.Conv2d(cout + extra_in, cout, 1) if extra_in else None
        self.act = _act(act)
        self.blocks = nn.Sequential(*[ResBlock(cout, act, norm=norm) for _ in range(blocks)])

    def forward(self, x, skip=None):
        if self.mode == "pixelshuffle":
            x = F.pixel_shuffle(self.conv(x), 2)
        else:
            x = self.conv(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        x = self.act(self.n(x))
        if self.merge is not None:
            x = self.act(self.merge(torch.cat([x, skip], 1)))
        return self.blocks(x)


class MaskSR(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        cfg = {**DEFAULT_MODEL, **(cfg or {})}
        self.cfg = cfg
        w = cfg["width"]
        self.stem = nn.Conv2d(2 if cfg["backlight_input"] else 1, w, 3, padding=1)
        dil = cfg["dilations"] or [1] * cfg["blocks"]
        assert len(dil) == cfg["blocks"]
        self.trunk = nn.Sequential(*[ResBlock(w, cfg["act"], d, cfg["norm"]) for d in dil])
        self.trunk_out = nn.Conv2d(w, w, 3, padding=1)
        self.trunk_norm = _norm(cfg["norm_trunk_out"], w)
        ups, cin = [], w
        in_ch = 2 if cfg["backlight_input"] else 1
        extra = in_ch if cfg["input_skip"] else 0
        db = cfg["dec_blocks"]
        db = db if isinstance(db, list) else [db] * len(cfg["dec_channels"])
        for k, (cout, nb) in enumerate(zip(cfg["dec_channels"], db)):
            norm = cfg["norm"] if k < cfg["norm_dec_stages"] else None
            ups.append(Up(cin, cout, cfg["upsample"], extra, nb, cfg["act"], norm))
            cin = cout
        self.ups = nn.ModuleList(ups)
        self.head = nn.Conv2d(cin, 1, 3, padding=1)

    def _prep(self, x, backlight):
        if self.cfg["backlight_input"]:
            b = backlight.to(x.dtype).view(-1, 1, 1, 1).clamp_min(1e-3)
            rel = (x / b - 0.5) * 2.0          # backlight -> 1, dark object -> -1
            lvl = ((b - 0.65) * 4.0).expand_as(x)
            return torch.cat([rel, lvl], 1)
        return (x - 0.5) * 4.0

    @torch.no_grad()
    def trunk_absmax(self, x, backlight=None):
        """Diagnostic: largest trunk activation (tracks residual-stream blow-up)."""
        f = self.stem(self._prep(x, backlight))
        return float(self.trunk_norm(f + self.trunk_out(self.trunk(f))).abs().max())

    @torch.no_grad()
    def stage_absmax(self, x, backlight=None):
        """Diagnostic: largest activation after the trunk, each decoder stage, and head."""
        stats = []
        self.forward(x, backlight, stats)
        return stats

    def forward(self, x, backlight=None, stats=None):
        # x: (B, 1, 48 + 2c, 48 + 2c) sensor reading in [0, 1] with c = context_px;
        # backlight: (B,) value, same units. Output: (B, 1, 768, 768) logits for the
        # central 48 px.
        x = self._prep(x, backlight)
        f = self.stem(x)
        f = self.trunk_norm(f + self.trunk_out(self.trunk(f)))
        if stats is not None:
            stats.append(float(f.abs().max()))
        c = self.cfg["context_px"]
        m = min(c, self.cfg["context_margin"])
        if c - m:
            k = c - m
            f = f[..., k:-k, k:-k]
            x = x[..., k:-k, k:-k]
        for i, up in enumerate(self.ups):
            skip = None
            if self.cfg["input_skip"]:
                skip = F.interpolate(x, scale_factor=2 ** (i + 1), mode="bilinear",
                                     align_corners=False)
            f = up(f, skip)
            if stats is not None:
                stats.append(float(f.abs().max()))
        out = self.head(f)
        if stats is not None:
            stats.append(float(out.abs().max()))
        if m:
            e = m * 2 ** len(self.ups)
            out = out[..., e:-e, e:-e]
        return out
