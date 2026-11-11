"""Contact sheets for v3 corpora: clean labels, or render + extracted labels.

Clean corpus tiles draw the observed contour colored by base class with instance
endpoints. Extraction tiles show the rendered image with the extracted contour
overlaid; zero-weight (ignored) edges are drawn red, compounds green.

  python -m contour_model.review_v3 CORPUS_OR_EXTRACT OUTPUT [--per-page 20] [--limit N]
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .render_v3 import to_mm

COLORS = {0: (120, 120, 120), 1: (22, 138, 173), 2: (239, 131, 84), 3: (140, 94, 192)}
COMPOUND = (54, 166, 87)
IGNORED = (220, 38, 38)
TILE = 420


def _font(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


def _paths(root, entry):
    file = Path(entry['file'])
    if file.parent == Path('.'):
        folder = root/'sources'
    else:
        folder = root/file.parent/'sources'
    return folder/(entry['id']+'.json'), folder/(entry['id']+'.contours.npz')


def _draw_contours(draw, xy, a, transform, width=2):
    offsets = a['ring_offsets']; cls = a['edge_base_class']; comp = a['edge_compound_class']
    inst = a['edge_base_instance']; weight = a.get('edge_label_weight', np.ones(len(xy)))
    p = transform(xy)
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        for i in range(lo, hi):
            j = i+1 if i+1 < hi else lo
            color = IGNORED if weight[i] == 0 else COMPOUND if comp[i] else COLORS[int(cls[i])]
            draw.line([tuple(p[i]), tuple(p[j])], fill=color, width=width)
        ids = inst[lo:hi]
        for k in np.flatnonzero(ids != np.roll(ids, 1)):
            x, y = p[lo+k]
            draw.ellipse((x-2.5, y-2.5, x+2.5, y+2.5), outline=(17, 24, 39), fill=(255, 255, 255))


def tile(root, entry, extract):
    meta_path, contour_path = _paths(root, entry)
    meta = json.loads(meta_path.read_text())
    with np.load(contour_path) as z:
        a = {k: z[k] for k in z.files}
    im = Image.new('RGB', (TILE, TILE+44), 'white'); draw = ImageDraw.Draw(im)
    if extract:
        settings = meta['render']
        png = root/'sources'/f"{entry['parent_id']}.render.png"
        if png.exists():
            src = np.asarray(Image.open(png)).astype(np.float32)
            src = (255*src/settings['max_value']).clip(0, 255).astype(np.uint8)
        else:
            src = np.full((settings['height'], settings['width']), 200, np.uint8)
        m = np.asarray(settings['to_mm']['matrix']); o = np.asarray(settings['to_mm']['offset'])
        px = a['actual_xy_mm']@np.linalg.inv(m).T+o
        scale = TILE/max(src.shape)
        base = Image.fromarray(src).convert('RGB').resize((max(1, int(src.shape[1]*scale)), max(1, int(src.shape[0]*scale))))
        im.paste(base, (0, 44))
        _draw_contours(draw, px, a, lambda q: q*scale+[0, 44], width=1)
        info = (f"{entry['extractor']} · {settings['extent_px']:.0f}px · σ{settings['blur_sigma']:.1f} · "
                f"{settings['bits']}b · dirt {len(settings['contamination'])} · ign {entry.get('ignored_fraction', 0):.0%}")
    else:
        xy = a['actual_xy_mm']; lo, hi = xy.min(0), xy.max(0); s = (TILE-20)/max(hi-lo)
        _draw_contours(draw, xy, a, lambda q: np.c_[10+(q[:, 0]-lo[0])*s, 54+(hi[1]-q[:, 1])*s], width=2)
        style = meta.get('corner_style', {})
        info = (f"{entry['split']} · {len(a['ring_offsets'])-1} rings · {entry.get('instances', 0)} inst · "
                f"corners {style.get('applied', 0)} · ign {entry.get('ignored_fraction', 0):.0%}")
    draw.text((6, 4), f"{entry['id']} {entry['family']}" + (f" ({meta.get('csg_style')})" if meta.get('csg_style') else ''),
              fill=(15, 23, 42), font=_font(14))
    draw.text((6, 24), info, fill=(71, 85, 105), font=_font(12))
    return im


def build(source, output, per_page=20, limit=None, cols=5):
    source = Path(source); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    entries = json.loads((source/'index.json').read_text())
    entries = sorted(entries, key=lambda e: (e.get('parent_id', e['id']), e.get('extractor', '')))
    if limit:
        entries = entries[:limit]
    extract = any('extractor' in e for e in entries)
    pages = []
    for start in range(0, len(entries), per_page):
        group = entries[start:start+per_page]
        rows = (len(group)+cols-1)//cols
        sheet = Image.new('RGB', (cols*TILE, rows*(TILE+44)+40), (241, 245, 249))
        d = ImageDraw.Draw(sheet)
        legend = 'blue line · orange arc · purple circle · green compound · red ignored (zero weight) · dots = instance endpoints'
        d.text((8, 10), legend, fill=(30, 41, 59), font=_font(16))
        for k, e in enumerate(group):
            sheet.paste(tile(source, e, extract), ((k % cols)*TILE, 40+(k//cols)*(TILE+44)))
        name = f'sheet-{start//per_page+1:03d}.png'; sheet.save(output/name); pages.append(name)
    html = ['<!doctype html><meta charset="utf-8"><title>Contour v3 review</title>',
            '<body style="font:15px system-ui;background:#e2e8f0;margin:16px">',
            f'<h2>{source.name}: {len(entries)} samples</h2>']
    html += [f'<p>{p}</p><img src="{p}" style="max-width:100%;background:white">' for p in pages]
    (output/'index.html').write_text('\n'.join(html))
    return pages


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('source'); p.add_argument('output')
    p.add_argument('--per-page', type=int, default=20); p.add_argument('--limit', type=int)
    build(**vars(p.parse_args()))
