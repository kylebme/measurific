"""Render measured support and dimension witnesses with a separate report legend."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def annotate(image, report, *, level=4):
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] not in (3, 4)):
        raise ValueError('Annotation image must be uint8 grayscale, RGB or RGBA')
    source = Image.fromarray(array).convert('RGB')
    visible = {f['id'] for f in report['features'] if f.get('level', 1) <= level}
    rows = [m for m in report['measurements'] if any(ref in visible for ref in m['feature_refs'])]
    row_height = lambda m: 210 if m.get('thread_metrics') else 100 if m.get('field_bands') else 42
    canvas = Image.new('RGB', (source.width+580, max(source.height, 72+sum(row_height(m) for m in rows))), '#111820')
    canvas.paste(source, (0, 0))
    draw = ImageDraw.Draw(canvas, 'RGBA')
    font = ImageFont.load_default(size=14)
    colors = ['#00c2ff', '#ffb347', '#d78aff', '#30df9b', '#ff7999']
    label_boxes = []
    for i, feature in enumerate(report['features']):
        if feature['id'] not in visible or feature['status'] != 'fitted':
            continue
        color = colors[i % len(colors)]
        for q, closed in zip(feature.get('support', []), feature.get('closed', [])):
            vertices = [tuple(p) for p in q]
            if closed:
                vertices.append(vertices[0])
            draw.line(vertices, fill=color, width=1)
        geom = feature['geometry']
        if 'center' in geom:
            x, y = geom['center']
            draw.line([(x-4, y), (x+4, y)], fill=color, width=1)
            draw.line([(x, y-4), (x, y+4)], fill=color, width=1)
            label_x, label_y = x+5, y+4
            for _ in range(len(label_boxes)+1):
                box = draw.textbbox((label_x, label_y), feature['id'], font=font)
                if not any(box[0] < b[2]+3 and box[2]+3 > b[0] and box[1] < b[3]+3 and box[3]+3 > b[1] for b in label_boxes):
                    break
                label_y += 18
            label_boxes.append(box)
            draw.text((label_x, label_y), feature['id'], font=font, fill=color, stroke_width=1, stroke_fill='#111820')
    x = source.width+14
    draw.text((x, 14), 'Contour measurements | pixels', fill='white', font=font)
    draw.text((x, 35), 'Image coordinates: x right, y down', fill='#a8b9c9', font=font)
    row_y = 66
    for i, m in enumerate(rows):
        color = '#ff6575' if m['status'] != 'measured' or m.get('pass_fail') == 'fail' else '#70e3ac'
        # Virtual/fitted supports are visually distinct from observed edges.
        virtual = list(m.get('envelope_runs', []))
        if m.get('definition', {}).get('basis') == 'fitted':
            virtual.extend([np.asarray(band)[:, side] for band in m.get('field_bands', []) for side in (0, 1)])
        for q in virtual:
            for a, b in zip(np.asarray(q)[:-1], np.asarray(q)[1:]):
                length = np.linalg.norm(b-a)
                for start in np.arange(0, length, 8):
                    draw.line([tuple(a+(b-a)*start/length), tuple(a+(b-a)*min(start+4, length)/length)], fill='#e1b866', width=1)
        if m.get('selected_statistic') == 'average':
            for band in m.get('field_bands', []):
                points = np.asarray(band)
                draw.polygon([tuple(p) for p in np.concatenate([points[:, 0], points[::-1, 1]])], fill=(92, 217, 222, 40))
        if m.get('field_bands'):
            for name in ('minimum_witness', 'maximum_witness'):
                draw.line([tuple(p) for p in m[name]], fill=color, width=2)
        if 'witness' in m:
            draw.line([tuple(p) for p in m['witness']], fill=color, width=2)
            for px, py in m['witness']:
                draw.ellipse((px-2, py-2, px+2, py+2), fill=color)
        value = m['value']
        text = f'{value:.6g} {m["units"]}' if isinstance(value, (float, int)) else str(value)
        if m['status'] != 'measured':
            text = m.get('reason', m['status'])
        elif m.get('pass_fail'):
            text += '  '+m['pass_fail'].upper()
        y = row_y
        row_y += row_height(m)
        if m.get('field_bands'):
            draw.rectangle((x-5, y-5, x+550, y+88), outline=color, width=1)
            draw.text((x, y), f'{m["id"]}: gap between surfaces', fill='white', font=font)
            draw.text((x, y+22), 'min / max / average: '+' / '.join(f'{m[k]:.3f}' for k in ('minimum', 'maximum', 'average'))+' px', fill=color, font=font)
            draw.text((x, y+44), f'Measured area: {m["coverage"]*100:.1f}% | '+m['definition']['basis'], fill=color, font=font)
            draw.text((x, y+65), 'Average uses the highlighted area; dashed lines are fitted/virtual.', fill=color, font=font)
            anchor = tuple(m['field_bands'][0][0][0])
            draw.line([(x-5, y+44), anchor], fill=color, width=1)
            continue
        if m.get('thread_metrics'):
            draw.rectangle((x-5, y-5, x+550, y+195), outline=color, width=1)
            draw.text((x, y), 'Thread measurements | min / max / avg', fill='white', font=font)
            for j, (name, metric) in enumerate(m['thread_metrics'].items()):
                values = ' / '.join('—' if metric[k] is None else f'{metric[k]:.3f}'
                                    for k in ('minimum', 'maximum', 'average'))
                draw.text((x, y+25+j*24), f'{name.replace("_", " ")}: {values} {metric["units"]}', fill=color, font=font)
                for statistic in ('minimum', 'maximum'):
                    witness = metric.get(statistic+'_witness')
                    if not witness:
                        continue
                    points = [tuple(p) for p in witness]
                    draw.line(points, fill=color, width=2)
                    for px, py in points:
                        draw.ellipse((px-2, py-2, px+2, py+2), fill=color)
                    label = f'{name.replace("_", " ")} {statistic}: {metric[statistic]:.3f}'
                    lx, ly = points[-1]
                    lx = min(max(0, lx+5), max(0, source.width-draw.textlength(label, font=font)))
                    ly = max(0, min(ly, source.height-18))
                    for _ in range(len(label_boxes)+1):
                        box = draw.textbbox((lx, ly), label, font=font)
                        if not any(box[0] < b[2]+3 and box[2]+3 > b[0] and box[1] < b[3]+3 and box[3]+3 > b[1] for b in label_boxes):
                            break
                        ly += 18
                    label_boxes.append(box)
                    draw.line([points[-1], (lx, ly)], fill=color)
                    draw.text((lx, ly), label, font=font, fill=color, stroke_width=1, stroke_fill='#111820')
            feature = next((f for f in report['features'] if f['id'] in m['feature_refs'] and f.get('support')), None)
            if feature:
                draw.line([(x-5, y+95), tuple(feature['support'][0][0])], fill=color)
            continue
        draw.text((x, y), m['id'][:70], fill='white', font=font)
        draw.text((x, y+18), text[:70], fill=color, font=font)
    return canvas
