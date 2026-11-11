"""Build a training index mixing corpora (no data copied: absolute file paths).

  python autoresearch/make_mix.py OUT CORPUS[:REPEAT] [CORPUS[:REPEAT] ...] [--val-from CORPUS ...]

Train entries of each corpus are repeated REPEAT times (oversampling). Validation
entries come from every listed corpus. Test/ood entries are never included.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from contour_model.data import remap_path


def build(out, specs):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); rows = []; config = None; sources = []
    for spec in specs:
        path, _, repeat = spec.partition(':'); path = Path(path); repeat = float(repeat or 1)
        c = json.loads((path/'config.json').read_text()); config = config or c
        if c.get('sampling') != config.get('sampling') or c['label_version'] != config['label_version']:
            raise ValueError(f'Incompatible corpus {path}')
        index = json.loads((path/'index.json').read_text())
        train = [e for e in index if e['split'] == 'train']; val = [e for e in index if e['split'] == 'val']
        k = int(repeat); extra = train[:int(round((repeat-k)*len(train)))]
        for e in train*k + extra + val:
            f = Path(e['file']); f = f if f.is_absolute() else path/f
            rows.append(dict(e, file=str(remap_path(f))))
        sources.append(dict(path=str(path), repeat=repeat, train=len(train), val=len(val)))
    config = dict(config, source='mix', mix=sources)
    (out/'config.json').write_text(json.dumps(config, indent=2)); (out/'index.json').write_text(json.dumps(rows))
    print(json.dumps(sources)); return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('out'); p.add_argument('specs', nargs='+'); a = p.parse_args()
    build(a.out, a.specs)
