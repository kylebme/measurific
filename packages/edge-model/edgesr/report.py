"""Print the experiment ledger as a compact table.

Usage: python -m edgesr.report [--last N] [--sort] [--cat narrow]
"""

import argparse
import json
import os

from . import REPO_DIR

LEDGER = os.path.join(REPO_DIR, "ledger.jsonl")


def rows():
    if not os.path.exists(LEDGER):
        return []
    with open(LEDGER) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=0)
    ap.add_argument("--sort", action="store_true", help="sort by bscore")
    ap.add_argument("--cat", help="show this category's metrics instead of overall")
    a = ap.parse_args()
    rs = rows()
    if a.sort:
        rs.sort(key=lambda r: r["primary"]["bscore"] or 9)
    if a.last:
        rs = rs[-a.last:]
    hdr = (f"{'run_id':30s} {'bscore':>7s} {'miss%':>6s} {'extra%':>6s} {'loc':>6s} "
           f"{'loc95':>6s} {'comp%':>6s} {'iou':>7s} {'Mpar':>5s} {'min':>4s} {'kstep':>6s} "
           f"{'sps':>5s} {'MB':>5s}  description")
    print(hdr)
    for r in rs:
        m = r["metrics"]
        if a.cat:
            m = {**m, **{k: v for k, v in m["by_category"][a.cat].items() if v is not None}}

        def g(k, scale=1.0, fmt="6.3f"):
            v = m.get(k)
            return format(v * scale, fmt) if v is not None else "  -   "
        print(f"{r['run_id'][:30]:30s} {g('bscore', 1, '7.4f')} {g('missed_rate', 100, '6.2f')} "
              f"{g('extra_rate', 100, '6.2f')} {g('loc_error_mean', 1, '6.4f')} "
              f"{g('loc_error_p95', 1, '6.3f')} {g('component_miss_rate', 100, '6.2f')} "
              f"{g('iou', 1, '7.4f')} {r['params'] / 1e6:5.2f} "
              f"{r['budget']['train_seconds'] / 60:4.0f} {r['budget']['steps'] / 1e3:6.1f} "
              f"{r['system']['train_samples_per_s']:5.0f} {r['system']['peak_vram_mb']:5d}  "
              f"{r['description'][:60]}")


if __name__ == "__main__":
    main()
