"""Side-by-side table of coarse benchmark results (work/coarse/bench/*.json).

  python -m edgesr.compare_coarse classical_val coarse_ds_40m_val [...] [--slices illumination part_group]
"""

import argparse
import json
import os

from .bench_coarse import BENCH_DIR

MAIN = ("bscore", "recall@1.0", "recall@2.0", "precision@1.0", "iou", "part_miss",
        "false_parts_per_img", "inst_f1", "inst_pq", "inst_pq_strict", "merges_per_img", "predict_ms")
IMG = ("bscore", "recall@1.0", "part_miss", "false_parts_per_img", "inst_pq")
PART = ("recall@1.0", "gt_dist_mean", "part_miss", "inst_iou_mean")


def load(name):
    p = name if name.endswith(".json") else os.path.join(BENCH_DIR, name + ".json")
    return json.load(open(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--slices", nargs="*", default=["illumination", "blur", "layout", "part_group",
                                                     "part_material", "part_size"])
    a = ap.parse_args()
    rs = [load(n) for n in a.names]
    w = max(len(n) for n in a.names) + 2
    print(f"{'':{w}s}" + "".join(f"{k[:10]:>11s}" for k in MAIN))
    for n, r in zip(a.names, rs):
        print(f"{n:{w}s}" + "".join(f"{r.get(k, float('nan')):11.4f}" for k in MAIN))
    for s in a.slices:
        keys = PART if s.startswith("part_") else IMG
        print(f"\n[{s}]  " + " | ".join(keys) + "   (one column per result, in order)")
        for v in rs[0]["slices"][s]:
            cells = []
            for k in keys:
                cells.append(" ".join(f"{r['slices'][s].get(v, {}).get(k, float('nan')):.3f}" for r in rs))
            n = rs[0]["slices"][s][v].get("parts", rs[0]["slices"][s][v].get("images"))
            print(f"  {v:16s} n={n:<5d} " + " | ".join(cells))


if __name__ == "__main__":
    main()
