#!/usr/bin/env python3
"""Regenerate a compact, representative set of synthetic GUI fixtures."""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "synthetic-shapes" / "src"))

from inspection_shapes.dataset import sample  # noqa: E402
from inspection_shapes.taxonomy import PARTS  # noqa: E402


def main() -> None:
    output = ROOT / "app" / "demo-data"
    output.mkdir(parents=True, exist_ok=True)
    desired = {
        "drilled_plate": "drilled_plate",
        "stepped_shaft": "stepped_shaft",
        "grooved_pin": "grooved_pin",
        "slotted_bracket": "slotted_bracket",
    }
    config = {
        "seed": 20260907,
        "width": 1600,
        "height": 900,
        "supersample": 2,
        "views_per_design": 1,
        "contours_only": False,
    }
    manifest = []
    for index, family in enumerate(PARTS):
        if family not in desired:
            continue
        files, stats = sample((index, config))
        stem = desired[family]
        (output / f"{stem}.png").write_bytes(files["image.png"])
        # Preserve the generator's observed pixel contours for deterministic
        # model/UI testing; arbitrary user images still run through the edge model.
        with np.load(BytesIO(files["contours.npz"]), allow_pickle=False) as values:
            np.savez_compressed(
                output / f"{stem}.contours.npz",
                actual_xy_px=values["actual_xy_px"],
                ring_offsets=values["ring_offsets"],
            )
        metadata = json.loads(files["json"])
        manifest.append({
            "file": f"{stem}.png",
            "family": family,
            "sample_id": metadata["sample_id"],
            "feature_count": len(metadata["features"]),
            "severity": metadata["imperfections"]["severity"],
            "source": "synthetic-shapes",
        })
        print(f"wrote {stem}.png ({stats['family']})")
    missing = set(desired) - {entry["family"] for entry in manifest}
    if missing:
        raise RuntimeError(f"Unknown requested synthetic families: {sorted(missing)}")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
