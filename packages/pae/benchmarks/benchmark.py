"""Steady-state PAE benchmark; image conversion and all output are included."""

from __future__ import annotations

import argparse
import gc
import json
import platform
from statistics import median
from time import perf_counter

import numpy as np

from pae import SilhouetteWorkspace, detect, detect_silhouette, warmup


def make_image(height: int, width: int) -> np.ndarray:
    image = np.full((height, width), 220, dtype=np.uint8)
    margin_y = max(8, height // 4)
    margin_x = max(8, width // 4)
    image[margin_y:-margin_y, margin_x:-margin_x] = 20
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=3000)
    parser.add_argument("--width", type=int, default=4000)
    parser.add_argument("--threshold", type=float, default=25.0)
    parser.add_argument("--smoothing", type=int, default=0)
    parser.add_argument("--seeded", action="store_true")
    parser.add_argument("--silhouette-threshold", type=float, default=120.0)
    parser.add_argument("--search-radius", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    image = make_image(args.height, args.width)
    warmup(
        include_iterative=args.smoothing > 1,
        include_silhouette=args.seeded,
    )
    workspace = SilhouetteWorkspace(image.shape) if args.seeded else None

    def run_detector():
        if args.seeded:
            return detect_silhouette(
                image,
                args.threshold,
                silhouette_threshold=args.silhouette_threshold,
                search_radius=args.search_radius,
                workspace=workspace,
            )
        return detect(image, args.threshold, smoothing=args.smoothing)

    # Compile the exact dtype/mode path and fault in the benchmark image.
    run_detector()

    durations = []
    edge_count = 0
    gc.disable()
    try:
        for _ in range(args.repeats):
            start = perf_counter()
            edges = run_detector()
            durations.append(perf_counter() - start)
            edge_count = len(edges)
    finally:
        gc.enable()

    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "shape": [args.height, args.width],
        "megapixels": args.height * args.width / 1e6,
        "dtype": str(image.dtype),
        "smoothing": args.smoothing,
        "seeded": args.seeded,
        "search_radius": args.search_radius if args.seeded else None,
        "edge_count": edge_count,
        "repeats": args.repeats,
        "milliseconds": [round(value * 1e3, 3) for value in durations],
        "median_ms": round(median(durations) * 1e3, 3),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
