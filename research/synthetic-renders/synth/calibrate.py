"""Benchmark Cycles sample convergence on realized scenes without rerendering masks."""

import argparse
import json
import os
import statistics
import subprocess
import tempfile

from PIL import Image

from .cli import PROJECT_ROOT, WORKER, _write_json
from .sampling import sample_scene


def calibrate(indices, samples, size, seed, blender, threads):
    report = {"size": size, "indices": indices, "samples": samples, "scenes": []}
    with tempfile.TemporaryDirectory(prefix="synth-calibration-") as directory:
        for index in indices:
            scene_report = {"index": index, "family": None, "variants": {}}
            images = {}
            for count in samples:
                manifest = sample_scene(seed, index, size, size)
                manifest["render"]["cycles_samples"] = count
                scene_report["family"] = manifest["geometry"]["family"]
                target = os.path.join(directory, f"scene{index}_samples{count}")
                os.makedirs(target)
                path = os.path.join(target, "manifest.json")
                _write_json(path, manifest)
                cmd = [blender, "-b", "-noaudio", "--factory-startup", "-t", str(threads),
                       "--python", WORKER, "--", path, target, "--source-only"]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
                with Image.open(os.path.join(target, "image.png")) as image:
                    images[count] = list(image.getdata())
                with open(os.path.join(target, "metadata.json")) as stream:
                    result = json.load(stream)["render_result"]
                scene_report["variants"][str(count)] = {
                    "render_seconds": result["duration_seconds"],
                    "source_bytes": os.path.getsize(os.path.join(target, "image.png"))}
            reference = images[samples[-1]]
            for count in samples[:-1]:
                diffs = sorted(abs(a - b) for a, b in zip(images[count], reference))
                scene_report["variants"][str(count)].update({
                    "mean_absolute_dn_vs_reference": round(statistics.mean(diffs), 2),
                    "p99_absolute_dn_vs_reference": diffs[int(0.99 * (len(diffs) - 1))],
                    "mean_absolute_fraction_vs_reference": round(statistics.mean(diffs) / 65535, 6)})
            report["scenes"].append(scene_report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", type=int, nargs="+", default=[6, 14])
    parser.add_argument("--samples", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--blender", default="/snap/blender/current/blender")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if sorted(set(args.samples)) != args.samples or len(args.samples) < 2:
        parser.error("--samples must be strictly increasing with at least two counts")
    print(json.dumps(calibrate(args.indices, args.samples, args.size, args.seed, args.blender, args.threads), indent=2))
