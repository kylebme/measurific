import argparse
import json
from pathlib import Path
import sys
import time
from .dataset import generate, load_job
from .geometry import scene_from_list
from .validation import convergence
from .benchmark import generate_benchmark, evaluate_benchmark
from .studies import run_suite


def main():
    parser = argparse.ArgumentParser(description='Synthetic telecentric silhouette data (linear monochrome)')
    commands = parser.add_subparsers(dest='command', required=True)
    gen = commands.add_parser('generate', help='Generate seeded NPZ/JSON/PNG samples')
    gen.add_argument('job', type=Path)
    gen.add_argument('--output', type=Path, required=True)
    gen.add_argument('--count', type=int)
    check = commands.add_parser('converge', help='Measure spatial, angular and PSF-support convergence on a ROI')
    check.add_argument('job', type=Path)
    check.add_argument('--output', type=Path, required=True)
    check.add_argument('--levels', type=int, nargs='+', default=[8, 16, 32, 64])
    check.add_argument('--angular-levels', type=int, nargs='+')
    check.add_argument('--support-levels', type=float, nargs='+')
    check.add_argument('--tolerance', type=float, default=0.01)
    benchmark = commands.add_parser('benchmark', help='Generate artifact-grouped calibration and held-out samples')
    benchmark.add_argument('spec', type=Path)
    benchmark.add_argument('--output', type=Path, required=True)
    evaluate = commands.add_parser('evaluate', help='Fit calibration-only baseline correction and score held-out edges')
    evaluate.add_argument('dataset', type=Path)
    evaluate.add_argument('--output', type=Path, required=True)
    evaluate.add_argument('--channel', default='irradiance')
    suite = commands.add_parser('validation-suite', help='Run geometry/depth/phase/source and mesh convergence studies')
    suite.add_argument('spec', type=Path)
    suite.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    last = [0.]

    def progress(done, total):
        now = time.monotonic()
        if now-last[0] > 5 or done == total:
            print(f'Rendering tiles: {done}/{total}', file=sys.stderr)
            last[0] = now

    try:
        if args.command == 'generate':
            result = generate(args.job, args.output, args.count, progress)
            print(f"Wrote {len(result['samples'])} samples to {args.output.resolve()}")
        elif args.command == 'benchmark':
            result = generate_benchmark(args.spec, args.output, progress)
            print(f"Wrote {len(result['samples'])} benchmark samples to {args.output.resolve()}")
        elif args.command == 'evaluate':
            result = evaluate_benchmark(args.dataset, args.output, args.channel)
            print(json.dumps({key: result[key] for key in ('channel', 'calibration_rank', 'splits')}, indent=2))
        elif args.command == 'validation-suite':
            result = run_suite(args.spec, args.output, progress)
            print(json.dumps({'cases': len(result['cases']), 'all_within_tolerance': result['all_within_tolerance']}, indent=2))
            if not result['all_within_tolerance']:
                return 2
        else:
            if args.output.exists():
                raise ValueError('Convergence output already exists')
            job, config, base = load_job(args.job)
            scene = scene_from_list(job['shapes'], base)
            report = convergence(scene, config, job.get('roi'), args.levels, args.angular_levels, args.support_levels, args.tolerance, progress)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            print(json.dumps(report, indent=2))
            if not report['last_comparisons_within_tolerance']:
                return 2
    except (ValueError, OSError, TypeError, KeyError) as error:
        parser.exit(1, f'error: {error}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
