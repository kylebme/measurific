"""One autoresearch run: (resumable) training, fixed benchmark, one ledger line.

  python autoresearch/ar_run.py RUN_ID --desc "..." --cache CACHE [--init CKPT] [--post JSON] -- TRAIN_ARGS...

Outputs go to T7: autoresearch/runs/RUN_ID/{best.pt,last.pt,history.jsonl,bench.json}.
With --checkpoint (no training) only the benchmark runs, e.g. for post-processing changes.
"""
import argparse, json, os, subprocess, sys, time, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
RUNS = Path('/media/kyle/T7 Shield/inspection-contour-training/autoresearch/runs')
LEDGER = HERE/'ledger.jsonl'


def git_commit():
    try: return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception: return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('run_id'); ap.add_argument('--desc', required=True)
    ap.add_argument('--cache'); ap.add_argument('--init'); ap.add_argument('--checkpoint')
    ap.add_argument('--wait-for', help='sleep until this path exists (e.g. a corpus audit.json)')
    ap.add_argument('--mix', nargs='+', help='CORPUS[:REPEAT] specs; builds runs/RUN_ID/mix cache and trains on it')
    ap.add_argument('--post', default='{}'); ap.add_argument('--sets', default='edge,edge_val,orig')
    argv = sys.argv[1:]; train = []
    if '--' in argv: i = argv.index('--'); argv, train = argv[:i], argv[i+1:]
    a = ap.parse_args(argv)
    out = RUNS/a.run_id; out.mkdir(parents=True, exist_ok=True); py = sys.executable
    env = dict(os.environ, PYTHONPATH=f"{ROOT/'src'}:{ROOT.parents[1]/'research'/'synthetic-shapes'/'src'}")
    while a.wait_for and not Path(a.wait_for).exists():
        print(f'waiting for {a.wait_for}', flush=True); time.sleep(120)
    if a.mix:
        sys.path.insert(0, str(HERE)); from make_mix import build as build_mix
        if not (out/'mix'/'index.json').exists(): build_mix(out/'mix', a.mix)
        a.cache = str(out/'mix')
    t0 = time.time(); history = []
    if a.checkpoint: ckpt = Path(a.checkpoint)
    else:
        done = (out/'history.jsonl').exists() and (out/'last.pt').exists() and (out/'train.done').exists()
        if not done:
            if (out/'config.json').exists() and not (out/'last.pt').exists():
                for stale in ('config.json', 'history.jsonl'): (out/stale).unlink(missing_ok=True)
            cmd = [py, '-u', '-m', 'contour_model.train_primitives', a.cache, str(out), '--device', 'cuda', '--resume', *train]
            if a.init: cmd += ['--init', a.init]
            subprocess.run(cmd, cwd=ROOT, env=env, check=True); (out/'train.done').write_text('ok')
        ckpt = out/'best.pt'
        history = [json.loads(l) for l in open(out/'history.jsonl')]
    bench = out/'bench.json'
    subprocess.run([py, '-u', str(ROOT/'scripts'/'bench.py'), str(ckpt), str(bench), '--sets', a.sets, '--kwargs', a.post],
                   cwd=ROOT, env=env, check=True)
    # (autoresearch/queue.py was renamed gpuqueue.py: it shadowed the stdlib 'queue'.)
    sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != HERE]
    sys.path.insert(0, str(ROOT/'scripts')); from bench import headline
    report = json.loads(bench.read_text())
    settings = json.loads((out/'config.json').read_text()) if (out/'config.json').exists() else {}
    best = max(history, key=lambda h: h['selection_score']) if history else {}
    line = dict(run_id=a.run_id, time=time.strftime('%Y-%m-%d %H:%M:%S'), commit=git_commit(), desc=a.desc,
                checkpoint=str(ckpt), post=json.loads(a.post), init=a.init or settings.get('init'), cache=a.cache,
                train_args=train, steps=settings.get('steps'), best_step=best.get('step'),
                val_selection=best.get('selection_score'),
                peak_vram_mb=max((h.get('peak_cuda_mb', 0) for h in history), default=0),
                train_seconds=history[-1]['elapsed_seconds'] if history else 0, wall_seconds=time.time()-t0,
                metrics={k: headline(v) for k, v in report['results'].items()},
                latency_p95={k: round(v['latency_p95'], 3) for k, v in report['results'].items()})
    with open(LEDGER, 'a') as f: f.write(json.dumps(line)+'\n')
    print(json.dumps(line['metrics']), flush=True)


if __name__ == '__main__': main()
