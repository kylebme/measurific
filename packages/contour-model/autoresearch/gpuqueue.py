"""Durable sequential GPU queue for contour-model autoresearch.

queue.json (gitignored) holds entries:
  {"run_id", "argv": [python args...], "desc", "status": pending|running|done|failed,
   "fallback": bool, "after": run_id|null}
`run` executes pending entries one at a time (re-reading the file between runs, so
entries may be appended while it works); fallback entries run only when the main
queue is empty. Entries are expected to be resumable (ar_run.py resumes training
from last.pt). A lock prevents two runners; relaunching is always safe.

  python autoresearch/gpuqueue.py run --lane N # one per parallel GPU job, via setsid nohup
  python autoresearch/gpuqueue.py add RUN_ID --desc "..." [--fallback] [--after X] -- ARGS...
  python autoresearch/gpuqueue.py status
"""
import argparse, fcntl, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
QUEUE = os.path.join(HERE, 'queue.json'); LOCK = os.path.join(HERE, '.runner.lock')
EDIT = os.path.join(HERE, '.edit.lock'); LOG = os.path.join(HERE, 'runner.log')
LOGDIR = '/media/kyle/T7 Shield/inspection-contour-training/autoresearch/logs'


def load():
    return json.load(open(QUEUE)) if os.path.exists(QUEUE) else []


def edit(fn):
    with open(EDIT, 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX); q = load(); fn(q)
        with open(QUEUE+'.tmp', 'w') as f: json.dump(q, f, indent=1)
        os.replace(QUEUE+'.tmp', QUEUE)


def update(rid, **kw):
    edit(lambda q: [e.update(kw) for e in q if e['run_id'] == rid])


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"; print(line, flush=True)
    open(LOG, 'a').write(line+'\n')


def alive(pid):
    try: os.kill(int(pid), 0); return True
    except (OSError, TypeError, ValueError): return False


def claim():
    """Atomically pick the next runnable entry and mark it running with our pid."""
    chosen = []
    def pick(q):
        status = {e['run_id']: e.get('status', 'pending') for e in q}
        ready = [e for e in q if (e.get('status', 'pending') == 'pending' or
                                  (e.get('status') == 'running' and not alive(e.get('pid'))))
                 and (not e.get('after') or status.get(e['after']) == 'done')]
        main = [e for e in ready if not e.get('fallback')]
        if main or ready:
            e = (main or ready)[0]; e.update(status='running', pid=os.getpid(), started=time.strftime('%Y-%m-%d %H:%M:%S'))
            chosen.append((dict(e), not main))
    edit(pick)
    return chosen[0] if chosen else (None, False)


def run(lane=0):
    lock = open(f'{LOCK}.{lane}', 'w')
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError: print('another runner holds this lane', file=sys.stderr); return 1
    log(f'runner lane={lane} start pid={os.getpid()}'); os.makedirs(LOGDIR, exist_ok=True)
    idle = 0
    while True:
        e, fallback = claim()
        if e is None:
            # Other lanes may still be running prerequisites of 'after' entries.
            if idle >= 20: log(f'lane {lane}: queue empty; runner exit'); return 0
            idle += 1; time.sleep(30); continue
        idle = 0; rid = e['run_id']
        if fallback: log(f'lane {lane}: main queue empty; running fallback {rid}')
        log(f"lane {lane} start {rid}: {' '.join(e['argv'])}")
        with open(os.path.join(LOGDIR, rid+'.log'), 'a') as out:
            rc = subprocess.call([sys.executable, '-u', *e['argv']], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        update(rid, status='done' if rc == 0 else 'failed', returncode=rc, finished=time.strftime('%Y-%m-%d %H:%M:%S'))
        log(f"lane {lane} {'done' if rc == 0 else 'FAILED'} {rid} rc={rc}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    rr = sub.add_parser('run'); rr.add_argument('--lane', type=int, default=0); sub.add_parser('status')
    a = sub.add_parser('add'); a.add_argument('run_id'); a.add_argument('--desc', required=True)
    a.add_argument('--fallback', action='store_true'); a.add_argument('--after'); a.add_argument('--front', action='store_true')
    r = sub.add_parser('retry'); r.add_argument('run_id')
    argv = sys.argv[1:]; rest = []
    if '--' in argv: i = argv.index('--'); argv, rest = argv[:i], argv[i+1:]
    x = ap.parse_args(argv)
    if x.cmd == 'run': sys.exit(run(x.lane))
    if x.cmd == 'add':
        argv = rest
        def add(q):
            assert all(e['run_id'] != x.run_id for e in q), 'duplicate run_id'
            entry = dict(run_id=x.run_id, argv=argv, desc=x.desc, status='pending', fallback=x.fallback, after=x.after)
            if x.front:
                i = next((i for i, e in enumerate(q) if e.get('status', 'pending') == 'pending'), len(q)); q.insert(i, entry)
            else: q.append(entry)
        edit(add)
    if x.cmd == 'retry': update(x.run_id, status='pending')
    for e in load():
        print(f"{e.get('status','pending'):8s} {e['run_id']:28s} {e['desc'][:70]}{' [fallback]' if e.get('fallback') else ''}")


if __name__ == '__main__': main()
