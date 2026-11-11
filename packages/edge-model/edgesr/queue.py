"""Durable sequential experiment queue.

queue.json (repository dir, gitignored) is a list of entries:
  {"run_id": ..., "config": "configs/x.json" | null, "set": ["train.lr=1e-3", ...],
   "desc": "hypothesis", "status": "pending" | "running" | "done" | "failed"}

`python -m edgesr.queue run` executes pending entries one at a time in
subprocesses, re-reading the file between runs so entries can be appended while
it works. An interrupted entry is resumed from its last checkpoint. Relaunching
the runner is always safe; a PID lock prevents two runners.

`python -m edgesr.queue add RUN_ID [--config C] [--set k=v ...] --desc "..."`
`python -m edgesr.queue status`
"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time

from . import REPO_DIR, RUNS_DIR, WORK_DIR

QUEUE = os.path.join(REPO_DIR, "queue.json")
LOCK = os.path.join(WORK_DIR, "queue_runner.lock")
LOG = os.path.join(WORK_DIR, "queue_runner.log")


EDIT_LOCK = os.path.join(WORK_DIR, "queue_edit.lock")


def load():
    if not os.path.exists(QUEUE):
        return []
    with open(QUEUE) as f:
        return json.load(f)


def save(q):
    with open(QUEUE + ".tmp", "w") as f:
        json.dump(q, f, indent=1)
    os.replace(QUEUE + ".tmp", QUEUE)


def update(run_id, **kw):
    """Locked read-modify-write (several runner lanes share queue.json)."""
    with open(EDIT_LOCK, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        q = load()
        for e in q:
            if e["run_id"] == run_id:
                e.update(kw)
        save(q)


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def run(lane="all"):
    """lane: "train" runs training entries, "eval" runs cmd entries, "all" runs both."""
    lock = open(LOCK if lane in ("all", "train") else LOCK + ".eval", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another runner holds the lock", file=sys.stderr)
        return 1
    log(f"runner start pid={os.getpid()}")
    while True:
        q = load()
        if lane == "train":
            q = [e for e in q if not e.get("cmd")]
        elif lane == "eval":
            q = [e for e in q if e.get("cmd")]
        status = {e["run_id"]: e.get("status", "pending") for e in load()}
        pending = [e for e in q if e.get("status", "pending") in ("pending", "running")
                   and not e.get("fallback")
                   and (not e.get("after") or status.get(e["after"]) == "done")]
        if not pending:
            # Never idle the GPU: fall back to low-priority entries (long runs, seeds).
            pending = [e for e in q if e.get("status", "pending") in ("pending", "running")]
            pending = [e for e in pending
                       if not e.get("after") or status.get(e["after"]) == "done"]
            if pending:
                log(f"main queue empty; running fallback {pending[0]['run_id']}")
        if not pending:
            if lane == "eval":        # the eval lane idles, waiting for new scoring jobs
                time.sleep(60)
                continue
            log("queue empty; runner exit")
            return 0
        e = pending[0]
        rid = e["run_id"]
        run_dir = os.path.join(RUNS_DIR, rid)
        # A previous runner may have left this entry's process running: wait for it.
        while subprocess.run(["pgrep", "-f", f"run-id {rid}( |$)"],
                             capture_output=True).returncode == 0:
            time.sleep(30)
        if e.get("cmd"):
            update(rid, status="running", started=time.strftime("%Y-%m-%d %H:%M:%S"))
            log(f"start cmd {rid}: {' '.join(e['cmd'])}")
            os.makedirs(os.path.join(WORK_DIR, "cmd_logs"), exist_ok=True)
            with open(os.path.join(WORK_DIR, "cmd_logs", rid + ".log"), "a") as out:
                rc = subprocess.call([sys.executable, *e["cmd"], "--run-id-tag", rid]
                                     if e.get("tag_arg") else [sys.executable, *e["cmd"]],
                                     cwd=REPO_DIR, stdout=out, stderr=subprocess.STDOUT)
            update(rid, status="done" if rc == 0 else "failed", returncode=rc,
                   finished=time.strftime("%Y-%m-%d %H:%M:%S"))
            log(f"{'done' if rc == 0 else 'FAILED'} cmd {rid} rc={rc}")
            continue
        if os.path.exists(os.path.join(run_dir, "metrics.json")):
            update(rid, status="done")
            continue
        cmd = [sys.executable, "-m", e.get("module", "edgesr.train")]
        if e.get("config"):
            cmd.append(e["config"])
        cmd += ["--run-id", rid, "--desc", e.get("desc", "")]
        if os.path.exists(os.path.join(run_dir, "last.pt")):
            cmd.append("--resume")
        elif e.get("set"):
            cmd += ["--set", *e["set"]]
        if e.get("no_ledger"):
            cmd.append("--no-ledger")
        update(rid, status="running", started=time.strftime("%Y-%m-%d %H:%M:%S"))
        log(f"start {rid}: {' '.join(cmd[2:])}")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "stdout.log"), "a") as out:
            rc = subprocess.call(cmd, cwd=REPO_DIR, stdout=out, stderr=subprocess.STDOUT)
        ok = rc == 0 and os.path.exists(os.path.join(run_dir, "metrics.json"))
        update(rid, status="done" if ok else "failed", returncode=rc,
               finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        log(f"{'done' if ok else 'FAILED'} {rid} rc={rc}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a_run = sub.add_parser("run")
    a_run.add_argument("--lane", choices=["all", "train", "eval"], default="all")
    sub.add_parser("status")
    a_cmd = sub.add_parser("add-cmd", help="queue a python module invocation")
    a_cmd.add_argument("run_id")
    a_cmd.add_argument("--desc", required=True)
    a_cmd.add_argument("--fallback", action="store_true")
    a_cmd.add_argument("--after", help="run only after this entry is done")
    a_cmd.add_argument("args", nargs=argparse.REMAINDER, help="-m module args...")
    a_add = sub.add_parser("add")
    a_add.add_argument("run_id")
    a_add.add_argument("--config")
    a_add.add_argument("--set", nargs="*", default=[])
    a_add.add_argument("--desc", required=True)
    a_add.add_argument("--no-ledger", action="store_true")
    a_add.add_argument("--module", default="edgesr.train",
                       help="training module, e.g. edgesr.train_cfi")
    a_add.add_argument("--fallback", action="store_true",
                       help="low priority: runs only when no normal entry is pending")
    a = ap.parse_args()
    if a.cmd == "run":
        sys.exit(run(a.lane))
    if a.cmd == "add-cmd":
        q = load()
        assert all(e["run_id"] != a.run_id for e in q), "duplicate run_id"
        q.append({"run_id": a.run_id, "cmd": a.args, "desc": a.desc, "status": "pending",
                  "fallback": a.fallback, "after": a.after})
        save(q)
    if a.cmd == "add":
        if a.module == "edgesr.train_cfi":
            from .train_cfi import default_config
        elif a.module == "edgesr.train_coarse":
            from .train_coarse import default_config
        elif a.module == "edgesr.train_sdf":
            from .train_sdf import default_config
        else:
            from .train import default_config
        from .train import apply_sets
        apply_sets(default_config(), a.set)   # fail now, not hours later in the runner
        q = load()
        assert all(e["run_id"] != a.run_id for e in q), "duplicate run_id"
        q.append({"run_id": a.run_id, "config": a.config, "set": a.set, "desc": a.desc,
                  "status": "pending", "no_ledger": a.no_ledger, "fallback": a.fallback,
                  "module": a.module})
        save(q)
    for e in load():
        tag = " [fallback]" if e.get("fallback") else ""
        print(f"{e.get('status', 'pending'):8s} {e['run_id']:32s} {e['desc'][:60]}{tag}")


if __name__ == "__main__":
    main()
