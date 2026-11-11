"""Resume a local catalog render run when its scheduler is no longer active.

Usage: watch_full_catalog.py [full_catalog|physical_v3]   (default full_catalog)
"""

import fcntl
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRIVE = Path("/media/kyle/T7 Shield")
DATA = DRIVE / "inspection-synthetic-renders"
CATALOG = DATA / "review-300-2048/catalog.json"
# name -> (output directory, target bytes, extra synth.parallel arguments)
RUNS = {
    "full_catalog": (DATA / "full-catalog-5gb-2048", 5_000_000_000, (
        "--seed", "20260923", "--target-gb", "5", "--samples", "128", "--tile-size", "2048",
        "--workers", "8", "--threads-per-worker", "2")),
    # GPU Cycles; untilted masks are shared from the full_catalog run.
    "physical_v3": (DATA / "physical-v3-30gb-2048", 30_000_000_000, (
        "--profile", "physical_v3", "--seed", "20260927", "--target-gb", "30",
        "--tile-size", "2048", "--workers", "6", "--threads-per-worker", "3",
        "--base-masks", str(DATA / "full-catalog-5gb-2048"))),
}
RUN = sys.argv[1] if len(sys.argv) > 1 else "full_catalog"
OUTPUT, TARGET_BYTES, EXTRA = RUNS[RUN]
COMMAND = (sys.executable, "-m", "synth.parallel", "--catalog", str(CATALOG),
           *EXTRA, "--output", str(OUTPUT))


def _message(message):
    print(f"{datetime.now(timezone.utc).isoformat()} {message}", flush=True)


def _matching_pids():
    pids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        args = [item.decode("utf-8", "replace") for item in argv if item]
        if ("-m" in args and args[args.index("-m") + 1:args.index("-m") + 2] == ["synth.parallel"]
                and "--output" in args
                and args[args.index("--output") + 1:args.index("--output") + 2] == [str(OUTPUT)]):
            pids.append(int(entry.name))
    return pids


def _completed_bytes():
    # Recount after a process exits: progress.json can lag the final in-flight scenes.
    total = 0
    for scene in OUTPUT.glob("scene_*"):
        if (scene / "metadata.json").is_file():
            for root, _, names in os.walk(scene):
                for name in names:
                    total += (Path(root) / name).stat().st_size
    return total


def main():
    if DRIVE.is_mount():
        OUTPUT.mkdir(exist_ok=True)
    if not DRIVE.is_mount() or not OUTPUT.is_dir() or not CATALOG.is_file():
        _message("ERROR: render drive, output directory, or catalog is unavailable")
        return 1
    with (OUTPUT / ".watchdog.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (OUTPUT / ".watchdog.paused").exists():
            _message("PAUSED by request")
            return 0
        pids = _matching_pids()
        if pids:
            _message(f"RUNNING scheduler PID(s) {pids}")
            return 0
        completed = _completed_bytes()
        if completed >= TARGET_BYTES:
            _message(f"COMPLETE {completed} / {TARGET_BYTES} bytes")
            return 0
        _message(f"RESTART {completed} / {TARGET_BYTES} bytes")
        with (OUTPUT / "run.log").open("a") as log:
            log.write(f"WATCHDOG_RESUME {datetime.now(timezone.utc).isoformat()}\n")
            log.flush()
            process = subprocess.Popen(COMMAND, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        _message(f"STARTED scheduler PID {process.pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
