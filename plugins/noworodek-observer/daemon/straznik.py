#!/usr/bin/env python3
# ==========================================
# AUTHOR: M. SZUL
# AI MODEL: Claude Opus 5
# TIMESTAMP: 2026-08-26 21:27:51
# REASON FOR CREATION: A learner that runs unattended overnight must not be able to take
#   the machine down with it. "Trying to be polite" is not a limit - a runaway loop is
#   impolite by accident. This puts a hard ceiling on what the training daemon may use and
#   gives one command that stops it dead.
# MECHANICS: Launches a child process pinned to a chosen subset of cores, watches its
#   memory, and kills it if it crosses the ceiling. Writes a heartbeat so anything can see
#   whether it is alive and what it is costing. Honours a stop file, checked every second,
#   so stopping never requires finding a process id.
# SYSTEM PART: Noworodek, training daemon.
# ARCHITECTURE FUNCTION: The safety layer under the learner. Nothing trains on this machine
#   except through here.
# DEPENDENCIES/LINKS: wraps any command; used by the training daemon; heartbeat read by the
#   dashboard.
# TECH STACK: Python 3, standard library plus optional psutil. Standard library because
#   this must run when everything else is broken, and a safety wrapper with dependencies
#   is a safety wrapper that can fail to start.
# LOCAL WORKSPACE: C:\Users\User\.claude\noworodek-observer\daemon\straznik.py
# GIT COMMIT: PENDING
# GITHUB METADATA: jpytka666-jpg/wpc-engine, branch noworodek-cbms-training
# ==========================================
"""Straznik - hard limits and a kill switch for anything that learns."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
ROOT = HOME / ".claude" / "noworodek-observer" / "daemon"
STOP_FILE = ROOT / "STOP"
HEARTBEAT = ROOT / "heartbeat.json"

# Defaults agreed after measuring what the machine has: 4 of 8 threads and 8 GB of 64.
# The point is not that the learner needs little - it is that M. Szul must still be able
# to work while it runs.
DEFAULT_CORES = 4
DEFAULT_MEMORY_GB = 8.0
CHECK_EVERY_S = 1.0


def rss_bytes(pid):
    """Resident memory of a process and its children, best effort."""
    try:
        import psutil  # optional; the wrapper must work without it
    except ImportError:
        return None
    try:
        proc = psutil.Process(pid)
        total = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except Exception:
                pass
        return total
    except Exception:
        return None


def pin_to_cores(pid, cores):
    """Restrict a process to the first `cores` logical processors.

    An affinity mask is a hard limit the scheduler enforces. Asking a process to behave
    is not - it has no idea how busy the rest of the machine is.
    """
    try:
        import psutil
        psutil.Process(pid).cpu_affinity(list(range(cores)))
        return True
    except Exception:
        return False


def beat(state):
    state["at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
        HEARTBEAT.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass  # a lost heartbeat must never stop the thing it reports on


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cores", type=int, default=DEFAULT_CORES)
    ap.add_argument("--memory-gb", type=float, default=DEFAULT_MEMORY_GB)
    ap.add_argument("--label", default="nauka")
    ap.add_argument("--log", default=None, help="where to append the child's output")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    if not args.command:
        print("straznik.py [--cores N] [--memory-gb G] [--label nazwa] -- <polecenie>")
        print(f"zatrzymanie: utworz plik {STOP_FILE}")
        return 2
    cmd = args.command[1:] if args.command[0] == "--" else args.command

    ROOT.mkdir(parents=True, exist_ok=True)
    # A stop file left from last time would kill the next run before it started.
    if STOP_FILE.exists():
        STOP_FILE.unlink()

    log_handle = open(args.log, "a", encoding="utf-8") if args.log else None
    started = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=log_handle or subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    pinned = pin_to_cores(proc.pid, args.cores)
    limit_bytes = int(args.memory_gb * (1024 ** 3))
    state = {
        "label": args.label,
        "pid": proc.pid,
        "command": " ".join(cmd)[:400],
        "cores": args.cores,
        "pinned": pinned,
        "memory_limit_gb": args.memory_gb,
        "status": "biegnie",
        "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
    }
    beat(state)
    print(f"straznik: {args.label} pid={proc.pid} rdzeni={args.cores} "
          f"pamiec<={args.memory_gb} GB przypiete={'tak' if pinned else 'NIE'}")
    if not pinned:
        print("  UWAGA: nie udalo sie przypiac do rdzeni (brak psutil?) - limit rdzeni NIE dziala")

    peak = 0
    reason = None
    try:
        while proc.poll() is None:
            time.sleep(CHECK_EVERY_S)
            if STOP_FILE.exists():
                reason = "zatrzymany_recznie"
                break
            used = rss_bytes(proc.pid)
            if used is not None:
                peak = max(peak, used)
                if used > limit_bytes:
                    reason = "przekroczyl_pamiec"
                    break
            state.update(
                elapsed_s=round(time.time() - started, 1),
                memory_gb=round((used or 0) / (1024 ** 3), 2),
                peak_gb=round(peak / (1024 ** 3), 2),
            )
            beat(state)
    except KeyboardInterrupt:
        reason = "przerwany"

    if reason:
        # Terminate, then insist. A learner mid-write deserves a moment to stop cleanly;
        # it does not get to refuse.
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        state["status"] = reason
    else:
        state["status"] = "skonczyl" if proc.returncode == 0 else f"blad_{proc.returncode}"

    state.update(
        elapsed_s=round(time.time() - started, 1),
        peak_gb=round(peak / (1024 ** 3), 2),
        exit_code=proc.returncode,
    )
    beat(state)
    if log_handle:
        log_handle.close()

    print(f"straznik: {state['status']} po {state['elapsed_s']} s, "
          f"szczyt pamieci {state['peak_gb']} GB")
    return 0 if state["status"] == "skonczyl" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"straznik: awaria wrappera: {exc}", file=sys.stderr)
        sys.exit(1)
