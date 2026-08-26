#!/usr/bin/env python3
# ==========================================
# AUTHOR: M. SZUL
# AI MODEL: Claude Opus 5
# TIMESTAMP: 2026-08-26 22:07:24
# REASON FOR CREATION: A cycle runs once and exits, which is what makes it safe. Something
#   still has to decide whether to run another. Putting that decision inside the trainer
#   would give it an endless loop to debug while running; putting it here keeps every
#   individual run finite and every decision visible.
# MECHANICS: Checks resources, runs one cycle, reads the state it wrote, decides. Stops on
#   the first ERROR, on the stop file, on a run of plateaus, on a wall-clock budget, or
#   when there is nothing left to learn from. Each decision is recorded with its reason, so
#   the question "why did it stop at four in the morning" has an answer that does not
#   require reading a log.
# SYSTEM PART: Noworodek, training daemon.
# ARCHITECTURE FUNCTION: The operator. AIONS's boot conductor starts this; it starts
#   cycles. Nothing below it decides how long to keep going.
# DEPENDENCIES/LINKS: cykl.py, straznik.py, stan.json, historia.jsonl.
# TECH STACK: Python 3, standard library.
# LOCAL WORKSPACE: C:\Users\User\.claude\noworodek-observer\daemon\dyrygent.py
# GIT COMMIT: PENDING
# GITHUB METADATA: jpytka666-jpg/wpc-engine, branch noworodek-cbms-training
# ==========================================
"""Dyrygent - decyduje, ile cykli nauki uruchomic i kiedy przestac."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
DAEMON = HOME / ".claude" / "noworodek-observer" / "daemon"
STATE_FILE = DAEMON / "stan.json"
STOP_FILE = DAEMON / "STOP"
RUN_LOG = DAEMON / "przebiegi.jsonl"

# Below this the machine has no room to learn in, and taking the last of it would make
# the owner's own work miserable. Checked before every cycle, not only at the start.
MIN_FREE_GB = 6.0
MIN_DISK_GB = 5.0


def free_memory_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        return None


def free_disk_gb(path):
    try:
        return shutil.disk_usage(path).free / (1024 ** 3)
    except Exception:
        return None


def read_state():
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def record(entry):
    entry["at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with RUN_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-cykli", type=int, default=12)
    ap.add_argument("--max-minut", type=float, default=180.0)
    ap.add_argument("--plateau-pod-rzad", type=int, default=3,
                    help="ile plateau z rzedu konczy przebieg")
    ap.add_argument("--przerwa-s", type=float, default=5.0)
    # Everything after -- is handed to the cycle unchanged, so the conductor never has to
    # know what a code book is.
    ap.add_argument("cykl_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    cycle_args = args.cykl_args[1:] if args.cykl_args and args.cykl_args[0] == "--" else args.cykl_args
    if not cycle_args:
        print("dyrygent.py [--max-cykli N] [--max-minut M] -- <argumenty dla cykl.py>")
        return 2

    DAEMON.mkdir(parents=True, exist_ok=True)
    started = time.time()
    plateau_run = 0
    cycles = 0
    best = None
    summary = {"cykli": 0, "learning": 0, "plateau": 0, "powod": None}

    print(f"dyrygent: start, do {args.max_cykli} cykli, do {args.max_minut} minut")
    record({"zdarzenie": "start", "max_cykli": args.max_cykli, "max_minut": args.max_minut})

    while True:
        if STOP_FILE.exists():
            summary["powod"] = "plik STOP"
            break
        if cycles >= args.max_cykli:
            summary["powod"] = f"limit cykli ({args.max_cykli})"
            break
        minutes = (time.time() - started) / 60
        if minutes >= args.max_minut:
            summary["powod"] = f"limit czasu ({args.max_minut} min)"
            break

        mem = free_memory_gb()
        disk = free_disk_gb(DAEMON)
        if mem is not None and mem < MIN_FREE_GB:
            summary["powod"] = f"za malo wolnej pamieci ({mem:.1f} GB < {MIN_FREE_GB})"
            record({"zdarzenie": "wstrzymanie", "powod": summary["powod"]})
            break
        if disk is not None and disk < MIN_DISK_GB:
            summary["powod"] = f"za malo miejsca na dysku ({disk:.1f} GB < {MIN_DISK_GB})"
            record({"zdarzenie": "wstrzymanie", "powod": summary["powod"]})
            break

        cycles += 1
        print(f"\n--- cykl {cycles} (wolne: {mem:.1f} GB pamieci, {disk:.1f} GB dysku) ---"
              if mem and disk else f"\n--- cykl {cycles} ---")
        proc = subprocess.run([sys.executable, str(DAEMON / "cykl.py")] + cycle_args,
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
        tail = "\n".join((proc.stdout or "").strip().splitlines()[-4:])
        if tail:
            print(tail)

        state = read_state()
        stan = state.get("stan", "ERROR")
        po = state.get("po")
        if po is not None and (best is None or po < best):
            best = po

        record({"zdarzenie": "cykl", "numer": cycles, "stan": stan,
                "przed": state.get("przed"), "po": po, "najlepszy": best,
                "przyjete": state.get("przyjete"), "odrzucone": state.get("odrzucone"),
                "sekund": state.get("sekund")})

        # An error stops the run at once. Carrying on past one means the next cycle starts
        # from a state nobody has looked at, which is how a small fault becomes a long
        # night of quietly wrong training.
        if stan == "ERROR":
            summary["powod"] = f"ERROR w cyklu {cycles}: {state.get('krok')} / {state.get('powod', '')}"
            break
        if stan == "STOP":
            summary["powod"] = f"cykl {cycles}: {state.get('powod', 'nic do nauki')}"
            break
        if stan == "LEARNING":
            summary["learning"] += 1
            plateau_run = 0
        elif stan == "PLATEAU":
            summary["plateau"] += 1
            plateau_run += 1
            if plateau_run >= args.plateau_pod_rzad:
                summary["powod"] = f"{plateau_run} plateau z rzedu - nie ma czego wycisnac"
                break

        time.sleep(args.przerwa_s)

    summary["cykli"] = cycles
    summary["najlepszy"] = best
    summary["minut"] = round((time.time() - started) / 60, 1)
    record({"zdarzenie": "koniec", **summary})

    print()
    print(f"dyrygent: koniec po {cycles} cyklach, {summary['minut']} min")
    print(f"  uczacych sie : {summary['learning']}")
    print(f"  plateau      : {summary['plateau']}")
    print(f"  najlepszy    : {best}")
    print(f"  powod konca  : {summary['powod']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ndyrygent: przerwany recznie")
        sys.exit(1)
    except Exception as exc:
        record({"zdarzenie": "awaria", "szczegol": f"{type(exc).__name__}: {exc}"})
        print(f"dyrygent: awaria: {exc}", file=sys.stderr)
        sys.exit(1)
