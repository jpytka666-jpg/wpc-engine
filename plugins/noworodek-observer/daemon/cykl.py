#!/usr/bin/env python3
# ==========================================
# AUTHOR: M. SZUL
# AI MODEL: Claude Opus 5
# TIMESTAMP: 2026-08-26 21:47:54
# REASON FOR CREATION: The parts exist and none of them talk to each other - the observer
#   writes traces, the gate refuses the worthless ones, the codec makes symbols, the
#   trainer learns and stops when it stops learning, the guard keeps it off the user's
#   machine. This is the loop that runs them in order, once, and says plainly what
#   happened.
# MECHANICS: One cycle, then exit. A daemon that loops forever is a daemon that has to be
#   debugged while running; a cycle that exits can be scheduled, watched, and killed by
#   deleting nothing. Reads traces, refuses what teaches nothing, extends the code book,
#   encodes to symbols, trains from the previous cycle's weights under the resource guard,
#   and records the result as one of five states.
# SYSTEM PART: Noworodek, training daemon.
# ARCHITECTURE FUNCTION: The element AIONS's conductor starts. Everything else is a part;
#   this is the machine.
# DEPENDENCIES/LINKS: straznik.py; record-event.mjs traces; traces-to-lessons.mjs;
#   cbms binary (build, ids); train-cbms binary.
# TECH STACK: Python 3, standard library. Same reason as the guard: this has to run when
#   other things are broken.
# LOCAL WORKSPACE: C:\Users\User\.claude\noworodek-observer\daemon\cykl.py
# GIT COMMIT: PENDING
# GITHUB METADATA: jpytka666-jpg/wpc-engine, branch noworodek-cbms-training
# ==========================================
"""Jeden cykl nauki Noworodka: slady -> lekcje -> symbole -> trening -> stan."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
ROOT = HOME / ".claude" / "noworodek-observer"
DAEMON = ROOT / "daemon"
WORK = DAEMON / "praca"
STATE_FILE = DAEMON / "stan.json"
HISTORY = DAEMON / "historia.jsonl"
STOP_FILE = DAEMON / "STOP"

# The five states M. Szul asked for. Nothing else may be written here: a state vocabulary
# that grows is a state vocabulary nobody can act on.
RUN, LEARNING, PLATEAU, STOP, ERROR = "RUN", "LEARNING", "PLATEAU", "STOP", "ERROR"

# Improvement smaller than this is noise in the third decimal, not learning.
REAL_IMPROVEMENT = 0.01


def write_state(state, **fields):
    payload = {"stan": state, "at": time.strftime("%Y-%m-%d %H:%M:%S"), **fields}
    DAEMON.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    with HISTORY.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload


def run(cmd, cwd=None, timeout=None):
    """Run and return (ok, output). Never raises: a cycle that dies on a subprocess
    leaves no state behind, and a daemon with no state is a daemon nobody can inspect."""
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           timeout=timeout, encoding="utf-8", errors="replace")
        return r.returncode == 0, (r.stdout or "") + (r.stderr or "")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def number_after(text, label):
    m = re.search(rf"{label}\s*:?\s*([0-9]+\.[0-9]+)", text)
    return float(m.group(1)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", default=str(Path("D:/AIONS/.noworodek/observer-data/traces")))
    ap.add_argument("--cbms", required=True, help="sciezka do binarki cbms")
    ap.add_argument("--trainer", required=True, help="sciezka do binarki train-cbms")
    ap.add_argument("--book", required=True, help="ksiazka kodow na starcie")
    ap.add_argument("--scripts", default=str(ROOT / "scripts"))
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--memory-gb", type=float, default=8.0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--inter", type=int, default=256)
    ap.add_argument("--patience", type=int, default=4)
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    started = time.time()

    if STOP_FILE.exists():
        write_state(STOP, powod="plik STOP istnieje")
        print("STOP: plik STOP istnieje, nic nie robie")
        return 0

    write_state(RUN, krok="start")

    # ---- 1. traces -> lessons, through the gate --------------------------------
    lessons = WORK / "lekcje.txt"
    ok, out = run(["node", str(Path(args.scripts) / "traces-to-lessons.mjs"),
                   args.traces, str(lessons)], timeout=300)
    if not ok:
        write_state(ERROR, krok="lekcje", szczegol=out[-600:])
        print(out[-600:])
        return 1
    accepted = int(re.search(r"lekcji PRZYJETYCH\s*:\s*(\d+)", out).group(1)) if "PRZYJETYCH" in out else 0
    rejected = int(re.search(r"lekcji ODRZUCONYCH\s*:\s*(\d+)", out).group(1)) if "ODRZUCONYCH" in out else 0
    chars = lessons.stat().st_size if lessons.exists() else 0
    print(f"lekcje: przyjete {accepted}, odrzucone {rejected}, znakow {chars}")

    # Nothing worth learning from is not a failure. It is the normal state of a machine
    # whose owner spent the day reading rather than verifying.
    if accepted == 0 or chars < 200:
        write_state(STOP, powod="brak nowych lekcji", przyjete=accepted, odrzucone=rejected)
        print("STOP: nie ma z czego sie uczyc w tym cyklu")
        return 0

    # ---- 2. extend the book, encode ---------------------------------------------
    book_out = WORK / "ksiazka.txt"
    ok, out = run([args.cbms, args.book, "build", str(lessons), str(book_out), "20000", "1"], timeout=600)
    if not ok or not book_out.exists():
        write_state(ERROR, krok="ksiazka", szczegol=out[-600:])
        print(out[-600:])
        return 1

    ids = WORK / "lekcje.u16"
    ok, out = run([args.cbms, str(book_out), "ids", str(lessons), str(ids)], timeout=600)
    if not ok:
        write_state(ERROR, krok="symbole", szczegol=out[-600:])
        print(out[-600:])
        return 1
    vocab = int(re.search(r"vocab size\s*:\s*(\d+)", out).group(1))
    n_ids = int(re.search(r"^ids\s*:\s*(\d+)", out, re.M).group(1))
    print(f"symbole: {n_ids} numerow, slownik {vocab}")

    # ---- 3. train, under the guard, from the last cycle's weights -----------------
    prev = WORK / "wagi.bin"
    nxt = WORK / "wagi-nowe.bin"
    cmd = [args.trainer, "--ids", str(ids), "--vocab", str(vocab),
           "--hidden", str(args.hidden), "--inter", str(args.inter),
           "--steps", str(args.steps), "--patience", str(args.patience),
           "--save", str(nxt)]
    if prev.exists():
        # A checkpoint the trainer refuses must not block every future night. It is set
        # aside under a dated name - never deleted, because it may be the only copy of
        # something worth recovering - and the cycle starts fresh, saying so out loud.
        ok_probe, probe = run([args.trainer, "--ids", str(ids), "--vocab", str(vocab),
                               "--hidden", str(args.hidden), "--inter", str(args.inter),
                               "--steps", "1", "--patience", "0", "--load", str(prev)],
                              timeout=600)
        if ok_probe:
            cmd += ["--load", str(prev)]
        else:
            reason = probe.strip().splitlines()[-1] if probe.strip() else "nieznany powod"
            # Two different faults that must not share a response.
            #
            # A file this build simply cannot read - written by an older format, or a
            # newer one - is a compatibility problem. Quarantine it and start fresh: no
            # learning is lost that was not already unreachable.
            #
            # A file that IS ours and is damaged is a different matter. Starting fresh
            # there would throw away real accumulated learning on the strength of a guess
            # about what went wrong. That stops and waits for a person.
            incompatible = ("to nie jest plik wag" in reason) or ("wersja pliku" in reason)
            if incompatible:
                aside = prev.with_name(f"wagi-niezgodne-{time.strftime('%Y%m%d-%H%M%S')}.bin")
                prev.rename(aside)
                print(f"NIEZGODNY FORMAT: {reason}")
                print(f"       odlozony do {aside.name}, cykl zaczyna od nowa")
                write_state(RUN, krok="niezgodny_format", odlozony=str(aside), powod=reason)
            else:
                write_state(ERROR, krok="uszkodzony_checkpoint", powod=reason,
                            plik=str(prev))
                print(f"USZKODZONY CHECKPOINT: {reason}")
                print(f"       plik {prev} NIE zostal ruszony.")
                print("       Nie zaczynam od zera na wlasna reke - to by wyrzucilo")
                print("       dotychczasowa nauke na podstawie domyslu. Obejrzyj go.")
                return 1

    log = WORK / "trening.log"
    if log.exists():
        log.unlink()
    write_state(RUN, krok="trening", przyjete=accepted, numerow=n_ids, slownik=vocab)

    ok, guard_out = run(
        [sys.executable, str(DAEMON / "straznik.py"),
         "--cores", str(args.cores), "--memory-gb", str(args.memory_gb),
         "--label", "nauka", "--log", str(log), "--"] + cmd,
        timeout=None,
    )
    trained = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""

    if not nxt.exists():
        write_state(ERROR, krok="trening", szczegol=(guard_out + trained)[-800:])
        print((guard_out + trained)[-800:])
        return 1

    before = number_after(trained, "odlozone po wczytaniu") or number_after(trained, "odlozone PRZED")
    after = number_after(trained, "odlozone PO")
    unigram = number_after(trained, r"PROG 2, czestosc")
    chance = number_after(trained, r"PROG 1, kostka")

    # ---- 4. decide, and only then adopt the new weights --------------------------
    # The trainer already refuses to save something worse than it was given, so adopting
    # is safe. This second check exists because a guarantee that is only asserted in one
    # place is a guarantee that breaks the day that place is edited.
    improved = before is not None and after is not None and after < before - REAL_IMPROVEMENT
    regressed = before is not None and after is not None and after > before + REAL_IMPROVEMENT

    if regressed:
        write_state(ERROR, krok="regresja", przed=before, po=after,
                    szczegol="nowe wagi gorsze niz poprzednie - NIE przyjeto")
        nxt.unlink(missing_ok=True)
        print(f"ODRZUCONO: {before:.4f} -> {after:.4f}, nowe wagi skasowane")
        return 1

    nxt.replace(prev)
    elapsed = round(time.time() - started, 1)
    state = LEARNING if improved else PLATEAU
    payload = write_state(
        state, przed=before, po=after, prog_czestosc=unigram, prog_kostka=chance,
        przyjete=accepted, odrzucone=rejected, numerow=n_ids, slownik=vocab,
        sekund=elapsed, wagi=str(prev),
    )
    print(f"{state}: {before} -> {after}  (prog czestosci {unigram})  w {elapsed} s")
    if unigram and after and after < unigram:
        print("       ponizej progu czestosci - uzywa kontekstu")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        write_state(ERROR, krok="awaria", szczegol=f"{type(exc).__name__}: {exc}")
        print(f"awaria cyklu: {exc}", file=sys.stderr)
        sys.exit(1)
