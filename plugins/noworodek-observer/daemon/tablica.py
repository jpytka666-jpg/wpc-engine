#!/usr/bin/env python3
# ==========================================
# AUTHOR: M. SZUL
# AI MODEL: Claude Opus 5
# TIMESTAMP: 2026-08-26 22:09:42
# REASON FOR CREATION: A daemon that learns overnight is only useful if its owner can see
#   what it did without reading logs. Everything it records is already on disk; nothing
#   assembles it into an answer to "what happened last night".
# MECHANICS: Reads the current state, the run log and the cycle history, and prints the
#   nine things worth knowing. Read-only by construction - a dashboard that can change
#   what it reports is a dashboard that can lie about it.
# SYSTEM PART: Noworodek, training daemon.
# ARCHITECTURE FUNCTION: The window. Everything shown here is measured by something else;
#   this only arranges it.
# DEPENDENCIES/LINKS: reads stan.json, historia.jsonl, przebiegi.jsonl, heartbeat.json and
#   the lesson rejection log. Writes nothing.
# TECH STACK: Python 3, standard library.
# LOCAL WORKSPACE: C:\Users\User\.claude\noworodek-observer\daemon\tablica.py
# GIT COMMIT: PENDING
# GITHUB METADATA: jpytka666-jpg/wpc-engine, branch noworodek-cbms-training
# ==========================================
"""Tablica przyrzadow - co demon zrobil, bez czytania dziennikow."""

import argparse
import json
import os
import time
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
DAEMON = HOME / ".claude" / "noworodek-observer" / "daemon"


def load_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default if default is not None else {}


def load_lines(path, limit=None):
    try:
        lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return lines[-limit:] if limit else lines
    except Exception:
        return []


def age(stamp):
    try:
        then = time.mktime(time.strptime(stamp, "%Y-%m-%d %H:%M:%S"))
        secs = int(time.time() - then)
        if secs < 60:
            return f"{secs} s temu"
        if secs < 3600:
            return f"{secs // 60} min temu"
        if secs < 86400:
            return f"{secs // 3600} godz temu"
        return f"{secs // 86400} dni temu"
    except Exception:
        return "?"


def bar(value, floor, ceiling, width=28):
    """Where the held-out figure sits between chance and the frequency baseline."""
    if value is None or floor is None or ceiling is None or ceiling <= floor:
        return ""
    pos = (ceiling - value) / (ceiling - floor)
    pos = max(0.0, min(1.2, pos))
    filled = int(pos * width)
    return "[" + "#" * min(filled, width) + "." * max(0, width - filled) + "]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ile", type=int, default=10, help="ile ostatnich cykli pokazac")
    args = ap.parse_args()

    state = load_json(DAEMON / "stan.json")
    heart = load_json(DAEMON / "heartbeat.json")
    runs = load_lines(DAEMON / "przebiegi.jsonl")
    history = [h for h in load_lines(DAEMON / "historia.jsonl")
               if h.get("stan") in ("LEARNING", "PLATEAU")]

    stan = state.get("stan", "?")
    print("=" * 66)
    print(f"  NOWORODEK   stan: {stan}   ({age(state.get('at', ''))})")
    print("=" * 66)

    if stan == "ERROR":
        print(f"  BLAD w kroku : {state.get('krok', '?')}")
        print(f"  powod        : {state.get('powod', '?')}")
        if state.get("plik"):
            print(f"  plik         : {state['plik']}   (NIE ruszony)")
        print()
    elif stan == "STOP":
        print(f"  powod        : {state.get('powod', '?')}")
        print()

    # What the last cycle actually did.
    przed, po = state.get("przed"), state.get("po")
    chance, unigram = state.get("prog_kostka"), state.get("prog_czestosc")
    best = min((h["po"] for h in history if h.get("po") is not None), default=None)

    print(f"  cykli w dzienniku : {len(history)}")
    if przed is not None and po is not None:
        kier = "lepiej" if po < przed - 0.01 else ("bez zmian" if po <= przed + 0.01 else "GORZEJ")
        print(f"  ostatni cykl      : {przed:.4f} -> {po:.4f}   ({kier})")
    # What the checkpoint actually holds is `po`. The best ever seen may come from a run
    # whose weights were replaced or deleted, and a dashboard that prints the better of
    # the two without saying which is which claims the model is better than it is.
    if best is not None:
        if po is not None and best < po - 0.0001:
            print(f"  najlepszy KIEDYKOLWIEK : {best:.4f}   (NIE w obecnym checkpoincie)")
            print(f"  obecny checkpoint      : {po:.4f}")
        else:
            print(f"  najlepszy = obecny     : {best:.4f}")
    if unigram is not None:
        print(f"  prog czestosci    : {unigram:.4f}   <- ponizej = uzywa kontekstu")
    if chance is not None:
        print(f"  prog kostki       : {chance:.4f}")
    if po is not None and unigram is not None and chance is not None:
        print(f"  gdzie jest        : {bar(po, unigram, chance)}")
        if po < unigram:
            print(f"                      ponizej progu czestosci o {unigram - po:.4f}")

    print()
    print(f"  lekcje przyjete   : {state.get('przyjete', '?')}")
    print(f"  lekcje odrzucone  : {state.get('odrzucone', '?')}")
    print(f"  symboli w cyklu   : {state.get('numerow', '?')}")
    print(f"  slownik           : {state.get('slownik', '?')}")
    print(f"  czas cyklu        : {state.get('sekund', '?')} s")

    wagi = Path(state.get("wagi", "")) if state.get("wagi") else DAEMON / "praca" / "wagi.bin"
    if wagi.exists():
        mb = wagi.stat().st_size / (1024 ** 2)
        print(f"  checkpoint        : {mb:.1f} MB, {age(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(wagi.stat().st_mtime)))}")
    else:
        print("  checkpoint        : BRAK")

    if heart:
        print(f"  ostatni straznik  : {heart.get('status', '?')}, "
              f"szczyt {heart.get('peak_gb', '?')} GB, {heart.get('cores', '?')} rdzeni")

    # Rejections say what the gate is refusing. A gate refusing everything is as much a
    # problem as a gate refusing nothing.
    rejects = load_lines(DAEMON / "praca" / "odrzucone-lekcje.jsonl")
    if rejects:
        counts = {}
        for r in rejects:
            counts[r.get("reason", "?")] = counts.get(r.get("reason", "?"), 0) + 1
        print()
        print(f"  odrzucenia bramki ({len(rejects)} lacznie):")
        for reason, n in sorted(counts.items(), key=lambda kv: -kv[1])[:5]:
            print(f"      {n:>4}  {reason}")

    if history:
        print()
        print(f"  ostatnie {min(args.ile, len(history))} cykli:")
        for h in history[-args.ile:]:
            p, q = h.get("przed"), h.get("po")
            mark = "  " if h["stan"] == "PLATEAU" else "->"
            print(f"      {h['at'][11:]}  {h['stan']:<8} {p if p is None else f'{p:.4f}'} {mark} "
                  f"{q if q is None else f'{q:.4f}'}")

    last_run = next((r for r in reversed(runs) if r.get("zdarzenie") == "koniec"), None)
    if last_run:
        print()
        print(f"  ostatni przebieg  : {last_run.get('cykli')} cykli, {last_run.get('minut')} min, "
              f"uczacych {last_run.get('learning')}, plateau {last_run.get('plateau')}")
        print(f"  dlaczego stanal   : {last_run.get('powod')}")

    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
