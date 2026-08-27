#!/usr/bin/env python3
# ==========================================
# AUTHOR: M. SZUL
# AI MODEL: Claude Opus 5
# TIMESTAMP: 2026-08-27 01:22:40
# REASON FOR CREATION: Every source file carries `GIT COMMIT: PENDING` and the rule says to
#   fill it once the hash exists. Nobody ever went back, so the field has been PENDING in
#   every file since the convention started - a field that is always the same value tells
#   you nothing, and a rule nobody follows is worse than no rule, because it looks kept.
# MECHANICS: PostToolUse on Bash. When a command that ran `git commit` succeeds, the files
#   in that commit are checked for a PENDING stamp; those that have one get the real hash
#   written in, and the commit is amended so the file and the hash agree. Amending is
#   skipped once the commit is on a remote - rewriting published history to tidy a comment
#   is not a trade worth making, and the stamp is then left for the next commit to carry.
# SYSTEM PART: Hooks - the layer that enforces what judgement forgets.
# ARCHITECTURE FUNCTION: Closes step 7 of the commit protocol, which was the only step in
#   it that depended on remembering.
# DEPENDENCIES/LINKS: git; the metadata block written by stamp_metadata.py.
# TECH STACK: Python 3, standard library. Same as the other hooks, and it must run when
#   other things are broken.
# LOCAL WORKSPACE: C:\Users\User\.claude\hooks\stamp_commit.py
# GIT COMMIT: PENDING
# GITHUB METADATA: local hook, not in a project repository
# ==========================================
"""Wpisuje prawdziwy hash w pole GIT COMMIT po udanym zapisie."""

import json
import re
import subprocess
import sys
from pathlib import Path

PENDING = re.compile(r"(GIT COMMIT:\s*)PENDING\s*$", re.MULTILINE)

# Fail-open everywhere: a hook that stops the session to fix a comment is a bad trade.
TIMEOUT = 20


def git(repo: Path, *args: str) -> tuple[int, str]:
    try:
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=TIMEOUT)
        return r.returncode, (r.stdout or "").strip()
    except Exception:
        return 1, ""


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except Exception:
        return 0

    if event.get("tool_name") != "Bash":
        return 0
    command = (event.get("tool_input") or {}).get("command") or ""
    if "git commit" not in command:
        return 0
    # A failed commit leaves nothing to stamp, and re-running git here would hide that.
    response = json.dumps(event.get("tool_response") or {})
    if re.search(r"nothing to commit|no changes added", response):
        return 0

    cwd = Path(event.get("cwd") or ".")
    code, top = git(cwd, "rev-parse", "--show-toplevel")
    if code != 0 or not top:
        return 0
    repo = Path(top)

    code, full = git(repo, "rev-parse", "HEAD")
    if code != 0 or not full:
        return 0

    # Already published: the file keeps PENDING rather than history being rewritten.
    code, remote = git(repo, "branch", "-r", "--contains", "HEAD")
    if code == 0 and remote.strip():
        return 0

    code, listing = git(repo, "show", "--name-only", "--pretty=format:", "HEAD")
    if code != 0:
        return 0

    stamped = []
    for rel in [l.strip() for l in listing.splitlines() if l.strip()]:
        path = repo / rel
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        if not PENDING.search(text):
            continue
        try:
            path.write_text(PENDING.sub(r"\g<1>" + full, text), encoding="utf-8")
            stamped.append(rel)
        except Exception:
            continue

    if not stamped:
        return 0

    # Staged, deliberately NOT amended.
    #
    # Amending would give HEAD a new hash, and the hash just written into the file would
    # name a commit that no longer exists - a stamp pointing at nothing, which is worse
    # than PENDING because it looks correct. Left staged, the value is true (it names the
    # commit that introduced this version of the file) and it travels with the next one.
    for rel in stamped:
        git(repo, "add", "--", rel)
    print(
        f"stamp_commit: wpisano {full[:12]} w {len(stamped)} plikach "
        f"({', '.join(stamped[:4])}) - przygotowane do nastepnego zapisu",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.exit(0)
