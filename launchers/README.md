# Launchers — talking to the models from the desktop

These five files are the entire user-facing surface of the WPC runtime on M. Szul's
machine. They lived only on `D:\skrypty` and the Windows desktop, in no repository at
all, until 2026-08-25. They are copied here so a disk failure does not take them.

**These are copies, not the live files.** Editing them here changes nothing. The live
paths are listed below; change both, or copy from here to there.

| File | Live location | What it does |
|---|---|---|
| `QWEN START.bat` | Desktop | Opens a chat with the big model |
| `qwen.sh` | `D:\skrypty\qwen.sh` | The shell script the above runs, inside WSL |
| `QWEN AGENT.bat` | Desktop | Big model with hands — calls AIONS tools, asks before each one |
| `QWEN MALY START.bat` | Desktop | Opens a chat with the small model |
| `qwen-maly.sh` | `D:\skrypty\qwen-maly.sh` | The shell script the above runs, inside WSL |

## The two-file pattern

A `.bat` on the desktop is only a doorway. It prefers Windows Terminal when present —
that one handles Polish characters and lets you copy text — and falls back to a plain
console otherwise. All it does is start WSL and hand it a shell script. The script holds
everything that matters: which engine, which weights, which flags.

Change behaviour in the `.sh`. Touch the `.bat` only to point at a different script.

## The two models

|  | Big | Small |
|---|---|---|
| Model | Qwen3-Coder-30B-A3B (MoE) | **Qwen3-4B-Instruct-2507** |
| Weights | `/home/aions/qwen3-coder-wpc4`, 15 GB | `/home/aions/qwen3-4b-wpc4`, 2.0 GB |
| Config + norms + tokenizer | `/home/aions/qwen3-coder-run` (36 MB, slim) | `/mnt/e/models-src/qwen3-4b-it` (7.6 GB, full) |
| Scheme | v4 (4-bit) | v4 (4-bit) |
| Speed | ~45 tokens/min | **~240 tokens/min** |
| Language | Polish is usable | **English only — see below** |

The small model has no slim "run" directory of its own yet. It reads its 1D tensors
straight out of the full 7.6 GB source on E:, which costs nothing noticeable — the load
measured 146–306 ms because the weights are mmapped and only the norms are touched.
Building a slim directory like `qwen3-coder-run` would still be tidier.

## Two things that are not obvious and will bite

**`--chat` is mandatory for the small model.** It is instruction-tuned, so without the
conversation markers it treats the prompt as an unfinished document and writes the next
sentence instead of answering. Measured 2026-08-25: without `--chat` the answer to
"Napisz jedno krotkie zdanie o Rust." was `(Zmieszuj się tylko z podstawowymi
podstawowymi podstawowymi` — a loop. With `--chat`, a real answer.

**Write to the small model in English.** Measured the same day:

- EN: *"The Rust programming language is a modern, memory-safety-focused, and
  performance-optimized language that provides strong compile-time checks to prevent
  errors like nullity and memory leaks."*
- PL: *"Rust nie tylko 保证ą bezpieczeństwie, but także szyła konstrukcje kodu"*

This is **not** the compression's fault, which was the obvious guess and the wrong one.
The 6-bit build (`qwen3-4b-wpc3`, 3.0 GB) produces the same language-mixing and takes
35.8 s to prefill against v4's 12.2 s. A 4B model simply knows little Polish. v4 wins on
speed and ties on quality, so there is no reason to reach for v3 here.

## Muting

The engine narrates: architecture, `loaded layer 1/36` thirty-six times, timings, raw
token ids. All of it goes to **stderr**; only the answer goes to stdout. So `2>/dev/null`
silences everything and leaves the answer. Verified end to end: stderr came back
**0 bytes**.

One more quirk: the engine prints the prompt and then appends its answer to it, so the
scripts strip the echoed prompt with `sed "s|^${PYTANIE}||"`.

## A shell trap that cost two runs

`pkill -f wpc-runtime` **kills its own shell.** The `-f` pattern matches the full command
line of the `bash -lc` process, which contains that very string. Both attempts returned
nothing at all, with no error, because the shell died before running anything. Use
`pkill -x wpc-runtime`, which matches the process name exactly.
