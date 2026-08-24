# Recovered WPC Runtime Build — 2026-08-24

## Status

A complete local Rust release build of `wpc-runtime` was recovered from:

`C:\\temp\\aions-wpc-check2\\target\\release\\wpc-runtime.exe`

This is **not yet identified as the Qwen KV probe executable**. It is recorded here because it is a valuable preserved WPC Runtime build and may provide the closest reproducible runtime baseline.

## Artifact identity

| Field | Value |
|---|---|
| Artifact | `wpc-runtime.exe` |
| Size | `4,949,504` bytes |
| SHA-256 | `b28c8aca492dee63b7b2355d2db757628252e33eba6f3e47ff2dc0e377615165` |
| Created | `2026-08-23 09:31:35.983Z` |
| Modified | `2026-08-23 09:31:36.467Z` |
| PDB | `wpc_runtime.pdb` (`2,609,152` bytes) |
| Duplicate | `target/release/deps/wpc_runtime.exe` (same size/timestamp) |

## Source/build provenance

The preserved workspace is:

`C:\\temp\\aions-wpc-check2`

Its Git checkout is on:

`integration/aions-unified-stack`

with local HEAD:

`e5021b3a2f5453d6ad51089f7ec76f06d824cc50`

Commit message:

`chore: keep CI formatting check strict`

GitHub confirms commit `e5021b3a2f5453d6ad51089f7ec76f06d824cc50` exists in `jpytka666-jpg/wpc-engine`.

## Static contents check

The recovered PE/Rust binary contains Qwen runtime source-path strings including:

- `wpc-runtime\\src\\qwen3_model.rs`
- `wpc-runtime\\src\\qwen3_moe_model.rs`

It also contains `Qwen3-MoE` strings.

The following probe-specific strings were **not** found in the executable's ASCII string table:

- `kv_probe`
- `AIONS_KV_PROBE`
- `wpc-runtime-qwen-kv-probe`
- `cd6923ba`
- `24172701`
- `raw KV`
- `192 KiB`

Therefore this artifact should currently be treated as a **preserved Qwen-capable WPC Runtime baseline**, not as the recovered KV-probe executable.

## Why this matters

The repository README documents Qwen3-Coder-30B-A3B as a supported Qwen3-MoE runtime target. PR #27 adds an opt-in KV statistics probe to the newer branch. This recovered build predates that probe commit and gives us a concrete, hashed runtime baseline for comparison.

## Safety / handling

The binary has not been executed during recovery. The local file has not been moved, copied, overwritten, or modified.

Next forensic step: compare this build against the exact source/build introduced by `cd6923bafe1240bcd95b1e80b91382985c009a4a` and determine whether a matching probe binary can be recovered from any remaining local cache.