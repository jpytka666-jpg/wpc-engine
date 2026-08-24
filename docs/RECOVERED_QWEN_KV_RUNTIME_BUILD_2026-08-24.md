# Exact Qwen KV Runtime Build — `cd6923ba` — 2026-08-24

## Status

A fresh release build was produced from the exact Qwen3-MoE KV probe commit:

`cd6923bafe1240bcd95b1e80b91382985c009a4a`

GitHub commit title: `chore(memory-kv): scope probe explicitly to Qwen3-MoE [2026-08-24]`.

This is the **actual runtime build containing the Qwen3-MoE read-only KV probe**. The probe is compiled into the `wpc-runtime` binary; it is not a separate Cargo binary.

## Local build location

Fresh build-only checkout created on 2026-08-24:

`C:\\temp\\aions-qwen-kv-probe-rebuild-2026-08-24`

The checkout HEAD was verified as:

`cd6923bafe1240bcd95b1e80b91382985c009a4a`

Runtime executable:

`C:\\temp\\aions-qwen-kv-probe-rebuild-2026-08-24\\target\\release\\wpc-runtime`

WSL-equivalent path:

`/mnt/c/temp/aions-qwen-kv-probe-rebuild-2026-08-24/target/release/wpc-runtime`

## Build identity

| Field | Value |
|---|---|
| Commit | `cd6923bafe1240bcd95b1e80b91382985c009a4a` |
| Binary | `target/release/wpc-runtime` |
| Format | ELF 64-bit LSB PIE, x86-64, Linux |
| Size | `6,114,040` bytes |
| SHA-256 | `cf7c9a2085394c14f9b0e36c9d7af9bfdbd5432c767e530c92f61d0ecfb4680a` |
| Modified | `2026-08-24 19:47:53 +0100` |
| Build ID | `822ba223e9a075fef0fa6bc1cbafc0e958f9eda1` |

## Release outputs

The same fresh checkout produced these release outputs:

- `target/release/wpc-runtime` — 6,114,040 B
- `target/release/libwpc_runtime.rlib` — 6,986,250 B
- `target/release/wpc-resident` — 5,712,200 B
- `target/release/aions-agent` — 5,855,664 B

## Probe implementation provenance

The build dependency manifest explicitly includes:

- `wpc-runtime/src/kv_probe.rs`
- `wpc-runtime/src/qwen3_moe_model.rs`
- `wpc-runtime/src/qwen3_model.rs`

`cd6923ba` changes `wpc-runtime/src/kv_probe.rs` to scope the observation contract to Qwen3-MoE / Qwen3-Coder-30B-A3B.

The probe is activated at runtime with:

`AIONS_KV_PROBE=1`

The Qwen3-MoE cache installs `StatsKvProbe` when that variable is enabled. The probe is read-only: it counts KV observations and accumulates exact raw f32 byte volume, then emits final statistics to stderr when the probe is dropped.

## Important naming clarification

There is **no separate Cargo target named `wpc-runtime-qwen-kv-probe`** in this commit.

The earlier temporary name:

`wpc-runtime-qwen-kv-probe-2026-08-24`

referred to the runtime experiment/workspace naming, not to a separate executable target.

The canonical executable for this commit is:

`target/release/wpc-runtime`

with `AIONS_KV_PROBE=1` enabled for the KV experiment.

## Safety / provenance rules

- This build was created only in the new dated checkout above.
- Older workspaces were not reset, cleaned, overwritten, moved, or modified.
- The build used a checkout pinned to the exact commit listed above.
- The local binary was verified as ELF and hashed before any runtime experiment.
- No model files are embedded or modified by the probe.

## Next experiment

Use the exact binary above with the external Qwen3-Coder-30B-A3B model inputs and run the KV probe with `AIONS_KV_PROBE=1`.

The benchmark sequence should be recorded as separate runs (for example 8, 24, 40, 64, 128, 256 generated tokens) and the resulting KV statistics should be appended to GitHub as benchmark evidence.