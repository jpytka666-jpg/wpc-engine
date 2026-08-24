# Qwen3-Coder-30B-A3B WPC v4 — KV Probe Experiment

<!--
Created: 2026-08-24
Created by: ChatGPT
Reason: Preserve the verified real-model Qwen 30B WPC v4 inference/KV-probe results and define the repeatable experiment for Codex and Claude.
This document is new work created on 2026-08-24; it does not overwrite an existing document.
-->

## Scope

Target is **Qwen3-Coder-30B-A3B**, running through the Rust `qwen3_moe` runtime with **WPC v4 4-bit packed weights**.

Model inputs used by the verified run:

- Model/norm source: `/home/aions/qwen3-coder-run`
- WPC v4 weights: `/home/aions/qwen3-coder-wpc4`
- Probe binary: `wpc-runtime-qwen-kv-probe-2026-08-24`
- Probe enabled with `AIONS_KV_PROBE=1`
- Probe sampling: `AIONS_KV_PROBE_STRIDE=64`

These model/WPC directories are treated as external read-only inputs. They were not modified by this experiment.

## Verified Run — 2026-08-24

Prompt: `Write one short sentence about Rust.`

Generation: 8 tokens.

Output:

> Rust is a systems programming language

This is semantically correct and directly answers the prompt.

Runtime:

- Architecture: Qwen3Moe
- Hidden size: 2048
- Layers: 48
- Attention heads: 32
- KV heads: 4
- Head dimension: 128
- Experts: 128
- Active experts: 8
- WPC scheme: v4, 4-bit packed

Timing:

- Model load: 227.43396 ms
- Prefill: 9.508580886 s for 8 tokens
- Generation: 10.673457356 s for 8 tokens
- Generation rate: ~44.89 tokens/minute
- Output text rate: ~33.72 words/minute for the 6-word answer

## Verified KV statistics

Probe output:

```text
kv-probe: calls=3072 positions=16 raw_kv_bytes=3145728 sampled_values=49152 sample_min=-27.059315 sample_max=29.051640 sample_mean=-0.001050 sample_rms=1.332484 stride=64
```

The `calls` count matches the runtime geometry exactly:

`48 layers × 4 KV heads × 16 positions = 3072 calls`

Raw resident KV measured at 16 positions:

- `3,145,728` bytes = `3 MiB`
- `192 KiB` per resident token at this FP32 representation

Important: this is the measured **raw resident KV representation**, separate from the WPC-compressed model weights.

## Warm-up / repeated-run status

Three completed fresh-binary runs were requested to measure warm-up, generation quality, and KV growth. The first verified fresh-binary run above completed successfully. Subsequent synchronous WSL launches were blocked by the Desktop Commander transport timeout before returning reliable output.

Therefore **no invented warm-up averages or second/third-run statistics are recorded here**.

The next repeat should use the same binary, same model paths, same prompt, and 8/16/24/40 token caps as practical. Capture for every run:

1. prefill time;
2. generation time;
3. generated token count;
4. tokens/minute;
5. output text and semantic correctness;
6. `kv-probe` calls, positions, raw bytes, sampled values and ranges.

## What this proves

1. The WPC v4 compressed Qwen3-Coder-30B-A3B model performs real inference in the Rust runtime.
2. The resident K/V cache is reached by the probe at the actual append boundary.
3. The measured K/V volume grows with resident sequence length and is an independent memory budget from the compressed model weights.
4. The current probe can observe exact raw K/V byte volume while sampling value statistics at a configurable stride.

## Guidance for Codex / Claude

Do not replace the Qwen runtime with Gemma for this experiment.

Do not replace the WPC v4 model with an uncompressed full-weight model.

Do not overwrite the old runtime binary used by previous experiments. Build fresh binaries under a new timestamped name/path.

Do not modify `/home/aions/qwen3-coder-run` or `/home/aions/qwen3-coder-wpc4`.

The next experiment is a measurement pass, not a code rewrite. Use the existing Qwen3-MoE probe and collect repeated real-model runs.

For KV growth, compare positions such as 8, 16, 32, 64, 128, and 256 where runtime time permits. Check that `raw_kv_bytes / positions` remains consistent with the expected resident KV geometry.

For generation quality, preserve the exact generated text from every run before doing any optimization.

## GitHub state at experiment time

PR #27: `test(memory-kv): Qwen 30B resident KV probe contract`

Head branch: `feature/memory-kv-real-model-bridge`

Head commit at experiment start: `cd6923bafe1240bcd95b1e80b91382985c009a4a`

The PR remained Draft/open during this experiment. No CI run had yet been observed for this exact HEAD when this document was created.
