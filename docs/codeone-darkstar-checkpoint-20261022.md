<!-- AUTHOR: GPT-6 / Codeone architect; MODEL: GPT-6 -->
<!-- TIMESTAMP: 2026-10-22; REVISION: 1 (GitHub checkpoint) -->
<!-- PURPOSE: Propagate the latest owner-authorized Darkstar staging and CPU master evidence from Asana without declaring untested WPC GPU inference. -->
<!-- SYSTEM: Codeone CPU master + WPC P40 worker; MECHANICS: evidence ledger and deployment gates; DEPENDENCIES: existing ARMORY/OpenCode/Government. -->
<!-- PATH: docs/codeone-darkstar-checkpoint-20261022.md; SOURCE: Asana 1219796979054899; EVIDENCE: 1219802547416772, 1219802935369417, 1219803062886638. -->

# Darkstar Codeone checkpoint — 22 October 2026

**Scope:** reporting the current owner-authorized, **Darkstar-only** staging and operational evidence, as reported by Codex/Nez in Asana. The GitHub author did **not** independently log into Darkstar, examine the 148 GB of model shards, rerun their hashes, or benchmark either physical GPU or CPU in this checkpoint. This document does not imply a passing WPC4 runtime or an implemented Qwen3.6 GPU forward pass.

## Current state versus original feature-branch preflight

| Component | Reported and bounded status | Not yet proved |
|---|---|---|
| SSD storage | `/rpool/armory-workspaces/codeone/` on verified ZFS SSD mirror. **`/garage` is on rotational HDD** and must not be treated as equivalent SSD staging. | Current live free disk after download, independent read of filesystem/ACL |
| Source download | **Complete** at 2026-10-22 20:48:08Z: 40 selected files, **148,453,850,869 bytes**, background unit exit 0. During transfer individual sizes/checksums were recorded. | Independent second pass of all 148 GB of hashes; continuous integrity |
| CPU master | Qwen3.5-122B-A10B Q4_K_M on existing Darkstar CPU Ollama/OpenCode, replacing active GPT-OSS 120B following **later explicit owner GO**. | Sustained 250K actual input with output reservation; broad quality parity with clouds |
| CPU tools | Real OpenCode tool calls for shell, Asana create/update/comment, websearch/webfetch, documented test. | Large held-out coding benchmark, persistent context fidelity |
| CPU performance | **4.36–4.92 generated tokens/s** in warm prompts, 20 compute threads; NUMA/affinity trial not proven >5% gain. | Predictable long-context prefill/TTFT and long-horizon tool reliability |
| CPU context | Server/OpenCode configured for **262,144 total tokens**. | At least 250,000 actual usable incoming tokens incl system/tools/output reserve |
| CPU → P40 delegation | Real `p40_delegate` → Asana/SSH relay → P40 OpenCode → `p40_result` → CPU, an exact sentinel returned, **8/8 bridge tests PASS**. | Correctness/safety of arbitrary generated code; WPC backend use |
| P40 incumbent | Existing P40 Qwen3-Coder/Ollama/OpenCode remains serving. | New WPC model **not** loaded, tested or substituted |
| P40 next-generation model | Original Qwen3.6-35B-A3B checkpoint **downloaded** in pinned safetensors shards. | Complete WPCv4 compile, full tensor coverage, correct native Rust Qwen3.6 forward, VRAM and 65K tests |
| M60A/M60B | Learner/trainer and serving reported active, not modified in this work. | Their learning quality, full end-to-end live evaluation |

### Immutable source pins from the Codex report

**CPU Master (live via Ollama; not WPC):**

- Official quant source: `unsloth/Qwen3.5-122B-A10B-GGUF`.
- Pinned snapshot revision: `51eab4d59d53f573fb9206cb3ce613f1d0aa392b`.
- Variant: `Q4_K_M`; **3 GGUF shards**. Selection inventory contains **5 files / 76,537,063,075 bytes** including metadata. An earlier staging report of 76,536,964,608 bytes describes the three GGUF shards alone; do not confuse two totals.
- Imported into existing Ollama 0.34.2 with same-filesystem hardlinks to avoid duplicating model bytes. CPU-only serving and OpenCode 1.19.3 used in the reported successful tool smoke tests. Old GPT-OSS data retained for rollback.
- The official llama.cpp CPU-only build was also reported as passing and **staged** on SSD; it is **not** the serving implementation. Do not describe production as using `llama-server` or our Rust WPC runtime.

**P40 Worker candidate (downloaded, not running):**

- Official source: `Qwen/Qwen3.6-35B-A3B`.
- Pinned snapshot revision: `995ad96eacd98c81ed38be0c5b274b04031597b0`.
- **26 original safetensors shards**, selection of **35 files / 71,916,787,794 bytes** including metadata.
- Model-native text architecture: `qwen3_5_moe_text`, not the existing Rust runtime's `qwen3_moe`. Qwen3.6 uses hybrid Gated DeltaNet plus full attention and stacked rank-3 expert tensors. Merely downloading the shard set is **not** compression or inference.

Reported total: **76,537,063,075 + 71,916,787,794 = 148,453,850,869 bytes**, 5 + 35 = 40 files.

## CPU → P40 delegation boundary (verified for transport only)

Reported real route:

```text
CPU Qwen3.5 / OpenCode
    -> existing bounded p40_delegate tool and SSH/Asana relay
    -> existing P40 Qwen3-Coder / OpenCode (incumbent)
    -> child result with request/response correlation
    -> p40_result tool
    -> CPU master independent evaluation
```

The returned exact probe string was `P40_CHILD_BRIDGE_OK_20261022` (Asana relay request/comment `1219802769244108`), with a matching completion state. Eight focused bridge regressions passed. **This proves message transport/correlation, not code quality or P40 WPC execution.** Existing OpenCode+ARMORY/Government remain the agent execution, authorization and logging infrastructure; do not build a second uncontrolled agent execution loop.

The owner requires: global host **read-only** operations, writable/executable paths only in the bounded ARMORY workspaces, explicit approval when required, no unrelated service changes, no Python on Darkstar, and exactly five provenance-header lines in newly authored source where the format permits. For JSON/binaries, use provenance sidecars instead of invalidating file formats.

## GitHub WPC branch scope — work still outstanding

At the original checkpoint, [draft PR #36](https://github.com/jpytka666-jpg/wpc-engine/pull/36) included:

1. Rust `codeone-p40-capacity` arithmetic preflight for 65,536 total context / 8,192 maximum output and theoretical WPC4 lower-bound VRAM.
2. WPC converter change allowing **allowlisted stacked rank-3 MoE expert matrices**, with synthetic safetensors→WPC4 integration tests.
3. Recovered historical fused WPC4 CUDA GEMV kernel, compiled **offline** for SM61 in GitHub Actions; an unexecuted Rust CUDA Driver bridge.
4. CPU prototype OpenAI-compatible adapter for the **older** `qwen3_moe` architecture; not an actual Qwen3.6 inference engine, and its buffered SSE/tool flow is not the required serving contract.

**Next implementation gates, in order:**

- **D4 — source/tests on Darkstar only:** review pinned PR SHA, Cargo build/compiler integration tests, read full input tensor/shape manifest, verify 3D expert handling and uncovered tensors. Never mark unknown/uncompressed essential tensors as included.
- **D5 — CPU-only WPC4 compile:** perform in bounded SSD workspace without touching GPU, match all required Qwen3.6 text tensors against WPC4 payload or deliberate higher-precision residuals, and report exact output byte totals, source pins, SHA256 and quality/decode errors. If coverage incomplete, STOP and report.
- **D6 — Rust hybrid forward:** implement all Gated DeltaNet/state/conv paths, full gated-attention/KV paths, routed+shared MoE, tokenizer/template, fully resident model and actual SSE+tool-call API; independent reference parity tests. Do **not** reuse the old `qwen3_moe` forward.
- **D7 — physical Tesla P40 SM61 GPU tests:** only after separate **owner GPU GO**. Explicit interruption plan and rollback for current Qwen, observe VRAM/50K+ actual input/65K total context, timing, CUDA kernels, tool calls and cancellation.
- **D8 — OpenCode provider:** only after native Rust WPC backend passes the real tests. Keep the incumbent Ollama provider as rollback; use existing Government/ARMORY owner gate, do not bypass host RO/workspace RW controls.

No local model or compiler artifact should be copied to Fedora for execution. Avoid force pushes, unreviewed production changes, or unverified claims such as “35B WPC fully deployed”.

## Primary Asana evidence links

- [Master plan and source checkpoint](https://app.asana.com/1/1212466479846504/project/1219479440101227/task/1219796979054899) — checkpoint at 21:38 BST, downloader complete, dual-model roles, full chronology.
- [GPU WPC design and audit task](https://app.asana.com/1/1212466479846504/project/1219479440101227/task/1219785093178794).
- [ARMORY workspace/permissions](https://app.asana.com/1/1212466479846504/project/1219479440101227/task/1219721341528888).
- Asana comment IDs: `1219809279043684` (operational CPU), `1219801937477936` (tool/context), `1219802547416772` (complete source download and owner rules), `1219802935369417` (live delegation), `1219803062886638` (master checkpoint).
