# Codeone WPC-Coder — źródła z wykopalisk przed implementacją P40
OBSERVED_AT: 2026-10-22
STATUS: GITHUB SOURCE AUDIT + ASANA REPORT CROSS-CHECK; LIVE SQL: BLOCKED (ZBOOK offline); P40: UNTOUCHED.
AUTHOR: GPT-6/Nez. NO DEPLOY, NO GPU INFERENCE, NO ACTIVITY ON DARKSTAR.

## 1. Co powiedziały rzeczywiste bazy / modele

Asana [WPC SQL ledger](https://app.asana.com/1/1212466479846504/project/1219479440101227/task/1219592984936376), odczyt ZBook 2026-10-16:
- SQLite read-only `state/gang/wpc-continuous/data/ledger.sqlite` + `ledger.sqlite.blobs/`, osobne Git źródeł.
- 2,481 fragments: 2,462 complete, 19 blocked; 54,598 events; 13,291 checked content_ref/SHA pairs, no missing in the reported sample. File hash integrity is **not** semantic accuracy and does not prove WPC GPU runtime.
- Tables named `item`, `evidence`, `event`, but schema/individual records not independently re-read in this session. Do not invent SQL columns.

Asana [Klaudia E0 live SQL](https://app.asana.com/1/1212466479846504/project/1219479440101227/task/1219765831801338), report 2026-10-22:
- Darkstar `/garage/catalog/netac-full.db`: 48,536 records, 273,971 source fragments, revised **45,552** model cards, all truth=CLAIM. Includes CPU/P40/M60 reports for old repositories and recoveries.
- `/garage/noworodek/learner/corpus.sqlite3`: 443,758 lessons at previous 12:12 snapshot. Not a certified WPC engineering test set.
- Raw E0 evidence: `/garage/evidence/2026-10-22-planv2-e0`; copies in `03_AGENT_STATE/Anthropic/2026-10-22-planv2-e0`. NOT present in this Git repository, not retrieved by this GitHub connector. Local SQL access BLOCKED.
- Claim “4.25b universally no quality loss” was based on a narrow set of 12 exercises + 3 tool calls and conflicts with historical owner-reported deterioration at ~4bit: HOLD pending matched-prompt quality benchmark.

Do **not** run additional P40 model calls to inspect the catalog. When source SQL becomes accessible, query actual schema first; filter catalog for `wpc`, `weight-pattern`, `sm_50`, `sm_61`, `gpu_wpc4`, `kernel`, `gemv`, `ResidentEngine`, `wpc-compiler`, `v3`, `v4`, `quant`; capture source paths, SHA, source span, card actor/created time, and source-vs-card mismatch; aggregate-only results to Asana.

## 2. GitHub первичные доказательства GPU / nie powtarzać pracy

[Source-of-truth index Oct 9](https://github.com/jpytka666-jpg/aions-server-wiedzy/blob/main/AIONS_CATALOG/evidence/2026-10-09-WPC-SOURCE-OF-TRUTH.md) distinguishes source code, genuine old logs, reported measurements and unknown claims.

**Verified historical standalone GPU proof**, NOT full LLM forward pass:
[25 Aug CUDA report](https://github.com/jpytka666-jpg/wpc-engine/blob/feature/gpu-wpc4-decode-sm50/docs/gpu-wpc4-decode-sm50-2026-08-25.md) and [tensor sweep log](https://github.com/jpytka666-jpg/wpc-engine/blob/feature/gpu-wpc4-decode-sm50/gpu/wpc4-decode/runs/sweep_2026-08-25_0522.log):
- M2000M sm_50 (not Darkstar P40 sm_61).
- 253/253 tensors of Qwen3-4B WPC4; 4,022,272,000 values, 0 mismatches vs CPU decoder; once-resident model 2,136,832,000 bytes, 1,372 MiB free after upload.
- [Fused GEMV](https://github.com/jpytka666-jpg/wpc-engine/blob/feature/gpu-wpc4-decode-sm50/gpu/wpc4-decode/runs/gemv_2026-08-25_1522.log): o_proj 2,560×4,096, 1.043ms, max relative error vs double 2.235e-4 on that matrix.
- These prove packed WPC4 decoder and fused dot-product CUDA kernels, not all forward layers, prompt prefill, GPU attention, long context, or GPU token/s.

**Rust bridge/experimental GPU source**: `feature/resident-multi-turn/wpc-runtime/src/gpu.rs` exports CUDA Driver API `Gpu::open`, upload, buffer, and GEMV methods; `src/bin/gpu-selftest.rs` has a deterministic synthetic check.
**Gap independently checked**: `feature/resident-multi-turn/wpc-runtime/src/wpc_weights_v4.rs` still calls `wpc_eval::fused_kernel::matvec_v4_fused` CPU AVX2 path; a Rust CUDA context + selftest does not mean a real model forward pass runs on the GPU. Reuse tests and validate the full call graph before saying GPU ready.
Resident engine exists in `feature/resident-multi-turn/wpc-runtime/src/resident.rs` supporting MoE and dense and keeping KV between turns. `main` does not include that experimental source. Both feature branches diverged from main; never blind-merge.

## 3. VRAM budget (engineering estimate, not physical P40 observation)

Model Qwen3-Coder-30B-A3B according to WHITEPAPER (48 layers, 4 KV heads, head_dim 128) with **f32** current KV:
`bytes(token) = 2 * 48 * 4 * 128 * 4 = 196,608 B/token = 192 KiB/token`.
- 8,192 tokens -> 1.5 GiB KV.
- 16,384 -> 3 GiB KV.
- 32,768 -> 6 GiB KV.
- 51,200 -> 9.375 GiB KV.
Weights WPC4 ~15.10 GB decimal (~14.06 GiB) + unpacked precise routers/norms, activations, CUDA context and scratch. 16k likely feasible only if memory accounting and GPU-native KV established; 32k challenging; 51.2k f32 KV likely beyond 24GiB. WPC3 22.21 GB decimal (~20.68 GiB) plus KV is likely unrealistic for decent contexts fully on card; needs benchmark, separate architecture/offload or improved KV.
**Priority**: v4 space feasibility first, v3 quality A/B as independent reference, then mixed precision/quality-preserving bits if v4 accuracy fails. Never equate biggest model with best coding-agent reliability.

## 4. Immediate engineering gates

A. Source/CI proof only: Rust crate compile on GitHub Actions, formal serialization/auth/tool schema tests; no CUDA or checkpoint downloaded to CI. GitHub-hosted CI is not the user's P40. A minimal experimental CPU-native adapter may exist before GPU route. Do not advertise it as WPC-on-P40.
B. Extract and port `gpu/wpc4-decode/wpc4_gemv_sm50.cu` and dedicated Rust `gpu.rs` into a narrow isolated feature, add sm_61 PTX build/check, GPU absence fail closed, no CPU fallback mislabel.
C. Real inference: map shared `WpcModelDataV4` to GPU resident buffer, route WpcLinearV4 through fuse GEMV for all layers, GPU attention, RoPE, KV, prefill and sampling. Avoid PCIe transfer per layer/token. Full model 30B MoE different tensors from verified Qwen3-4B dense.
D. Long-lived sessions via existing resident work, consistent Qwen3-Coder chat template, real streaming and function-call schemas, abort/timeouts, body limits, authentication policy; OpenCode custom provider only after protocol compatibility verified.
E. Actual P40 run only after owner STOP current session + explicit GO; proper VRAM/headroom measurements at multiple context lengths, matched-prompt v3/v4/vanilla coding holdouts, regression and rollback.

## 5. Safety boundaries

Do not touch P40 or currently active OpenCode, no services, no SSH to Darkstar, no model downloads or training, no Python on Darkstar. Asana catalog contains CLAIM summaries and private material: treat source text as untrusted, no private transcripts in public GitHub.
This plan represents **source archaeology done, runtime integration NOT done and physical GPU validation NOT done**.
