# Codeone WPC / Tesla P40 / Qwen3.6-35B-A3B — design and deployment gates
2026-10-22 / GPT-6 / status: PROTOTYPE SOURCE ONLY. No GPU operations and no Darkstar changes.

## Model selection
PRIMARY: Qwen/Qwen3.6-35B-A3B. Model metadata https://huggingface.co/Qwen/Qwen3.6-35B-A3B . Exact architecture https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json . Model is Apache-2.0, 35,951,824,188 parameters, 3B active, type qwen3_5_moe, text_config qwen3_5_moe_text. 40 layers, 30 Gated DeltaNet linear_attention + 10 full_attention, 256 MoE experts (8 routed), 2 KV heads each dimension 256, native context max 262144, separate vision tower. Need text-only WPC compiler and first ensure full tensor mapping, then optionally vision.

COMPARATORS: Qwen3-Coder-30B-A3B-Instruct (existing Rust Qwen3-MoE support), Qwen3.8-27B (powerful agent model but also new hybrid architecture, smaller parameter count). 80B / 120B not fit in one 24GiB GPU with quality-preserving WPC v4 and 65K context: 80B×4.25/8=42.5GB. 80B×2.25/8=22.5GB BEFORE KV/scratch and poor historical quality. Do NOT claim 80B GPU capable here.

## WPC accuracy and memory
Choose WPC v4 4.25bpw as first CAPACITY hypothesis. 35.9518B×4.25/8=19.10 GB decimal theoretical compressed weight lower bound. WPC v3 6.25bpw ~=28.09 GB, does not fit. The v4 file will be larger depending on exact tensor composition, untouched high-precision router, embeddings, norms, shared experts, multimodal tensors and WPC manifest; compare true compiled bytes. WPC v5 2.25b has prior quality failures and is rejected by default.
Target 65536 total context; 8192 max output; 57344 maximum input >=50000. Qwen3.6 10 full attention layers × 2 KV heads × 256 head dim × (K+V) × 2-byte FP16 = 20480 bytes per token; 65536 tokens => 1.25GiB full attention KV. 30 linear-attention layers use Gated DeltaNet constant-size recurrent state which MUST be measured, plus convolution caches. Conservative planning reserves 3GiB runtime and 0.75GiB for precision exceptions on top of calculated compressed weights+KV. Offline Rust codeone-p40-capacity CLI tests fit with >=1GiB budget headroom; this is NOT a measured VRAM number. Queue/model concurrency = 1 until measured.

## Native Rust port gap — not yet implemented
Existing wpc-runtime main can run qwen3_moe but NOT qwen3_5_moe_text. Never silently reinterpret the new model as old Qwen3-MoE. Implement new safetensors/tensor maps, 30 Gated DeltaNet layers with recurrent states, 10 gated GQA attention layers (partial MRoPE, gates), 256 expert router with shared expert and precision policy, output head, full 65536-step cache states, model-native tokenizer and tool parser. Compare every operator to independent trusted reference and deterministic fixtures. Initial codeone-wpc-api is CPU-only Qwen3-MoE test bed and MUST NOT advertise serving this Qwen3.6 model.

Recovered historical CUDA sources: codeone-p40/kernels/wpc4_gemv_sm50.cu from feature/gpu-wpc4-decode-sm50 and wpc-runtime/src/gpu.rs / src/bin/codeone-gpu-selftest.rs from feature/resident-multi-turn. Dynamic CUDA driver bridge is dormant until explicitly opened. Historical 253/253 Qwen3-4B tensors GPU decode plus fused GEMV validated on sm50 M2000M, not full model and not P40. GitHub Actions compiles PTX for sm61 but CANNOT run it as hardware inference. New Rust model must fuse WPC4 dequant into real GPU matvec, keep packed weights, attention, recurrent state and KV device-resident; never label CPU fallback GPU success. CUDA 12.x for sm61; investigate Pascal INT8/DP4A throughput, no Tensor Cores, no accelerated BF16.

## OpenCode ARMORY Environment (Asana task 1219721341528888)
Use existing /gov xterm.js carpet and OpenCode session, never a second orchestration loop. The target is a separate custom OpenAI-compatible provider (not replace existing Ollama) with /v1/models and /v1/chat/completions, real token-by-token SSE, function/tool calls qwen3_coder_xml parser, stop/control/cancellation, per-session cache, bounded 65K and explicit 8K output. A GitHub config template is NOT a deployed config until API capability and owner GO.
Permission contract Government pre-prompt owner approvals; ARMORY global HOST READ ONLY, workspace RW/exec/build/test only inside /garage/armory/workspaces/{model_id}/{job_id} sandbox; reject symlink/path escapes, no root, production writes, secrets/webfetch bypass, Docker host sock or external cloud calls. Preserve Qwen/Ollama state; P40 currently BUSY.

## Independent community evidence vs direct proof
- OpenCode official custom provider @ai-sdk/openai-compatible, baseURL, model limits: https://docs.opencode.ai/docs/providers/
- Reddit local Qwen3.5-35B-A3B OpenCode owner reports good agentic code/tool behavior on RTX3090 131K / ~22GiB: https://www.reddit.com/r/LocalLLaMA/comments/1rdxfdu/qwen3535ba3b_is_a_gamechanger_for_agentic_coding/ . Anecdote, different hardware/engine.
- A direct single Tesla P40 sm61 reference used Qwen3.6-35B-A3B at 128K Q3_K_XL in llama.cpp; reported ~51.6 tokens/s with MTP and ~216 input tokens/s at 78K. This proves compatibility of model family to older hardware in THAT pipeline, not performance of WPC: https://github.com/schmared/tesla-p40-wsl2-cuda-llamacpp ; https://github.com/ggml-org/llama.cpp/discussions/25978 .
- Quality caveat: strong agentic behavior in OpenCode depends on correct tool XML parser, prompt contracts, incremental streaming, context compaction, correct max_tokens and source-grounded tool results; some models look as though tools are ignored if parser is wrong. https://freshcontext.io/benchmarks/models/qwen3.6-35b/
- Existing WPC source-of-truth https://github.com/jpytka666-jpg/aions-server-wiedzy/blob/main/AIONS_CATALOG/evidence/2026-10-09-WPC-SOURCE-OF-TRUTH.md . Reports quality degradation at aggressive compression; v4 does not yet meet measured no-quality-loss goal.

## CI and deployment gate
G0 complete Rust build/test and SM61 PTX static build with sha, consistent config geometry; no CUDA runtime smoke on public GitHub runner.
G1 implement and verify qwen3_5_moe_text CPU reference forward against official weights / fixtures; no private training data.
G2 WPC compile model on separate host/storage only after staging permission, exact tensor manifest and SHA, compare BF16 vs WPC v4 on >=50K context.
G3 CUDA GPU WPC4 SM61 full residency, GPU attention/Gated DeltaNet/recurrent/KV, measured under 24GiB at 65536 context, no false CUDA fallback.
G4 OpenCode E2E tool calls, 50+ test cases and ARMORY sandbox permission/owner approval/abort/stream replay, coding baseline held-out.
G5 OWNER explicitly authorizes P40 window and cutover from current live Qwen; first backup, controlled GPU benchmark, rollback, then option to choose provider.
At present only GitHub-source compilation exists; the critical native Qwen3.6 forward and whole GPU model still need engineering. No Python on Darkstar.
