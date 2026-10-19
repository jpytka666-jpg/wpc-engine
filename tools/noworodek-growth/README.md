# Noworodek CBMS — controlled model growth (Rust + CUDA)

Source package for evidence-gated, single-writer capacity growth. Not a claim
of AGI, independent reasoning or unlimited physically feasible growth.
The current approach widens a single decoder layer's MLP while preserving
its previous function at the moment of expansion.

## Layout
- `src/main.rs` and `src/kernels.cu`: CUDA trainer, Tesla M60 compatible.
- `src/bin/noworodek-grow.rs`: NWRD-v1 weight-widening tool.
- `tools/auto-growth.mjs`: optional between-cycle controller (Node 24).
- `tools/promote-inter512-20261019.sh`: historical local deployment example.
  Do not execute on live machines before independent review and adaptation.
- `docs/`: technical reports; private machine identifiers redacted.

This public package intentionally omits all corpus and weight files.
Before use, set `AIONS_NOWORODEK_GPU_UUID`, optionally
`AIONS_NOWORODEK_SSH_ALIAS`, and review path/storage configuration.
Do not commit secrets.

## Verified 19 October 2026
- 15,088,000 -> 15,186,304 f32 params through MLP width 256 -> 512.
- Rust: 4 unit tests passed. First faulty tensor mapping rejected on GPU,
  then corrected and parity-tested.
- At deployment, GPU parity held: 20.4153 / 20.4153 and 21.4349 / 21.4349.
- Same technical training input, batch=256, lr=.005:
  old 19.5058 -> 12.7160; grown 19.5058 -> 12.7155.
- Three full production training cycles accepted after first widening.
- Auto-growth code activated at 17:07 BST with 24h cooldown, 6 spaced
  fixed-benchmark observations, CPU/GPU/disk checks, GPU parity and A/B gates.
- A later automatically triggered second expansion had NOT yet been observed
  when this record was prepared. Next-token loss is NOT factual-reasoning QA.

Tests (from repository root):
`cargo test --manifest-path tools/noworodek-growth/Cargo.toml --bin noworodek-grow`
`node tools/noworodek-growth/tools/auto-growth.mjs --self-test`

Project license and contribution policies follow the parent wpc-engine.
