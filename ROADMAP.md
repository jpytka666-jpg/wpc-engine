# WPC Engine / AIONS Roadmap

## Track A — Core WPC runtime
- [x] WPC format/compiler foundations
- [x] Batch attention / GEMM runtime foundation
- [ ] Complete resident AIONS runtime integration
- [ ] Full CI gate green
- [ ] Merge only after integration and runtime verification

## Track B — AIONS Local CI / Coding Agent
See `tools/aions-local-ci/ROADMAP.md`.

Goal: a controlled local coding agent that can inspect, implement, test, diagnose and repair code, while treating independent verification as the source of truth.

## Track C — KV Cache / CBMS
See `docs/kv-cache/ROADMAP.md`.

Goal: build a tiered KV architecture with hot KV in fast memory and optional compressed/persistent representations outside the token-critical path.

## Track D — WPC-compressed KV research
- [ ] Define KV block representation
- [ ] Build lossless baseline for storage/replay
- [ ] Prototype approximate compression
- [ ] Measure compression ratio
- [ ] Measure reconstruction latency
- [ ] Measure attention/output quality impact
- [ ] Compare against uncompressed KV
- [ ] Only then consider runtime integration

## Architecture principle
Do not couple experimental KV compression to the production resident runtime until the standalone benchmark proves that storage savings and reconstruction cost are useful.
