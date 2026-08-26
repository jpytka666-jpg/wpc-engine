# Noworodek — Real Transformer Backprop V1

## Project metadata
- project: Noworodek
- repository: jpytka666-jpg/wpc-engine
- branch: Noworodek
- workstream: transformer-backprop
- experiment: ce-full-decoder-v1
- implementation language: Rust
- parameter storage: external WeightSet via ParameterHandle
- provenance: GPT-5.6 Luna

## Objective
Replace the previous controlled `lm_head` teaching signal with analytic reverse-mode differentiation through the complete mini decoder.

## Backward path
`CE(last-token logits) -> LM head -> final RMSNorm -> decoder layers -> MLP (down/gate/up/SiLU) -> residual -> causal attention (softmax/Q/K/V/O) -> attention RMSNorm -> embedding rows`.

## Success criteria
1. Cross-entropy decreases after repeated steps.
2. At least one internal tensor such as `q_proj.weight` changes.
3. Weight updates are written only through `ParameterHandle` into the external WeightSet.
4. No finite-difference gradient is used by the training step.
5. Later Observatory runs can correlate `ΔW` and influence-map changes.

## TDD provenance
- RED commit: `47815c514f015640ee888ab668dff530147d9d09`
- Implementation: `aef8eed5f7cda4046935c1c57fcbcc1565443aac`
- Review/fix lineage: `5a7b9327e5feb118c78c555a4941815d70c25587`
- Module wiring: `072928b72418090568e2fe6c27b53e9676e2ab77`
- Runner: `d7f2d867f74cfa1a36e68f349b590be2c77afe62`

## Status
Implementation is **PENDING local verification** until the user's machine reports `cargo test` and the real-backprop runner results.
