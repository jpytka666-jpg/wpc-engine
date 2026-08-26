# Noworodek Observatory — Train → Map → Train → Map Diff V1

## Metadata
- Project: WPC-ENGINE
- Branch: `Noworodek`
- Workstream: Noworodek / Observatory
- Experiment: `math.next_token.v3`
- Implementation: `noworodek/src/bin/noworodek-train-map-diff.rs`
- Core API: `noworodek/src/observatory.rs`
- TDD RED commit: `1855f3a5ad45010464d5b458921467c4289d34d4`
- TDD GREEN core commit: `7f59164897211dc442fb3e7ae31dc6e443556f53`
- Public export commit: `653b8aaa02d9dd4466f783e6425f21e4a587abcb`
- Runner commit: `1df264b6d96be9c1adf7e841f6847a31b1488c01`
- Author attribution: M. Szul via GPT-5.6 Luna / GitHub workflow

## Purpose
Compare tensor-level influence maps before and after a controlled training interval, while preserving the external `WeightSet` model contract.

## Protocol
1. Build deterministic one-layer decoder fixture.
2. Capture baseline loss and directional influence map `T0`.
3. Apply 200 deterministic target-directed `lm_head` updates.
4. Capture post-training loss and influence map `T1`.
5. Compute `T1 - T0` per tensor.

## Interpretation boundary
This V1 runner uses a **controlled teaching signal** on the external `lm_head`; it is not full end-to-end backpropagation through the Transformer. The experiment is therefore a probe of the Observatory data path and influence-map comparison, not a claim of general Transformer training competence.

## Expected outcome
The runner must compile, execute deterministically, report before/after loss, and produce a non-empty tensor influence diff with `RESULT map_diff_observed=true`.
