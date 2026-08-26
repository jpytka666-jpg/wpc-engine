# Noworodek — Real Transformer Weight Surgery V1

## Proven local experiment

Runtime: local Windows/Rust release build on 2026-08-26.

Architecture: `noworodek-decoder-v0`
- vocab: 8
- hidden: 4
- intermediate: 8
- layers: 1
- sequence: `[0, 1, 2]`

Surgical tensor:
`model.layers.00.attention.q_proj.weight`

Edit:
`element[0] += 1.0`

Measured result:
- `changed_logits = 16`
- `max_abs_logit_delta = 0.001879454`
- `restore_exact = true`

Observed logits for the first row were identical before/after. This is expected for causal attention at the first position: its attention domain contains only one key, therefore the softmax weight is exactly 1 and a change to Q does not alter that position's attended V output. Later rows have multiple available keys and therefore respond to the Q perturbation.

## Conclusion

This is a direct proof that a single externally mounted Transformer weight can be edited without retraining and that the edit propagates through the real decoder computation to downstream logits, while an exact checkpoint restoration returns the original computation.

This does **not** establish semantic meaning for arbitrary Transformer weights. It establishes the infrastructure property required for an editable, observable WeightSet: external parameter surgery can causally change model output and can be exactly reversed.

## Provenance

Project: `wpc-engine`
Branch: `Noworodek`
Workstream: `TRANSFORMER-SURGERY`
Observed locally from commit `e8c3a63a41b4270378d2d46242cb3020238d1986`.
Author: M. Szul via GPT-5.6 Luna
