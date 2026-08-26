# Noworodek — Weight Surgery Mathematics V1

Project: wpc-engine
Branch: Noworodek
Workstream: MATH / WEIGHT-SURGERY
Author: M. Szul via GPT-5.6 Luna
Date: 2026-08-26

## Purpose

Demonstrate that a trained external WeightSet can be edited surgically by changing one addressable parameter, changing model behaviour immediately, and restoring the exact prior checkpoint without retraining.

## Source

Runner: `noworodek/src/bin/noworodek-weight-surgery-math.rs`

Learned checkpoint:
`[-0.00048526586, -0.0000743697, 2.0000975, 2.9999862, 5.0000267, 6.994567]`

Feature order:
`[a, b, a², b², ab, 1]`

Target function:
`2a² + 3b² + 5ab + 7`

## Measured result

Baseline demo MSE: `0.00002974`

Surgical edit:
`ab coefficient ≈ 5.0000267 → 10.0`

Edited demo MSE: `1206.29858398`

The edited predictions changed substantially for all four samples, while all other coefficients remained unchanged.

Restore result:
`exact_checkpoint_match=true`

## What this proves

1. The trained representation is externally addressable.
2. A single semantic coefficient can be edited without retraining the model.
3. The edit changes inference behaviour immediately.
4. Restoring the external WeightSet restores the exact prior parameter vector.

## Scope limitation

This is a controlled linear-feature experiment, not evidence that arbitrary neural-network weights have independently human-readable meanings. It establishes the engineering property of externally addressable and surgically editable parameters.
