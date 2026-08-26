# Noworodek Nonlinear Math Observation Experiment

Project: wpc-engine  
Branch: Noworodek  
Workstream: MATH

## Objective

Test whether an externally owned, editable WeightSet can learn a nonlinear mathematical relation while Observatory records the evolution of its parameters.

## Target function

`y = 2a^2 + 3b^2 + 5ab + 7`

## Input basis

The learner receives six explicit features:

`[a, b, a^2, b^2, ab, 1]`

The trainable parameter vector therefore has six externally stored values. The expected solution is:

`[0, 0, 2, 3, 5, 7]`

This deliberately separates the nonlinear feature construction from the trainable parameter storage, making the learned coefficient pattern directly inspectable.

## Observation contract

Every training sample is assigned an `ExperienceId`. The `ObservedLinearTrainer` records the loss and tensor delta statistics for each update, including changed element count, L1 delta, L2 delta and maximum absolute delta.

## Evaluation

The runner reports held-out MSE before and after training, the learned six coefficients, and the total number of observations. The experiment is considered successful only when held-out MSE improves; coefficient convergence is reported separately and is not assumed.

## Next comparison

After this reference experiment, compare the learned coefficient representation against Low-Rank and WPC representations using storage size, reconstruction error, execution latency and task quality.
