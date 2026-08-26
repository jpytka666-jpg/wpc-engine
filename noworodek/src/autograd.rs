//! Minimal reverse-mode primitives for externally owned parameters.
//!
//! This is intentionally small and deterministic: the model graph stays
//! separate from WeightSet storage, while gradients are accumulated by stable
//! parameter name. GPU kernels can replace the primitive math later without
//! changing the external-weight contract.

use std::collections::HashMap;

use crate::weightset::WeightSetError;
use crate::Tensor;

#[derive(Clone, Debug, Default)]
pub struct Gradients {
    values: HashMap<String, Tensor>,
}

impl Gradients {
    pub fn new() -> Self { Self::default() }

    pub fn accumulate(&mut self, name: impl Into<String>, grad: Tensor) -> Result<(), WeightSetError> {
        let name = name.into();
        if let Some(existing) = self.values.get_mut(&name) {
            if existing.shape() != grad.shape() {
                return Err(WeightSetError::Backend(format!("gradient shape mismatch for {name}")));
            }
            for (lhs, rhs) in existing.values_mut().iter_mut().zip(grad.values()) {
                *lhs += rhs;
            }
        } else {
            self.values.insert(name, grad);
        }
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> { self.values.get(name) }
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Tensor)> { self.values.iter().map(|(k, v)| (k.as_str(), v)) }
}

#[derive(Clone, Debug)]
pub struct LinearCache {
    pub input: Tensor,
    pub weight: Tensor,
}

/// Backward for Y = XW where X=[m,k], W=[k,n].
/// Returns dX and dW for an upstream gradient dY=[m,n].
pub fn linear_backward(cache: &LinearCache, grad_output: &Tensor) -> Result<(Tensor, Tensor), WeightSetError> {
    if grad_output.shape().len() != 2 || cache.input.shape().len() != 2 || cache.weight.shape().len() != 2 {
        return Err(WeightSetError::Backend("linear backward requires rank-2 tensors".into()));
    }
    let (m, k) = (cache.input.shape()[0], cache.input.shape()[1]);
    let (wk, n) = (cache.weight.shape()[0], cache.weight.shape()[1]);
    if wk != k || grad_output.shape() != [m, n] {
        return Err(WeightSetError::Backend("linear backward shape mismatch".into()));
    }
    let mut dx = vec![0.0; m * k];
    for i in 0..m {
        for p in 0..k {
            let mut sum = 0.0;
            for j in 0..n { sum += grad_output.values()[i * n + j] * cache.weight.values()[p * n + j]; }
            dx[i * k + p] = sum;
        }
    }
    let mut dw = vec![0.0; k * n];
    for p in 0..k {
        for j in 0..n {
            let mut sum = 0.0;
            for i in 0..m { sum += cache.input.values()[i * k + p] * grad_output.values()[i * n + j]; }
            dw[p * n + j] = sum;
        }
    }
    Ok((Tensor::from_vec(vec![m, k], dx)?, Tensor::from_vec(vec![k, n], dw)?))
}

/// Gradient of mean squared error: mean((prediction-target)^2).
pub fn mse_loss(prediction: &Tensor, target: &Tensor) -> Result<(f32, Tensor), WeightSetError> {
    if prediction.shape() != target.shape() {
        return Err(WeightSetError::Backend("mse shape mismatch".into()));
    }
    if prediction.values().is_empty() {
        return Err(WeightSetError::Backend("mse requires non-empty tensor".into()));
    }
    let count = prediction.values().len() as f32;
    let mut loss = 0.0;
    let mut grad = Vec::with_capacity(prediction.values().len());
    for (p, t) in prediction.values().iter().zip(target.values()) {
        let d = p - t;
        loss += d * d;
        grad.push(2.0 * d / count);
    }
    Ok((loss / count, Tensor::from_vec(prediction.shape().to_vec(), grad)?))
}

#[derive(Clone, Copy, Debug)]
pub struct Sgd {
    pub learning_rate: f32,
}

impl Sgd {
    pub fn step(&self, parameter: &mut Tensor, gradient: &Tensor) -> Result<(), WeightSetError> {
        if parameter.shape() != gradient.shape() {
            return Err(WeightSetError::Backend("optimizer shape mismatch".into()));
        }
        for (weight, grad) in parameter.values_mut().iter_mut().zip(gradient.values()) {
            *weight -= self.learning_rate * grad;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_backward_matches_small_hand_calculation() {
        let x = Tensor::from_vec(vec![1, 2], vec![2.0, 3.0]).unwrap();
        let w = Tensor::from_vec(vec![2, 1], vec![4.0, 5.0]).unwrap();
        let dy = Tensor::from_vec(vec![1, 1], vec![7.0]).unwrap();
        let (dx, dw) = linear_backward(&LinearCache { input: x, weight: w }, &dy).unwrap();
        assert_eq!(dx.values(), &[28.0, 35.0]);
        assert_eq!(dw.values(), &[14.0, 21.0]);
    }

    #[test]
    fn mse_returns_loss_and_gradient() {
        let p = Tensor::from_vec(vec![2], vec![2.0, 4.0]).unwrap();
        let t = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();
        let (loss, grad) = mse_loss(&p, &t).unwrap();
        assert!((loss - 2.5).abs() < 1e-6);
        assert_eq!(grad.values(), &[1.0, 2.0]);
    }

    #[test]
    fn sgd_updates_parameter_values() {
        let mut parameter = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();
        let grad = Tensor::from_vec(vec![2], vec![0.5, -1.0]).unwrap();
        Sgd { learning_rate: 0.1 }.step(&mut parameter, &grad).unwrap();
        assert_eq!(parameter.values(), &[0.95, 2.1]);
    }
}