//! Editable procedural weight representations.
//!
//! The representation is intentionally hybrid: a WeightSet can keep dense,
//! low-rank, pattern and procedural tensors side-by-side. Procedural values
//! are generated from a small parameter vector and a compact instruction set.
//! This is an optimization experiment, not a claim that formulas are always
//! faster than materialized weights.

use crate::{Tensor, WeightSetError};

#[derive(Clone, Debug, PartialEq)]
pub enum WeightRepresentation {
    Dense(Tensor),
    LowRank(LowRankWeight),
    Affine(AffineWeight),
}

#[derive(Clone, Debug, PartialEq)]
pub struct LowRankWeight {
    pub rows: usize,
    pub cols: usize,
    pub rank: usize,
    pub left: Vec<f32>,
    pub right: Vec<f32>,
}

impl LowRankWeight {
    pub fn new(rows: usize, cols: usize, rank: usize, left: Vec<f32>, right: Vec<f32>) -> Result<Self, WeightSetError> {
        if rows == 0 || cols == 0 || rank == 0 {
            return Err(WeightSetError::Backend("low-rank dimensions must be non-zero".into()));
        }
        if left.len() != rows * rank || right.len() != rank * cols {
            return Err(WeightSetError::Backend("low-rank factor shape mismatch".into()));
        }
        Ok(Self { rows, cols, rank, left, right })
    }

    pub fn materialize(&self) -> Tensor {
        let mut values = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                let mut sum = 0.0;
                for r in 0..self.rank {
                    sum += self.left[i * self.rank + r] * self.right[r * self.cols + j];
                }
                values[i * self.cols + j] = sum;
            }
        }
        Tensor::from_vec(vec![self.rows, self.cols], values).expect("validated low-rank shape")
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AffineWeight {
    pub shape: Vec<usize>,
    pub scale: f32,
    pub bias: f32,
    pub pattern: Vec<f32>,
}

impl AffineWeight {
    pub fn new(shape: Vec<usize>, scale: f32, bias: f32, pattern: Vec<f32>) -> Result<Self, WeightSetError> {
        let elements = shape.iter().copied().product::<usize>();
        if elements == 0 || pattern.len() != elements {
            return Err(WeightSetError::Backend("affine pattern shape mismatch".into()));
        }
        Ok(Self { shape, scale, bias, pattern })
    }

    pub fn materialize(&self) -> Tensor {
        let values = self.pattern.iter().map(|x| self.scale * *x + self.bias).collect();
        Tensor::from_vec(self.shape.clone(), values).expect("validated affine pattern shape")
    }

    pub fn edit(&mut self, scale: Option<f32>, bias: Option<f32>) {
        if let Some(value) = scale { self.scale = value; }
        if let Some(value) = bias { self.bias = value; }
    }
}

impl WeightRepresentation {
    pub fn materialize(&self) -> Tensor {
        match self {
            Self::Dense(tensor) => tensor.clone(),
            Self::LowRank(weight) => weight.materialize(),
            Self::Affine(weight) => weight.materialize(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_rank_representation_materializes_correctly() {
        let weight = LowRankWeight::new(2, 2, 1, vec![2.0, 3.0], vec![4.0, 5.0]).unwrap();
        assert_eq!(weight.materialize().values(), &[8.0, 10.0, 12.0, 15.0]);
    }

    #[test]
    fn affine_representation_is_editable_without_rewriting_pattern() {
        let mut weight = AffineWeight::new(vec![3], 2.0, 1.0, vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(weight.materialize().values(), &[3.0, 5.0, 7.0]);
        weight.edit(Some(3.0), Some(-1.0));
        assert_eq!(weight.materialize().values(), &[2.0, 5.0, 8.0]);
    }

    #[test]
    fn representation_is_hybrid_by_design() {
        let dense = WeightRepresentation::Dense(Tensor::from_vec(vec![1, 2], vec![1.0, 2.0]).unwrap());
        let affine = WeightRepresentation::Affine(AffineWeight::new(vec![2], 1.0, 0.0, vec![3.0, 4.0]).unwrap());
        assert_eq!(dense.materialize().values(), &[1.0, 2.0]);
        assert_eq!(affine.materialize().values(), &[3.0, 4.0]);
    }
}
