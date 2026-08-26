//! Low-rank representation primitives for the first alternative WeightSet backend.
//!
//! This module deliberately stays representation-level: factor storage and
//! metrics are defined here, while backend-specific GPU execution is a later
//! stage. The first experiment compares real tensors against the proven WPC
//! fused baseline before promoting low-rank to a default representation.

#[derive(Clone, Debug, PartialEq)]
pub struct LowRankMatrix {
    rows: usize,
    cols: usize,
    rank: usize,
    left: Vec<f32>,
    right: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RankMetrics {
    pub rank: usize,
    pub dense_elements: usize,
    pub factor_elements: usize,
    pub dense_bytes_f32: usize,
    pub factor_bytes_f32: usize,
    pub compression_ratio: f32,
    pub relative_frobenius_error: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LowRankError {
    InvalidShape,
    InvalidRank,
    DataLengthMismatch,
}

impl LowRankMatrix {
    pub fn new(rows: usize, cols: usize, rank: usize, left: Vec<f32>, right: Vec<f32>) -> Result<Self, LowRankError> {
        if rows == 0 || cols == 0 {
            return Err(LowRankError::InvalidShape);
        }
        if rank == 0 || rank > rows.min(cols) {
            return Err(LowRankError::InvalidRank);
        }
        if left.len() != rows * rank || right.len() != rank * cols {
            return Err(LowRankError::DataLengthMismatch);
        }
        Ok(Self { rows, cols, rank, left, right })
    }

    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }
    pub fn rank(&self) -> usize { self.rank }
    pub fn left(&self) -> &[f32] { &self.left }
    pub fn right(&self) -> &[f32] { &self.right }

    /// Compute y = A * (B * x) without materializing A*B.
    pub fn execute_vector(&self, x: &[f32]) -> Result<Vec<f32>, LowRankError> {
        if x.len() != self.cols {
            return Err(LowRankError::DataLengthMismatch);
        }
        let mut middle = vec![0.0; self.rank];
        for r in 0..self.rank {
            let mut acc = 0.0;
            for c in 0..self.cols {
                acc += self.right[r * self.cols + c] * x[c];
            }
            middle[r] = acc;
        }

        let mut output = vec![0.0; self.rows];
        for row in 0..self.rows {
            let mut acc = 0.0;
            for r in 0..self.rank {
                acc += self.left[row * self.rank + r] * middle[r];
            }
            output[row] = acc;
        }
        Ok(output)
    }

    pub fn materialize(&self) -> Vec<f32> {
        let mut dense = vec![0.0; self.rows * self.cols];
        for row in 0..self.rows {
            for col in 0..self.cols {
                let mut acc = 0.0;
                for r in 0..self.rank {
                    acc += self.left[row * self.rank + r] * self.right[r * self.cols + col];
                }
                dense[row * self.cols + col] = acc;
            }
        }
        dense
    }
}

pub fn rank_metrics(
    dense_rows: usize,
    dense_cols: usize,
    rank: usize,
    reconstruction: &[f32],
    reference: &[f32],
) -> Result<RankMetrics, LowRankError> {
    if dense_rows == 0 || dense_cols == 0 {
        return Err(LowRankError::InvalidShape);
    }
    if rank == 0 || rank > dense_rows.min(dense_cols) {
        return Err(LowRankError::InvalidRank);
    }
    if reconstruction.len() != dense_rows * dense_cols || reference.len() != reconstruction.len() {
        return Err(LowRankError::DataLengthMismatch);
    }

    let mut error_sq = 0.0f64;
    let mut reference_sq = 0.0f64;
    for (&approx, &actual) in reconstruction.iter().zip(reference.iter()) {
        let diff = f64::from(approx) - f64::from(actual);
        error_sq += diff * diff;
        reference_sq += f64::from(actual) * f64::from(actual);
    }

    let dense_elements = dense_rows * dense_cols;
    let factor_elements = (dense_rows + dense_cols) * rank;
    let compression_ratio = dense_elements as f32 / factor_elements as f32;
    let relative_frobenius_error = if reference_sq == 0.0 {
        if error_sq == 0.0 { 0.0 } else { f32::INFINITY }
    } else {
        (error_sq / reference_sq).sqrt() as f32
    };

    Ok(RankMetrics {
        rank,
        dense_elements,
        factor_elements,
        dense_bytes_f32: dense_elements * std::mem::size_of::<f32>(),
        factor_bytes_f32: factor_elements * std::mem::size_of::<f32>(),
        compression_ratio,
        relative_frobenius_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_rank_vector_execution_matches_materialized_product() {
        let low_rank = LowRankMatrix::new(
            2,
            3,
            1,
            vec![2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ).unwrap();
        let x = vec![1.0, 2.0, 3.0];
        let fast = low_rank.execute_vector(&x).unwrap();
        let dense = low_rank.materialize();
        let expected = vec![
            dense[0] * x[0] + dense[1] * x[1] + dense[2] * x[2],
            dense[3] * x[0] + dense[4] * x[1] + dense[5] * x[2],
        ];
        for (actual, expected) in fast.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn rank_metrics_report_factor_storage() {
        let reference = vec![1.0; 12];
        let reconstruction = reference.clone();
        let metrics = rank_metrics(3, 4, 1, &reconstruction, &reference).unwrap();
        assert_eq!(metrics.dense_elements, 12);
        assert_eq!(metrics.factor_elements, 7);
        assert_eq!(metrics.factor_bytes_f32, 28);
        assert_eq!(metrics.relative_frobenius_error, 0.0);
    }
}
