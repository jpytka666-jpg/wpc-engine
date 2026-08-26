//! Low-rank representation primitives for the first alternative WeightSet backend.
//!
//! This module deliberately stays representation-level: factor storage and
//! metrics are defined here, while backend-specific GPU execution is a later
//! stage. The first experiment compares real tensors against the proven WPC
//! fused baseline before promoting low-rank to a default representation.

use rayon::prelude::*;

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

/// Deterministic pseudo-random source.
///
/// A measurement that cannot be repeated is not a measurement, so the starting subspace
/// is drawn from a fixed seed rather than from the clock. Same input, same factors,
/// every time.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed })
    }

    /// Uniform in [-1, 1).
    fn next_signed(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let v = x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        (v as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
    }
}

/// C[m x n] = A[m x k] * B[k x n], all row-major.
fn mul(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    c.par_chunks_mut(n).enumerate().for_each(|(row, out)| {
        for kk in 0..k {
            let av = a[row * k + kk];
            if av == 0.0 {
                continue;
            }
            let brow = &b[kk * n..kk * n + n];
            for (o, &bv) in out.iter_mut().zip(brow) {
                *o += av * bv;
            }
        }
    });
    c
}

/// C[k x n] = A^T[k x m] * B[m x n], with A stored as m x k row-major.
fn mul_at_b(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; k * n];
    c.par_chunks_mut(n).enumerate().for_each(|(kk, out)| {
        for row in 0..m {
            let av = a[row * k + kk];
            if av == 0.0 {
                continue;
            }
            let brow = &b[row * n..row * n + n];
            for (o, &bv) in out.iter_mut().zip(brow) {
                *o += av * bv;
            }
        }
    });
    c
}

/// Make the columns of an `m x n` row-major matrix orthonormal, in place.
///
/// Modified Gram-Schmidt, accumulating in f64. The accumulation matters: at 4096 rows a
/// single-precision dot product loses enough that later columns drift out of
/// orthogonality, which quietly degrades every factor that follows.
fn orthonormalize_columns(mat: &mut [f32], m: usize, n: usize) {
    for j in 0..n {
        for i in 0..j {
            let mut dot = 0.0f64;
            for r in 0..m {
                dot += f64::from(mat[r * n + j]) * f64::from(mat[r * n + i]);
            }
            let d = dot as f32;
            if d != 0.0 {
                for r in 0..m {
                    mat[r * n + j] -= d * mat[r * n + i];
                }
            }
        }
        let mut norm_sq = 0.0f64;
        for r in 0..m {
            let v = f64::from(mat[r * n + j]);
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt() as f32;
        if norm > 1e-12 {
            for r in 0..m {
                mat[r * n + j] /= norm;
            }
        } else {
            // The matrix had fewer independent directions than the rank asked for.
            // A zero column contributes nothing rather than amplifying noise.
            for r in 0..m {
                mat[r * n + j] = 0.0;
            }
        }
    }
}

/// Factor a dense matrix into `A * B` of the requested rank.
///
/// This is the link the module was missing: `LowRankMatrix::new` demands finished factors
/// and nothing here produced them.
///
/// Method is randomized subspace iteration, not a full decomposition. A 2560x4096 tensor
/// has 2560 singular directions, and the question being asked only ever concerns the
/// strongest few dozen, so computing all of them would be paying for an answer nobody
/// reads. Two power iterations sharpen the subspace enough that the result sits close to
/// the best approximation available at that rank.
///
/// The approximation is one-sided by construction: `A` is an orthonormal basis of the
/// captured subspace and `B` is the matrix projected onto it. That is exactly the form
/// `LowRankMatrix` stores and `execute_vector` consumes.
pub fn low_rank_decompose(
    dense: &[f32],
    rows: usize,
    cols: usize,
    rank: usize,
) -> Result<LowRankMatrix, LowRankError> {
    low_rank_decompose_seeded(dense, rows, cols, rank, 2, 0)
}

/// As `low_rank_decompose`, with the sharpening passes and the seed exposed.
pub fn low_rank_decompose_seeded(
    dense: &[f32],
    rows: usize,
    cols: usize,
    rank: usize,
    power_iterations: usize,
    seed: u64,
) -> Result<LowRankMatrix, LowRankError> {
    if rows == 0 || cols == 0 {
        return Err(LowRankError::InvalidShape);
    }
    if rank == 0 || rank > rows.min(cols) {
        return Err(LowRankError::InvalidRank);
    }
    if dense.len() != rows * cols {
        return Err(LowRankError::DataLengthMismatch);
    }

    // A random starting subspace of the column space.
    let mut rng = Rng::new(seed);
    let mut omega = vec![0.0f32; cols * rank];
    for value in omega.iter_mut() {
        *value = rng.next_signed();
    }

    // Q spans the directions this matrix stretches most.
    let mut q = mul(dense, rows, cols, &omega, rank);
    orthonormalize_columns(&mut q, rows, rank);

    // Each pass pushes Q further toward the leading subspace and away from the noise the
    // random start brought with it.
    for _ in 0..power_iterations {
        let mut z = mul_at_b(dense, rows, cols, &q, rank);
        orthonormalize_columns(&mut z, cols, rank);
        q = mul(dense, rows, cols, &z, rank);
        orthonormalize_columns(&mut q, rows, rank);
    }

    // B = Q^T * W: the matrix as seen from inside that subspace.
    let b = mul_at_b(&q, rows, rank, dense, cols);
    LowRankMatrix::new(rows, cols, rank, q, b)
}

/// Find the smallest rank whose reconstruction stays within `max_relative_error`.
///
/// Doubling search rather than a linear walk: every attempt costs a full decomposition,
/// so the search should ask few questions. Returns the factors together with the metrics
/// that justified stopping, so the choice never has to be taken on trust.
///
/// Returns `InvalidRank` when even `rank_cap` cannot meet the target -- which is itself a
/// finding, and a more useful one than a silently poor factorisation.
pub fn low_rank_decompose_to_error(
    dense: &[f32],
    rows: usize,
    cols: usize,
    max_relative_error: f32,
    rank_cap: usize,
) -> Result<(LowRankMatrix, RankMetrics), LowRankError> {
    let ceiling = rank_cap.min(rows.min(cols));
    if ceiling == 0 {
        return Err(LowRankError::InvalidRank);
    }
    let mut rank = 1usize;
    loop {
        let factors = low_rank_decompose(dense, rows, cols, rank)?;
        let metrics = rank_metrics(rows, cols, rank, &factors.materialize(), dense)?;
        if metrics.relative_frobenius_error <= max_relative_error {
            return Ok((factors, metrics));
        }
        if rank >= ceiling {
            return Err(LowRankError::InvalidRank);
        }
        rank = (rank * 2).min(ceiling);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A matrix whose rank is known because it was built that way: the product of an
    /// `rows x rank` and a `rank x cols` factor pair drawn from a fixed seed.
    fn synthetic_of_known_rank(rows: usize, cols: usize, rank: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        let left: Vec<f32> = (0..rows * rank).map(|_| rng.next_signed()).collect();
        let right: Vec<f32> = (0..rank * cols).map(|_| rng.next_signed()).collect();
        LowRankMatrix::new(rows, cols, rank, left, right)
            .unwrap()
            .materialize()
    }

    fn relative_error(dense: &[f32], rows: usize, cols: usize, rank: usize) -> f32 {
        let factors = low_rank_decompose(dense, rows, cols, rank).unwrap();
        rank_metrics(rows, cols, rank, &factors.materialize(), dense)
            .unwrap()
            .relative_frobenius_error
    }

    #[test]
    fn recovers_a_matrix_whose_rank_is_known() {
        // The decisive test: if this fails, every number the sweep reports is measuring
        // the decomposer rather than the tensor.
        let dense = synthetic_of_known_rank(40, 60, 4, 11);
        assert!(
            relative_error(&dense, 40, 60, 4) < 1e-5,
            "rank 4 must reproduce a rank-4 matrix almost exactly"
        );
    }

    #[test]
    fn too_low_a_rank_cannot_reproduce_it() {
        let dense = synthetic_of_known_rank(40, 60, 4, 11);
        let short = relative_error(&dense, 40, 60, 3);
        let exact = relative_error(&dense, 40, 60, 4);
        assert!(
            short > exact * 100.0 && short > 1e-3,
            "rank 3 error {short} should be far worse than rank 4 error {exact}"
        );
    }

    #[test]
    fn error_never_grows_as_rank_grows() {
        let dense = synthetic_of_known_rank(32, 48, 8, 5);
        let mut previous = f32::INFINITY;
        for rank in [1usize, 2, 4, 8] {
            let err = relative_error(&dense, 32, 48, rank);
            assert!(err <= previous + 1e-6, "rank {rank} error {err} rose above {previous}");
            previous = err;
        }
    }

    #[test]
    fn factors_have_the_shapes_the_container_requires() {
        let dense = synthetic_of_known_rank(12, 20, 3, 7);
        let factors = low_rank_decompose(&dense, 12, 20, 3).unwrap();
        assert_eq!(factors.left().len(), 12 * 3);
        assert_eq!(factors.right().len(), 3 * 20);
        assert_eq!(factors.rank(), 3);
    }

    #[test]
    fn decomposed_factors_execute_without_materializing() {
        // The whole point of the representation: A*(B*x) must equal (A*B)*x.
        let dense = synthetic_of_known_rank(16, 24, 5, 3);
        let factors = low_rank_decompose(&dense, 16, 24, 5).unwrap();
        let x: Vec<f32> = (0..24).map(|i| (i as f32 * 0.37).sin()).collect();
        let fast = factors.execute_vector(&x).unwrap();
        let materialized = factors.materialize();
        for row in 0..16 {
            let slow: f32 = (0..24).map(|c| materialized[row * 24 + c] * x[c]).sum();
            assert!((fast[row] - slow).abs() < 1e-4, "row {row}: {} vs {slow}", fast[row]);
        }
    }

    #[test]
    fn searching_by_error_stops_at_the_rank_that_meets_it() {
        let dense = synthetic_of_known_rank(24, 32, 4, 9);
        let (factors, metrics) =
            low_rank_decompose_to_error(&dense, 24, 32, 1e-4, 24).unwrap();
        assert!(metrics.relative_frobenius_error <= 1e-4);
        assert!(
            factors.rank() <= 8,
            "a rank-4 matrix should not need rank {} to hit 1e-4",
            factors.rank()
        );
    }

    #[test]
    fn an_impossible_target_is_refused_rather_than_approximated() {
        let dense = synthetic_of_known_rank(20, 20, 12, 4);
        assert_eq!(
            low_rank_decompose_to_error(&dense, 20, 20, 1e-9, 2),
            Err(LowRankError::InvalidRank)
        );
    }

    #[test]
    fn the_same_seed_gives_the_same_factors() {
        let dense = synthetic_of_known_rank(18, 22, 6, 2);
        let a = low_rank_decompose_seeded(&dense, 18, 22, 6, 2, 1234).unwrap();
        let b = low_rank_decompose_seeded(&dense, 18, 22, 6, 2, 1234).unwrap();
        assert_eq!(a, b, "a measurement that cannot be repeated is not a measurement");
    }

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
