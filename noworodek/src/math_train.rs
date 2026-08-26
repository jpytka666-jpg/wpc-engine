//! Deterministic mathematics training experiment for Noworodek.
//!
//! This module provides a small controlled experiment around the existing
//! external-weight trainer. It deliberately separates training and held-out
//! evaluation so an improvement cannot be attributed to memorising the test set.

#[derive(Clone, Debug, PartialEq)]
pub struct MathSample {
    pub a: f32,
    pub b: f32,
    pub target: f32,
}

#[derive(Clone, Debug)]
pub struct MathDataset {
    pub train: Vec<MathSample>,
    pub held_out: Vec<MathSample>,
}

#[derive(Clone, Debug)]
pub struct EvalReport {
    pub mse: f32,
    pub exact_accuracy: f32,
}

#[derive(Clone, Debug)]
pub struct MathTrainReport {
    pub before: EvalReport,
    pub after: EvalReport,
    pub steps: usize,
}

/// Generate a deterministic y = 2a + 3b family with a disjoint held-out set.
/// The held-out set uses values outside the training grid to test interpolation.
pub fn generate_dataset() -> MathDataset {
    let train = (0..8)
        .flat_map(|a| (0..8).map(move |b| sample(a as f32, b as f32)))
        .collect();
    let held_out = (0..4)
        .flat_map(|a| (8..12).map(move |b| sample(a as f32, b as f32)))
        .collect();
    MathDataset { train, held_out }
}

fn sample(a: f32, b: f32) -> MathSample {
    MathSample {
        a,
        b,
        target: 2.0 * a + 3.0 * b,
    }
}

/// Evaluate a 2-feature linear model [w0, w1] on a dataset.
pub fn evaluate(weights: &[f32], samples: &[MathSample]) -> Result<EvalReport, String> {
    if weights.len() != 2 {
        return Err("math experiment expects exactly two weights".into());
    }
    if samples.is_empty() {
        return Err("cannot evaluate an empty dataset".into());
    }
    let mut sq = 0.0f32;
    let mut exact = 0usize;
    for s in samples {
        let pred = s.a * weights[0] + s.b * weights[1];
        let err = pred - s.target;
        sq += err * err;
        if err.abs() < 1e-4 {
            exact += 1;
        }
    }
    Ok(EvalReport {
        mse: sq / samples.len() as f32,
        exact_accuracy: exact as f32 / samples.len() as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_is_deterministic_and_disjoint() {
        let ds = generate_dataset();
        assert_eq!(ds.train.len(), 64);
        assert_eq!(ds.held_out.len(), 16);
        assert!(ds.train.iter().all(|x| x.a < 8.0 && x.b < 8.0));
        assert!(ds.held_out.iter().all(|x| x.b >= 8.0));
    }

    #[test]
    fn target_formula_is_correct() {
        let s = sample(2.0, 5.0);
        assert_eq!(s.target, 19.0);
    }

    #[test]
    fn evaluation_accepts_known_solution() {
        let ds = generate_dataset();
        let report = evaluate(&[2.0, 3.0], &ds.held_out).unwrap();
        assert_eq!(report.mse, 0.0);
        assert_eq!(report.exact_accuracy, 1.0);
    }

    #[test]
    fn wrong_weights_fail() {
        let ds = generate_dataset();
        let report = evaluate(&[0.0, 0.0], &ds.held_out).unwrap();
        assert!(report.mse > 0.0);
        assert_eq!(report.exact_accuracy, 0.0);
    }
}
