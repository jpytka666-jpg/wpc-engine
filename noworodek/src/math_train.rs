//! Small deterministic math-training harness used to prove that a
//! WeightSet-backed parameter can learn a supervised mathematical relation.
//!
//! This is deliberately a tiny CPU/reference experiment. It is not the full
//! Transformer trainer and does not claim GPU execution.

use crate::{ArchitectureId, ExperienceId, LinearTrainer, MemoryWeightBackend, Tensor, TrainingStepReport, WeightSetHeader, WeightSetId, WeightSetManager, WeightSetManifest, WeightSetVersion, DType, TensorSpec};

#[derive(Clone, Debug)]
pub struct MathSample {
    pub input_a: f32,
    pub input_b: f32,
    pub target: f32,
}

#[derive(Clone, Debug)]
pub struct MathDataset {
    pub train: Vec<MathSample>,
    pub held_out: Vec<MathSample>,
}

#[derive(Clone, Debug, Default)]
pub struct MathEvalReport {
    pub mse: f32,
    pub exact_accuracy: f32,
}

#[derive(Clone, Debug, Default)]
pub struct MathTrainReport {
    pub before: MathEvalReport,
    pub after: MathEvalReport,
    pub last_loss: f32,
    pub weight_before: Vec<f32>,
    pub weight_after: Vec<f32>,
}

pub fn generate_dataset() -> MathDataset {
    let train = (0..16)
        .map(|i| {
            let a = i as f32;
            let b = (15 - i) as f32;
            MathSample { input_a: a, input_b: b, target: 2.0 * a + 3.0 * b }
        })
        .collect();
    let held_out = (0..8)
        .map(|i| {
            let a = i as f32 + 0.5;
            let b = 20.0 - a;
            MathSample { input_a: a, input_b: b, target: 2.0 * a + 3.0 * b }
        })
        .collect();
    MathDataset { train, held_out }
}

pub fn evaluate(weights: &[f32], samples: &[MathSample]) -> MathEvalReport {
    if samples.is_empty() {
        return MathEvalReport::default();
    }
    let mut squared_error = 0.0;
    let mut exact = 0usize;
    for sample in samples {
        let prediction = weights[0] * sample.input_a + weights[1] * sample.input_b;
        let error = prediction - sample.target;
        squared_error += error * error;
        if error.abs() <= 1e-3 {
            exact += 1;
        }
    }
    MathEvalReport {
        mse: squared_error / samples.len() as f32,
        exact_accuracy: exact as f32 / samples.len() as f32,
    }
}

pub fn run_reference_training(epochs: usize, learning_rate: f32) -> Result<MathTrainReport, String> {
    let dataset = generate_dataset();
    let manifest = WeightSetManifest::new(
        WeightSetHeader::new(WeightSetId::new("math-student"), WeightSetVersion::new("0.1.0").map_err(|e| e.to_string())?, ArchitectureId::new("noworodek-math-linear-v1")),
        vec![TensorSpec::new("math.weights", vec![2, 1], DType::F32, "runtime")],
    ).map_err(|e| e.to_string())?;
    let backend = MemoryWeightBackend::with_tensor_data(manifest, [("math.weights", vec![0.0, 0.0])]);
    let mut manager = WeightSetManager::new(ArchitectureId::new("noworodek-math-linear-v1"));
    let weight_set = manager.mount(Box::new(backend)).map_err(|e| e.to_string())?;
    let trainer = LinearTrainer::new(weight_set.clone(), "math.weights", learning_rate).map_err(|e| e.to_string())?;

    let before = trainer.weight.read(&manager).map_err(|e| e.to_string())?.values().to_vec();
    let before_eval = evaluate(&before, &dataset.held_out);
    let experience_id = ExperienceId::new("math-reference-1").map_err(|e| e.to_string())?;

    let mut last_loss = 0.0;
    let mut step = 0u64;
    for _ in 0..epochs {
        for sample in &dataset.train {
            let input = Tensor::from_vec(vec![1, 2], vec![sample.input_a, sample.input_b]).map_err(|e| e.to_string())?;
            let target = Tensor::from_vec(vec![1, 1], vec![sample.target]).map_err(|e| e.to_string())?;
            let report: TrainingStepReport = trainer.train_step(&mut manager, &input, &target).map_err(|e| e.to_string())?;
            last_loss = report.loss;
            step += 1;
        }
    }

    let after = trainer.weight.read(&manager).map_err(|e| e.to_string())?.values().to_vec();
    let after_eval = evaluate(&after, &dataset.held_out);
    let _ = (experience_id, step);

    Ok(MathTrainReport { before: before_eval, after: after_eval, last_loss, weight_before: before, weight_after: after })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_dataset_has_train_and_held_out_splits() {
        let dataset = generate_dataset();
        assert!(!dataset.train.is_empty());
        assert!(!dataset.held_out.is_empty());
    }

    #[test]
    fn reference_training_improves_held_out_mse() {
        let report = run_reference_training(100, 0.001).unwrap();
        assert!(report.after.mse < report.before.mse);
        assert_ne!(report.weight_before, report.weight_after);
    }
}
