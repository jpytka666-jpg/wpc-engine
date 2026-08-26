//! Training orchestration over externally owned WeightSet tensors.
//!
//! The trainer deliberately routes parameter updates through ParameterHandle so
//! the model never takes ownership of trainable storage.

use crate::{ParameterHandle, Sgd, Tensor, WeightSetError, WeightSetId, WeightSetManager};

#[derive(Clone, Debug)]
pub struct TrainingStepReport {
    pub loss: f32,
    pub parameter_name: String,
    pub gradient_l2: f32,
    pub weight_l2_before: f32,
    pub weight_l2_after: f32,
}

pub struct LinearTrainer {
    pub weight: ParameterHandle,
    pub optimizer: Sgd,
}

impl LinearTrainer {
    pub fn new(weight_set: WeightSetId, tensor_name: impl Into<String>, learning_rate: f32) -> Result<Self, WeightSetError> {
        Ok(Self {
            weight: ParameterHandle::new(weight_set, tensor_name)?,
            optimizer: Sgd { learning_rate },
        })
    }

    /// One supervised step for Y = XW against a provided target.
    pub fn train_step(
        &self,
        manager: &mut WeightSetManager,
        input: &Tensor,
        target: &Tensor,
    ) -> Result<TrainingStepReport, WeightSetError> {
        let weight_before = self.weight.read(manager)?;
        let prediction = input.matmul(&weight_before)?;
        let (loss, grad_output) = crate::mse_loss(&prediction, target)?;
        let (_, grad_weight) = crate::linear_backward(
            &crate::LinearCache { input: input.clone(), weight: weight_before.clone() },
            &grad_output,
        )?;
        let gradient_l2 = l2(&grad_weight);
        let weight_l2_before = l2(&weight_before);
        let mut updated = weight_before.clone();
        self.optimizer.step(&mut updated, &grad_weight)?;
        self.weight.write(manager, &updated)?;
        let weight_l2_after = l2(&updated);

        Ok(TrainingStepReport {
            loss,
            parameter_name: self.weight.tensor_name().to_owned(),
            gradient_l2,
            weight_l2_before,
            weight_l2_after,
        })
    }
}

fn l2(tensor: &Tensor) -> f32 {
    tensor.values().iter().map(|v| v * v).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArchitectureId, DType, MemoryWeightBackend, TensorSpec, WeightSetHeader, WeightSetManifest, WeightSetVersion};

    fn manifest() -> WeightSetManifest {
        WeightSetManifest::new(
            WeightSetHeader::new(WeightSetId::new("student"), WeightSetVersion::new("0.1.0").unwrap(), ArchitectureId::new("noworodek-v0")),
            vec![TensorSpec::new("linear.weight", vec![2, 1], DType::F32, "x")],
        ).unwrap()
    }

    #[test]
    fn train_step_changes_external_weight() {
        let mut manager = WeightSetManager::new(ArchitectureId::new("noworodek-v0"));
        let backend = MemoryWeightBackend::with_tensor_data(manifest(), [("linear.weight", vec![0.0, 0.0])]);
        let id = manager.mount(Box::new(backend)).unwrap();
        let trainer = LinearTrainer::new(id.clone(), "linear.weight", 0.1).unwrap();
        let input = Tensor::from_vec(vec![1, 2], vec![2.0, 3.0]).unwrap();
        let target = Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap();
        let report = trainer.train_step(&mut manager, &input, &target).unwrap();
        assert!(report.loss > 0.0);
        let after = trainer.weight.read(&manager).unwrap();
        assert!(after.values().iter().any(|v| v.abs() > 0.0));
    }

    #[test]
    fn repeated_steps_reduce_linear_mse() {
        let mut manager = WeightSetManager::new(ArchitectureId::new("noworodek-v0"));
        let backend = MemoryWeightBackend::with_tensor_data(manifest(), [("linear.weight", vec![0.0, 0.0])]);
        let id = manager.mount(Box::new(backend)).unwrap();
        let trainer = LinearTrainer::new(id, "linear.weight", 0.1).unwrap();
        let input = Tensor::from_vec(vec![1, 2], vec![2.0, 3.0]).unwrap();
        let target = Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap();
        let first = trainer.train_step(&mut manager, &input, &target).unwrap();
        let mut last_loss = first.loss;
        for _ in 0..10 {
            last_loss = trainer.train_step(&mut manager, &input, &target).unwrap().loss;
        }
        assert!(last_loss < first.loss);
    }
}