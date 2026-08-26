//! Observed training bridge: records before/after weight statistics for a
//! parameter update and links the update to a stable experience identifier.

use crate::{ExperienceId, LinearTrainer, Tensor, TensorDeltaSummary, TrainingObservation, TrainingObservatory, WeightSetError, WeightSetManager};

pub struct ObservedLinearTrainer {
    pub inner: LinearTrainer,
    pub observatory: TrainingObservatory,
}

impl ObservedLinearTrainer {
    pub fn train_step(
        &mut self,
        manager: &mut WeightSetManager,
        input: &Tensor,
        target: &Tensor,
        experience_id: ExperienceId,
        step: u64,
    ) -> Result<crate::TrainingStepReport, WeightSetError> {
        let before = self.inner.weight.read(manager)?;
        let report = self.inner.train_step(manager, input, target)?;
        let after = self.inner.weight.read(manager)?;
        let delta = before
            .values()
            .iter()
            .zip(after.values())
            .map(|(a, b)| b - a)
            .collect::<Vec<_>>();

        self.observatory.record(TrainingObservation {
            experience_id: experience_id.as_str().to_owned(),
            step,
            weight_set: self.inner.weight.weight_set().clone(),
            loss: Some(report.loss),
            deltas: vec![TensorDeltaSummary::from_delta(self.inner.weight.tensor_name(), &delta)],
        });

        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArchitectureId, DType, MemoryWeightBackend, TensorSpec, WeightSetHeader, WeightSetId, WeightSetManifest, WeightSetVersion};

    fn manifest() -> WeightSetManifest {
        WeightSetManifest::new(
            WeightSetHeader::new(WeightSetId::new("student"), WeightSetVersion::new("0.1.0").unwrap(), ArchitectureId::new("noworodek-v0")),
            vec![TensorSpec::new("linear.weight", vec![2, 1], DType::F32, "x")],
        ).unwrap()
    }

    #[test]
    fn observed_training_links_weight_delta_to_experience() {
        let mut manager = WeightSetManager::new(ArchitectureId::new("noworodek-v0"));
        let id = manager.mount(Box::new(MemoryWeightBackend::with_tensor_data(manifest(), [("linear.weight", vec![0.0, 0.0])]))).unwrap();
        let mut trainer = ObservedLinearTrainer { inner: LinearTrainer::new(id, "linear.weight", 0.1).unwrap(), observatory: TrainingObservatory::new() };
        let input = Tensor::from_vec(vec![1, 2], vec![2.0, 3.0]).unwrap();
        let target = Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap();
        trainer.train_step(&mut manager, &input, &target, ExperienceId::new("exp-1").unwrap(), 1).unwrap();
        let observations = trainer.observatory.observations();
        assert_eq!(observations.len(), 1);
        assert_eq!(observations[0].experience_id, "exp-1");
        assert!(observations[0].deltas[0].changed_elements > 0);
        assert!(observations[0].deltas[0].max_abs > 0.0);
    }
}
