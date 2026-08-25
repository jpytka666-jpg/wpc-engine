use crate::weightset::WeightSetError;

#[derive(Clone, Debug)]
pub struct BigramLanguageModel {
    vocab_size: usize,
    learning_rate: f32,
    output_weights: Vec<f32>,
}

impl BigramLanguageModel {
    pub fn new(vocab_size: usize, _hidden_size: usize, learning_rate: f32, seed: u64) -> Self {
        assert!(vocab_size > 0, "vocab size must be positive");
        assert!(learning_rate.is_finite() && learning_rate > 0.0, "learning rate must be positive and finite");
        let mut state = seed.max(1);
        let mut output_weights = Vec::with_capacity(vocab_size * vocab_size);
        for _ in 0..(vocab_size * vocab_size) {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let unit = ((state >> 32) as u32) as f32 / u32::MAX as f32;
            output_weights.push((unit - 0.5) * 0.02);
        }
        Self { vocab_size, learning_rate, output_weights }
    }

    pub fn output_weights(&self) -> &[f32] { &self.output_weights }

    pub fn predict(&self, input: usize) -> Result<usize, WeightSetError> {
        self.validate_token(input)?;
        let row = &self.output_weights[input * self.vocab_size..(input + 1) * self.vocab_size];
        row.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(index, _)| index)
            .ok_or_else(|| WeightSetError::Backend("empty prediction row".into()))
    }

    pub fn loss(&self, pairs: &[(usize, usize)]) -> f32 {
        if pairs.is_empty() { return 0.0; }
        let mut total = 0.0;
        for &(input, target) in pairs {
            if input >= self.vocab_size || target >= self.vocab_size { return f32::INFINITY; }
            let row = &self.output_weights[input * self.vocab_size..(input + 1) * self.vocab_size];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = row.iter().map(|value| (*value - max).exp()).sum();
            let log_probability = row[target] - max - sum_exp.ln();
            total -= log_probability;
        }
        total / pairs.len() as f32
    }

    pub fn train_step(&mut self, pairs: &[(usize, usize)]) -> Result<f32, WeightSetError> {
        if pairs.is_empty() { return Ok(0.0); }
        let loss = self.loss(pairs);
        let mut gradient = vec![0.0f32; self.output_weights.len()];
        for &(input, target) in pairs {
            self.validate_token(input)?;
            self.validate_token(target)?;
            let start = input * self.vocab_size;
            let row = &self.output_weights[start..start + self.vocab_size];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = row.iter().map(|value| (*value - max).exp()).sum();
            for index in 0..self.vocab_size {
                let probability = (row[index] - max).exp() / sum_exp;
                let target_gradient = if index == target { 1.0 } else { 0.0 };
                gradient[start + index] += probability - target_gradient;
            }
        }
        let scale = self.learning_rate / pairs.len() as f32;
        for (weight, delta) in self.output_weights.iter_mut().zip(gradient) {
            *weight -= scale * delta;
        }
        Ok(loss)
    }

    fn validate_token(&self, token: usize) -> Result<(), WeightSetError> {
        if token >= self.vocab_size {
            Err(WeightSetError::Backend(format!("token {token} outside vocabulary of {}", self.vocab_size)))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BigramLanguageModel;
    use crate::{TrainingObservatory, WeightSetId};

    #[test]
    fn tiny_language_model_learns_a_repeated_transition() {
        let mut model = BigramLanguageModel::new(4, 4, 0.5, 7);
        let before = model.loss(&[(0, 1)]);
        for _ in 0..200 {
            model.train_step(&[(0, 1)]).unwrap();
        }
        let after = model.loss(&[(0, 1)]);
        assert!(after < before, "loss did not decrease: {before} -> {after}");
        assert_eq!(model.predict(0).unwrap(), 1);
    }

    #[test]
    fn training_step_reports_weight_delta_for_observatory() {
        let mut model = BigramLanguageModel::new(3, 3, 0.25, 11);
        let before = model.output_weights().to_vec();
        model.train_step(&[(0, 2)]).unwrap();
        let after = model.output_weights();
        assert!(before.iter().zip(after).any(|(a, b)| (a - b).abs() > 0.0));
    }

    #[test]
    fn observed_training_records_which_weights_changed() {
        let mut model = BigramLanguageModel::new(3, 3, 0.25, 19);
        let mut observatory = TrainingObservatory::new();
        model.train_step_observed(&[(0, 2)], 1, "rust-transition", WeightSetId::new("coding"), &mut observatory).unwrap();
        let observation = observatory.latest().unwrap();
        assert_eq!(observation.step, 1);
        assert_eq!(observation.experience_id, "rust-transition");
        assert_eq!(observation.deltas.len(), 1);
        assert!(observation.deltas[0].changed_elements > 0);
        assert!(observation.deltas[0].max_abs > 0.0);
    }
}
