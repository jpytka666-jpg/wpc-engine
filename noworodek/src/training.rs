#[cfg(test)]
mod tests {
    use super::BigramLanguageModel;

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
}
