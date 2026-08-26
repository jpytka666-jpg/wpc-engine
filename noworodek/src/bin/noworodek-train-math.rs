use noworodek::{
    evaluate_math_model, generate_math_dataset, ArchitectureId, DType, LinearTrainer,
    MathDataset, MemoryWeightBackend, Tensor, TensorSpec, WeightSetHeader, WeightSetId,
    WeightSetManifest, WeightSetManager, WeightSetVersion,
};

fn manifest() -> WeightSetManifest {
    WeightSetManifest::new(
        WeightSetHeader::new(
            WeightSetId::new("math-student"),
            WeightSetVersion::new("0.1.0").expect("version"),
            ArchitectureId::new("noworodek-math-linear-v0"),
        ),
        vec![TensorSpec::new(
            "math.linear.weight",
            vec![2, 1],
            DType::F32,
            "math-domain",
        )],
    )
    .expect("valid manifest")
}

fn evaluate_external(manager: &WeightSetManager, trainer: &LinearTrainer, dataset: &MathDataset) -> f32 {
    let weight = trainer.weight.read(manager).expect("read external weight");
    let values = weight.values();
    let weights = [values[0], values[1]];
    evaluate_math_model(&weights, &dataset.held_out)
        .expect("valid math model")
        .mse
}

fn main() {
    let dataset = generate_math_dataset();
    let mut manager = WeightSetManager::new(ArchitectureId::new("noworodek-math-linear-v0"));
    let backend = MemoryWeightBackend::with_tensor_data(
        manifest(),
        [("math.linear.weight", vec![0.0, 0.0])],
    );
    let id = manager.mount(Box::new(backend)).expect("mount WeightSet");
    let trainer = LinearTrainer::new(id, "math.linear.weight", 0.01).expect("create trainer");

    let before = evaluate_external(&manager, &trainer, &dataset);
    println!("NOWORODEK MATH TRAIN");
    println!("BEFORE heldout_mse={:.8}", before);

    let input_a = Tensor::from_vec(vec![1, 2], vec![0.0, 0.0]).expect("input tensor");
    let target_a = Tensor::from_vec(vec![1, 1], vec![0.0]).expect("target tensor");
    let mut last_train_loss = 0.0f32;

    // Small supervised family: y = 2a + 3b.
    for step in 0..2000usize {
        let mut epoch_loss = 0.0f32;
        for sample in &dataset.train {
            let input = Tensor::from_vec(vec![1, 2], vec![sample.a, sample.b]).expect("sample input");
            let target = Tensor::from_vec(vec![1, 1], vec![sample.target]).expect("sample target");
            epoch_loss += trainer
                .train_step(&mut manager, &input, &target)
                .expect("train step")
                .loss;
        }
        last_train_loss = epoch_loss / dataset.train.len() as f32;

        if step % 250 == 249 {
            let heldout_mse = evaluate_external(&manager, &trainer, &dataset);
            let weight = trainer.weight.read(&manager).expect("read weight");
            println!(
                "step={} train_mse={:.8} heldout_mse={:.8} w=[{:.6},{:.6}]",
                step + 1,
                last_train_loss,
                heldout_mse,
                weight.values()[0],
                weight.values()[1]
            );
        }
    }

    let after_weight = trainer.weight.read(&manager).expect("read final weight");
    let weights = [after_weight.values()[0], after_weight.values()[1]];
    let after = evaluate_math_model(&weights, &dataset.held_out).expect("evaluate final model");

    println!(
        "AFTER heldout_mse={:.8} exact={:.4} weights=[{:.6},{:.6}]",
        after.mse,
        after.exact_accuracy,
        weights[0],
        weights[1]
    );
    println!("RESULT train_loss={:.8}", last_train_loss);

    assert!(after.mse < before, "held-out MSE did not improve");
    let _ = (input_a, target_a);
}
