use noworodek::{ArchitectureId, DType, MemoryWeightBackend, Tensor, TensorSpec, WeightSetHeader, WeightSetId, WeightSetManager, WeightSetManifest, WeightSetVersion, ParameterHandle};

fn features(a: f32, b: f32) -> [f32; 6] { [a, b, a * a, b * b, a * b, 1.0] }
fn target(a: f32, b: f32) -> f32 { 2.0 * a * a + 3.0 * b * b + 5.0 * a * b + 7.0 }
fn eval(weights: &[f32], samples: &[(f32, f32)]) -> f32 {
    let mut mse = 0.0;
    for &(a, b) in samples {
        let f = features(a, b);
        let prediction: f32 = weights.iter().zip(f).map(|(w, x)| w * x).sum();
        let e = prediction - target(a, b);
        mse += e * e;
    }
    mse / samples.len() as f32
}
fn predict(weights: &[f32], a: f32, b: f32) -> f32 {
    let f = features(a, b);
    weights.iter().zip(f).map(|(w, x)| w * x).sum()
}

fn main() {
    // Start from the measured V2 converged checkpoint.
    let learned = vec![-0.00048526586, -0.0000743697, 2.0000975, 2.9999862, 5.0000267, 6.994567];
    let samples = vec![(1.25, -2.5), (3.0, 4.0), (-2.0, 1.5), (5.5, -1.0)];

    let manifest = WeightSetManifest::new(
        WeightSetHeader::new(
            WeightSetId::new("math-surgery-v1"),
            WeightSetVersion::new("0.1.0").unwrap(),
            ArchitectureId::new("noworodek-math-feature-v2"),
        ),
        vec![TensorSpec::new("math.weights", vec![6, 1], DType::F32, "surgery")],
    ).unwrap();
    let backend = MemoryWeightBackend::with_tensor_data(manifest, [("math.weights", learned.clone())]);
    let mut manager = WeightSetManager::new(ArchitectureId::new("noworodek-math-feature-v2"));
    let id = manager.mount(Box::new(backend)).unwrap();
    let handle = ParameterHandle::new(id.clone(), "math.weights").unwrap();

    let before = handle.read(&manager).unwrap().values().to_vec();
    println!("NOWORODEK WEIGHT SURGERY MATH V1");
    println!("baseline weights={before:?}");
    println!("baseline predictions:");
    for &(a, b) in &samples { println!("  ({a:>6.2},{b:>6.2}) -> pred={:.6} target={:.6}", predict(&before, a, b), target(a, b)); }

    // Index 4 is the ab coefficient. Surgical edit: 5 -> 10.
    let mut edited = before.clone();
    edited[4] = 10.0;
    handle.write(&mut manager, &Tensor::from_vec(vec![6, 1], edited.clone()).unwrap()).unwrap();
    let after = handle.read(&manager).unwrap().values().to_vec();

    println!("EDIT: ab coefficient 5 -> 10");
    println!("edited weights={after:?}");
    for &(a, b) in &samples { println!("  ({a:>6.2},{b:>6.2}) -> pred={:.6} original_target={:.6}", predict(&after, a, b), target(a, b)); }

    // Restore exact learned checkpoint and verify deterministic restoration.
    handle.write(&mut manager, &Tensor::from_vec(vec![6, 1], learned.clone()).unwrap()).unwrap();
    let restored = handle.read(&manager).unwrap().values().to_vec();
    println!("RESTORE exact_checkpoint_match={}", restored == learned);
    println!("baseline_demo_mse={:.8}", eval(&before, &samples));
    println!("edited_demo_mse={:.8}", eval(&after, &samples));
}
