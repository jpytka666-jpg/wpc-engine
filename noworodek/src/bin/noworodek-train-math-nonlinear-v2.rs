/*
 * ==========================================
 * AUTHOR: M. SZUL
 * AI MODEL: Claude Opus 5
 * ==========================================
 */

use noworodek::{ArchitectureId, DType, LinearTrainer, MemoryWeightBackend, Tensor, TensorSpec, WeightSetHeader, WeightSetId, WeightSetManager, WeightSetManifest, WeightSetVersion};

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

fn main() {
    let mut train = Vec::new();
    for a_i in -10..=10 { for b_i in -10..=10 { train.push((a_i as f32, b_i as f32)); } }
    let mut held = Vec::new();
    for a_i in -9..=9 { for b_i in -8..=8 { if (a_i + b_i) % 3 == 0 { held.push((a_i as f32 + 0.25, b_i as f32 - 0.5)); } } }

    let mut batch_inputs = Vec::with_capacity(train.len() * 6);
    let mut batch_targets = Vec::with_capacity(train.len());
    for &(a, b) in &train {
        batch_inputs.extend_from_slice(&features(a, b));
        batch_targets.push(target(a, b));
    }

    let input = Tensor::from_vec(vec![train.len(), 6], batch_inputs).unwrap();
    let target_t = Tensor::from_vec(vec![train.len(), 1], batch_targets).unwrap();

    let manifest = WeightSetManifest::new(
        WeightSetHeader::new(WeightSetId::new("math-nonlinear-v2"), WeightSetVersion::new("0.1.0").unwrap(), ArchitectureId::new("noworodek-math-feature-v2")),
        vec![TensorSpec::new("math.weights", vec![6, 1], DType::F32, "math-v2")],
    ).unwrap();
    let backend = MemoryWeightBackend::with_tensor_data(manifest, [("math.weights", vec![0.0; 6])]);
    let mut manager = WeightSetManager::new(ArchitectureId::new("noworodek-math-feature-v2"));
    let id = manager.mount(Box::new(backend)).unwrap();
    let trainer = LinearTrainer::new(id.clone(), "math.weights", 0.00002).unwrap();
    let before = trainer.weight.read(&manager).unwrap().values().to_vec();
    println!("NOWORODEK MATH NONLINEAR V2 (FULL-BATCH)");
    println!("train_samples={} heldout_samples={}", train.len(), held.len());
    println!("BEFORE heldout_mse={:.8} weights={:?}", eval(&before, &held), before);

    for step in 1..=50000usize {
        trainer.train_step(&mut manager, &input, &target_t).unwrap();
        if step % 5000 == 0 {
            let w = trainer.weight.read(&manager).unwrap();
            println!("step={} heldout_mse={:.8} weights={:?}", step, eval(w.values(), &held), w.values());
        }
    }
    let after = trainer.weight.read(&manager).unwrap().values().to_vec();
    println!("AFTER heldout_mse={:.8}", eval(&after, &held));
    println!("AFTER weights={:?}", after);
    println!("EXPECTED weights=[0,0,2,3,5,7]");
}
