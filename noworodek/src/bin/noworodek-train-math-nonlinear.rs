//! Nonlinear math experiment for the Noworodek WeightSet learning path.
//!
//! The learner is still a linear parameterization, but the input basis is
//! nonlinear: [a, b, a^2, b^2, a*b, 1]. This lets us inspect whether training
//! discovers the coefficient structure of a nonlinear mathematical relation.

use noworodek::{ArchitectureId, DType, ExperienceId, MemoryWeightBackend, ObservedLinearTrainer, Tensor, TensorSpec, TrainingObservatory, WeightSetHeader, WeightSetId, WeightSetManifest, WeightSetManager, WeightSetVersion};

#[derive(Clone, Copy)]
struct Sample { a: f32, b: f32 }

fn features(s: Sample) -> [f32; 6] {
    [s.a, s.b, s.a * s.a, s.b * s.b, s.a * s.b, 1.0]
}

fn target(s: Sample) -> f32 {
    2.0 * s.a * s.a + 3.0 * s.b * s.b + 5.0 * s.a * s.b + 7.0
}

fn samples() -> (Vec<Sample>, Vec<Sample>) {
    let mut train = Vec::new();
    for ia in -4..=4 {
        for ib in -4..=4 {
            train.push(Sample { a: ia as f32 / 2.0, b: ib as f32 / 2.0 });
        }
    }
    let held_out = vec![
        Sample { a: -1.75, b: 0.25 },
        Sample { a: -1.25, b: 1.75 },
        Sample { a: -0.25, b: -1.75 },
        Sample { a: 0.75, b: 1.25 },
        Sample { a: 1.75, b: -0.75 },
        Sample { a: 1.25, b: 1.75 },
    ];
    (train, held_out)
}

fn evaluate(manager: &WeightSetManager, trainer: &ObservedLinearTrainer, set: &[Sample]) -> f32 {
    let weight = trainer.inner.weight.read(manager).expect("read weight");
    let mut mse = 0.0;
    for sample in set {
        let x = features(*sample);
        let prediction: f32 = x.iter().enumerate().map(|(i, v)| *v * weight.values()[i]).sum();
        let error = prediction - target(*sample);
        mse += error * error;
    }
    mse / set.len() as f32
}

fn main() {
    let (train, held_out) = samples();
    let manifest = WeightSetManifest::new(
        WeightSetHeader::new(
            WeightSetId::new("math-nonlinear-student"),
            WeightSetVersion::new("0.2.0").expect("version"),
            ArchitectureId::new("noworodek-math-nonlinear-v1"),
        ),
        vec![TensorSpec::new("math.nonlinear.weights", vec![6, 1], DType::F32, "nonlinear-polynomial-basis")],
    ).expect("manifest");

    let backend = MemoryWeightBackend::with_tensor_data(
        manifest,
        [("math.nonlinear.weights", vec![0.0; 6])],
    );
    let mut manager = WeightSetManager::new(ArchitectureId::new("noworodek-math-nonlinear-v1"));
    let id = manager.mount(Box::new(backend)).expect("mount WeightSet");
    let mut trainer = ObservedLinearTrainer {
        inner: noworodek::LinearTrainer::new(id, "math.nonlinear.weights", 0.00005).expect("trainer"),
        observatory: TrainingObservatory::new(),
    };

    let before_weight = trainer.inner.weight.read(&manager).expect("before weight");
    let before_mse = evaluate(&manager, &trainer, &held_out);
    println!("NOWORODEK NONLINEAR MATH TRAIN");
    println!("TARGET y = 2a^2 + 3b^2 + 5ab + 7");
    println!("BASIS  [a,b,a^2,b^2,ab,1]");
    println!("BEFORE heldout_mse={before_mse:.8}");
    println!("BEFORE weights={:?}", before_weight.values());

    let mut step = 0u64;
    for epoch in 0..500usize {
        for sample in &train {
            let x = Tensor::from_vec(vec![1, 6], features(*sample).to_vec()).expect("input");
            let y = Tensor::from_vec(vec![1, 1], vec![target(*sample)]).expect("target");
            let experience_id = ExperienceId::new(format!("math-nonlinear-{step}" )).expect("experience id");
            let report = trainer.train_step(&mut manager, &x, &y, experience_id, step).expect("train step");
            step += 1;
            if step % 500 == 0 {
                let mse = evaluate(&manager, &trainer, &held_out);
                let weight = trainer.inner.weight.read(&manager).expect("weight");
                let obs = trainer.observatory.observations();
                let last = obs.last().expect("observation");
                println!(
                    "epoch={} step={} train_loss={:.8} heldout_mse={:.8} weights={:?} changed={} max_abs_delta={:.6}",
                    epoch + 1, step, report.loss, mse, weight.values(), last.deltas[0].changed_elements, last.deltas[0].max_abs
                );
            }
        }
    }

    let after_weight = trainer.inner.weight.read(&manager).expect("after weight");
    let after_mse = evaluate(&manager, &trainer, &held_out);
    let observations = trainer.observatory.observations();
    println!("AFTER heldout_mse={after_mse:.8}");
    println!("AFTER weights={:?}", after_weight.values());
    println!("OBSERVATIONS={}", observations.len());
    println!("EXPECTED weights=[0,0,2,3,5,7]");

    assert!(after_mse < before_mse, "held-out MSE did not improve");
    assert_eq!(after_weight.values().len(), 6);
}
