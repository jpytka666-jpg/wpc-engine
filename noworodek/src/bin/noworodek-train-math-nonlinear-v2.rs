use noworodek::{LinearTrainer, MemoryWeightBackend, Tensor, TensorSpec, WeightSetHeader, WeightSetId, WeightSetManager, WeightSetManifest, WeightSetVersion, ArchitectureId, DType};

fn features(a: f32, b: f32) -> [f32; 6] { [a, b, a*a, b*b, a*b, 1.0] }
fn target(a: f32, b: f32) -> f32 { 2.0*a*a + 3.0*b*b + 5.0*a*b + 7.0 }
fn predict(w: &[f32], a: f32, b: f32) -> f32 { features(a,b).iter().zip(w).map(|(x,y)| x*y).sum() }

fn eval(w: &[f32], samples: &[(f32,f32)]) -> f32 {
    samples.iter().map(|&(a,b)| { let d=predict(w,a,b)-target(a,b); d*d }).sum::<f32>() / samples.len() as f32
}

fn make_sets() -> (Vec<(f32,f32)>, Vec<(f32,f32)>) {
    let mut train=Vec::new(); let mut held=Vec::new();
    for ia in 0..=8 { for ib in 0..=8 {
        let a=-1.0 + ia as f32 * 0.25; let b=-1.0 + ib as f32 * 0.25;
        if (ia+ib)%3==0 { held.push((a,b)); } else { train.push((a,b)); }
    }}
    (train,held)
}

fn main() {
    let (train, held) = make_sets();
    let manifest = WeightSetManifest::new(
        WeightSetHeader::new(WeightSetId::new("math-nonlinear-v2"), WeightSetVersion::new("0.1.0").unwrap(), ArchitectureId::new("noworodek-math-feature-v2")),
        vec![TensorSpec::new("math.nonlinear.weights", vec![6,1], DType::F32, "runtime")],
    ).unwrap();
    let backend = MemoryWeightBackend::with_tensor_data(manifest, [("math.nonlinear.weights", vec![0.0;6])]);
    let mut manager=WeightSetManager::new(ArchitectureId::new("noworodek-math-feature-v2"));
    let id=manager.mount(Box::new(backend)).unwrap();
    let trainer=LinearTrainer::new(id.clone(), "math.nonlinear.weights", 0.01).unwrap();
    let before=trainer.weight.read(&manager).unwrap().values().to_vec();
    println!("NOWORODEK NONLINEAR MATH V2");
    println!("TRAIN={} HELDOUT={}", train.len(), held.len());
    println!("BEFORE heldout_mse={:.8} weights={:?}", eval(&before,&held), before);
    let mut step=0usize;
    for epoch in 0..500usize {
        for &(a,b) in &train {
            let x=Tensor::from_vec(vec![1,6], features(a,b).to_vec()).unwrap();
            let y=Tensor::from_vec(vec![1,1], vec![target(a,b)]).unwrap();
            let report=trainer.train_step(&mut manager,&x,&y).unwrap();
            step+=1;
            if step%500==0 {
                let w=trainer.weight.read(&manager).unwrap();
                let changed=w.values().iter().zip(&before).filter(|(a,b)| (*a-*b).abs()>1e-7).count();
                println!("epoch={} step={} train_loss={:.8} heldout_mse={:.8} weights={:?} changed={} max_abs_delta={:.6}", epoch+1, step, report.loss, eval(w.values(),&held), w.values(), changed, w.values().iter().zip(&before).map(|(a,b)| (*a-*b).abs()).fold(0.0,f32::max));
            }
        }
    }
    let after=trainer.weight.read(&manager).unwrap().values().to_vec();
    println!("AFTER heldout_mse={:.8}", eval(&after,&held));
    println!("AFTER weights={:?}", after);
    println!("EXPECTED weights=[0,0,2,3,5,7]");
}
