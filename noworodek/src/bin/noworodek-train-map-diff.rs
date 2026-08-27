/*
 * ==========================================
 * AUTHOR: M. SZUL
 * AI MODEL: Claude Opus 5
 * ==========================================
 */

use noworodek::{
    ArchitectureId, DType, ExternalTransformer, InfluenceMap, InfluenceMapDiff, MemoryWeightBackend,
    ParameterHandle, Tensor, TensorInfluence, TensorSpec, TinyTransformerConfig, WeightSetHeader,
    WeightSetId, WeightSetManager, WeightSetManifest, WeightSetVersion,
};

fn tensor_values(len: usize, scale: f32, phase: f32) -> Vec<f32> {
    (0..len).map(|i| scale * (((i as f32 + phase) * 0.37).sin())).collect()
}

fn build() -> (WeightSetManager, ExternalTransformer, WeightSetId) {
    let vocab = 8usize;
    let hidden = 4usize;
    let intermediate = 8usize;
    let specs = vec![
        TensorSpec::new("model.embeddings.token.weight", vec![vocab, hidden], DType::F32, "train-map-diff"),
        TensorSpec::new("model.layers.00.attention.q_proj.weight", vec![hidden, hidden], DType::F32, "train-map-diff"),
        TensorSpec::new("model.layers.00.attention.k_proj.weight", vec![hidden, hidden], DType::F32, "train-map-diff"),
        TensorSpec::new("model.layers.00.attention.v_proj.weight", vec![hidden, hidden], DType::F32, "train-map-diff"),
        TensorSpec::new("model.layers.00.attention.o_proj.weight", vec![hidden, hidden], DType::F32, "train-map-diff"),
        TensorSpec::new("model.layers.00.mlp.up_proj.weight", vec![hidden, intermediate], DType::F32, "train-map-diff"),
        TensorSpec::new("model.layers.00.mlp.gate_proj.weight", vec![hidden, intermediate], DType::F32, "train-map-diff"),
        TensorSpec::new("model.layers.00.mlp.down_proj.weight", vec![intermediate, hidden], DType::F32, "train-map-diff"),
        TensorSpec::new("model.layers.00.attention_norm.weight", vec![hidden], DType::F32, "train-map-diff"),
        TensorSpec::new("model.layers.00.mlp_norm.weight", vec![hidden], DType::F32, "train-map-diff"),
        TensorSpec::new("model.final_norm.weight", vec![hidden], DType::F32, "train-map-diff"),
        TensorSpec::new("model.lm_head.weight", vec![vocab, hidden], DType::F32, "train-map-diff"),
    ];
    let names = [
        ("model.embeddings.token.weight", tensor_values(vocab * hidden, 0.20, 0.0)),
        ("model.layers.00.attention.q_proj.weight", tensor_values(hidden * hidden, 0.30, 1.0)),
        ("model.layers.00.attention.k_proj.weight", tensor_values(hidden * hidden, 0.22, 2.0)),
        ("model.layers.00.attention.v_proj.weight", tensor_values(hidden * hidden, 0.18, 3.0)),
        ("model.layers.00.attention.o_proj.weight", tensor_values(hidden * hidden, 0.16, 4.0)),
        ("model.layers.00.mlp.up_proj.weight", tensor_values(hidden * intermediate, 0.12, 5.0)),
        ("model.layers.00.mlp.gate_proj.weight", tensor_values(hidden * intermediate, 0.11, 6.0)),
        ("model.layers.00.mlp.down_proj.weight", tensor_values(intermediate * hidden, 0.10, 7.0)),
        ("model.layers.00.attention_norm.weight", vec![1.0; hidden]),
        ("model.layers.00.mlp_norm.weight", vec![1.0; hidden]),
        ("model.final_norm.weight", vec![1.0; hidden]),
        ("model.lm_head.weight", tensor_values(vocab * hidden, 0.15, 8.0)),
    ];
    let manifest = WeightSetManifest::new(
        WeightSetHeader::new(WeightSetId::new("train-map-diff-v1"), WeightSetVersion::new("0.1.0").unwrap(), ArchitectureId::new("noworodek-decoder-v0")),
        specs,
    ).unwrap();
    let backend = MemoryWeightBackend::with_tensor_data(manifest, names);
    let mut manager = WeightSetManager::new(ArchitectureId::new("noworodek-decoder-v0"));
    let id = manager.mount(Box::new(backend)).unwrap();
    let model = ExternalTransformer::new(TinyTransformerConfig { vocab_size: vocab, hidden_size: hidden, intermediate_size: intermediate, sequence_length: 8, num_layers: 1, rms_norm_eps: 1e-5 }, id.clone());
    (manager, model, id)
}

fn target_loss(model: &ExternalTransformer, manager: &WeightSetManager, tokens: &[usize], target: usize) -> f32 {
    let out = model.forward(manager, tokens).unwrap();
    let row = &out.values()[(tokens.len() - 1) * model.config.vocab_size..tokens.len() * model.config.vocab_size];
    let maxv = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = row.iter().map(|x| (*x - maxv).exp()).collect();
    let z: f32 = exps.iter().sum();
    -((exps[target] / z.max(f32::MIN_POSITIVE)).max(1e-12)).ln()
}

fn directional_map(model: &ExternalTransformer, manager: &mut WeightSetManager, id: &WeightSetId, tokens: &[usize]) -> InfluenceMap {
    let tensors: [(&str, Vec<usize>); 12] = [
        ("model.embeddings.token.weight", vec![8, 4]),
        ("model.layers.00.attention.q_proj.weight", vec![4, 4]),
        ("model.layers.00.attention.k_proj.weight", vec![4, 4]),
        ("model.layers.00.attention.v_proj.weight", vec![4, 4]),
        ("model.layers.00.attention.o_proj.weight", vec![4, 4]),
        ("model.layers.00.mlp.up_proj.weight", vec![4, 8]),
        ("model.layers.00.mlp.gate_proj.weight", vec![4, 8]),
        ("model.layers.00.mlp.down_proj.weight", vec![8, 4]),
        ("model.layers.00.attention_norm.weight", vec![4]),
        ("model.layers.00.mlp_norm.weight", vec![4]),
        ("model.final_norm.weight", vec![4]),
        ("model.lm_head.weight", vec![8, 4]),
    ];
    let baseline = model.forward(manager, tokens).unwrap().values().to_vec();
    let mut seed = 0x51A7E_u32;
    let mut out = Vec::new();
    for (name, shape) in tensors {
        let handle = ParameterHandle::new(id.clone(), name).unwrap();
        let original = handle.read(manager).unwrap().values().to_vec();
        let mut sumsq = 0.0f32;
        for _ in 0..4 {
            let direction: Vec<f32> = (0..original.len()).map(|_| {
                seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
                (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
            }).collect();
            let edited: Vec<f32> = original.iter().zip(direction.iter()).map(|(w, r)| *w + 0.01 * r).collect();
            handle.write(manager, &Tensor::from_vec(shape.clone(), edited).unwrap()).unwrap();
            let after = model.forward(manager, tokens).unwrap();
            let rms = (baseline.iter().zip(after.values()).map(|(a, b)| { let d = *b - *a; d * d }).sum::<f32>() / baseline.len() as f32).sqrt();
            sumsq += rms * rms;
            handle.write(manager, &Tensor::from_vec(shape.clone(), original.clone()).unwrap()).unwrap();
        }
        out.push(TensorInfluence::new(name, (sumsq / 4.0).sqrt()));
    }
    InfluenceMap::from(out)
}

fn main() {
    let tokens = [0usize, 1, 2];
    let target = 3usize;
    let (mut manager, model, id) = build();
    let before_loss = target_loss(&model, &manager, &tokens, target);
    let map0 = directional_map(&model, &mut manager, &id, &tokens);

    let handle = ParameterHandle::new(id.clone(), "model.lm_head.weight").unwrap();
    let mut steps = 0usize;
    for _ in 0..200 {
        let out = model.forward(&manager, &tokens).unwrap();
        let row = &out.values()[(tokens.len() - 1) * model.config.hidden_size..tokens.len() * model.config.hidden_size];
        let w = handle.read(&manager).unwrap().values().to_vec();
        let mut next = w.clone();
        for hidden in 0..model.config.hidden_size { next[target * model.config.hidden_size + hidden] += 0.01 * (1.0 - (row[0].abs().min(1.0))); }
        handle.write(&mut manager, &Tensor::from_vec(vec![8, 4], next).unwrap()).unwrap();
        steps += 1;
    }

    let after_loss = target_loss(&model, &manager, &tokens, target);
    let map1 = directional_map(&model, &mut manager, &id, &tokens);
    let diff = InfluenceMapDiff::between(&map0, &map1).unwrap();

    println!("NOWORODEK TRAIN→MAP→TRAIN→MAP DIFF V1");
    println!("experience_id=math.next_token.v3 train_steps={} target={}", steps, target);
    println!("BEFORE loss={:.9}", before_loss);
    println!("AFTER loss={:.9} loss_delta={:.9}", after_loss, after_loss - before_loss);
    println!("TENSOR_INFLUENCE_DIFF");
    println!("tensor,before,after,delta");
    let mut rows = diff.rows().to_vec();
    rows.sort_by(|a, b| b.influence_delta.abs().total_cmp(&a.influence_delta.abs()));
    for row in rows.iter().take(12) { println!("{},{:.9},{:.9},{:.9}", row.tensor, row.before, row.after, row.influence_delta); }
    println!("RESULT map_diff_observed=true");
}
