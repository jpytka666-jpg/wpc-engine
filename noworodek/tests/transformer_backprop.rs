use noworodek::{ArchitectureId, DType, ExternalTransformer, MemoryWeightBackend, TensorSpec, TinyTransformerConfig, WeightSetHeader, WeightSetId, WeightSetManager, WeightSetManifest, WeightSetVersion};
use noworodek::model::transformer_backprop::train_step_ce;

fn values(len: usize, scale: f32, phase: f32) -> Vec<f32> {
    (0..len).map(|i| scale * (((i as f32 + phase) * 0.37).sin())).collect()
}

fn fixture() -> (WeightSetManager, ExternalTransformer) {
    let vocab = 8usize; let hidden = 4usize; let intermediate = 8usize;
    let mut specs = vec![
        TensorSpec::new("model.embeddings.token.weight", vec![vocab, hidden], DType::F32, "bp"),
        TensorSpec::new("model.layers.00.attention.q_proj.weight", vec![hidden, hidden], DType::F32, "bp"),
        TensorSpec::new("model.layers.00.attention.k_proj.weight", vec![hidden, hidden], DType::F32, "bp"),
        TensorSpec::new("model.layers.00.attention.v_proj.weight", vec![hidden, hidden], DType::F32, "bp"),
        TensorSpec::new("model.layers.00.attention.o_proj.weight", vec![hidden, hidden], DType::F32, "bp"),
        TensorSpec::new("model.layers.00.mlp.up_proj.weight", vec![hidden, intermediate], DType::F32, "bp"),
        TensorSpec::new("model.layers.00.mlp.gate_proj.weight", vec![hidden, intermediate], DType::F32, "bp"),
        TensorSpec::new("model.layers.00.mlp.down_proj.weight", vec![intermediate, hidden], DType::F32, "bp"),
        TensorSpec::new("model.layers.00.attention_norm.weight", vec![hidden], DType::F32, "bp"),
        TensorSpec::new("model.layers.00.mlp_norm.weight", vec![hidden], DType::F32, "bp"),
        TensorSpec::new("model.final_norm.weight", vec![hidden], DType::F32, "bp"),
        TensorSpec::new("model.lm_head.weight", vec![vocab, hidden], DType::F32, "bp"),
    ];
    let header = WeightSetHeader::new(WeightSetId::new("transformer-backprop-v1"), WeightSetVersion::new("0.1.0").unwrap(), ArchitectureId::new("noworodek-decoder-v0"));
    let manifest = WeightSetManifest::new(header, specs).unwrap();
    let data = [
        ("model.embeddings.token.weight", values(vocab*hidden, 0.20, 0.0)),
        ("model.layers.00.attention.q_proj.weight", values(hidden*hidden, 0.30, 1.0)),
        ("model.layers.00.attention.k_proj.weight", values(hidden*hidden, 0.22, 2.0)),
        ("model.layers.00.attention.v_proj.weight", values(hidden*hidden, 0.18, 3.0)),
        ("model.layers.00.attention.o_proj.weight", values(hidden*hidden, 0.16, 4.0)),
        ("model.layers.00.mlp.up_proj.weight", values(hidden*intermediate, 0.12, 5.0)),
        ("model.layers.00.mlp.gate_proj.weight", values(hidden*intermediate, 0.11, 6.0)),
        ("model.layers.00.mlp.down_proj.weight", values(intermediate*hidden, 0.10, 7.0)),
        ("model.layers.00.attention_norm.weight", vec![1.0; hidden]),
        ("model.layers.00.mlp_norm.weight", vec![1.0; hidden]),
        ("model.final_norm.weight", vec![1.0; hidden]),
        ("model.lm_head.weight", values(vocab*hidden, 0.15, 8.0)),
    ];
    let mut manager = WeightSetManager::new(ArchitectureId::new("noworodek-decoder-v0"));
    let id = manager.mount(Box::new(MemoryWeightBackend::with_tensor_data(manifest, data))).unwrap();
    let model = ExternalTransformer::new(TinyTransformerConfig { vocab_size:vocab, hidden_size:hidden, intermediate_size:intermediate, sequence_length:8, num_layers:1, rms_norm_eps:1e-5 }, id);
    (manager, model)
}

fn ce(model: &ExternalTransformer, manager: &WeightSetManager, tokens: &[usize], target: usize) -> f32 {
    let logits = model.forward(manager, tokens).unwrap();
    let row = &logits.values()[(tokens.len()-1)*model.config.vocab_size..tokens.len()*model.config.vocab_size];
    let maxv = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = row.iter().map(|x| (*x-maxv).exp()).collect();
    let z: f32 = exps.iter().sum();
    -((exps[target]/z.max(f32::MIN_POSITIVE)).max(1e-12)).ln()
}

#[test]
fn real_transformer_backprop_reduces_cross_entropy_and_updates_internal_weights() {
    let (mut manager, model) = fixture();
    let tokens = [0usize,1,2]; let target = 3usize;
    let before = ce(&model, &manager, &tokens, target);
    let q_before = noworodek::ParameterHandle::new(model.weight_set.clone(), "model.layers.00.attention.q_proj.weight").unwrap().read(&manager).unwrap();
    let mut internal_changed = false;
    let mut after = before;
    for _ in 0..40 {
        let report = train_step_ce(&model, &mut manager, &tokens, target, 0.01).unwrap();
        after = report.loss_after;
    }
    let q_after = noworodek::ParameterHandle::new(model.weight_set.clone(), "model.layers.00.attention.q_proj.weight").unwrap().read(&manager).unwrap();
    internal_changed = q_before.values().iter().zip(q_after.values()).any(|(a,b)| (*a-*b).abs() > 1e-9);
    assert!(after < before, "expected CE to decrease: before={before} after={after}");
    assert!(internal_changed, "expected analytic backward to update q_proj, not only lm_head");
}
