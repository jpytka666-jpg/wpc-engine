use noworodek::{ArchitectureId, DType, ExternalTransformer, MemoryWeightBackend, ParameterHandle, Tensor, TensorSpec, TinyTransformerConfig, WeightSetHeader, WeightSetId, WeightSetManager, WeightSetManifest, WeightSetVersion};

fn tensor_values(len: usize, scale: f32, phase: f32) -> Vec<f32> {
    (0..len).map(|i| scale * (((i as f32) + phase) * 0.37).sin()).collect()
}

fn main() {
    let vocab = 8usize;
    let hidden = 4usize;
    let intermediate = 8usize;
    let layers = 1usize;

    let mut specs = Vec::new();
    specs.push(TensorSpec::new("model.embeddings.token.weight", vec![vocab, hidden], DType::F32, "transformer-surgery"));
    specs.push(TensorSpec::new("model.layers.00.attention.q_proj.weight", vec![hidden, hidden], DType::F32, "transformer-surgery"));
    specs.push(TensorSpec::new("model.layers.00.attention.k_proj.weight", vec![hidden, hidden], DType::F32, "transformer-surgery"));
    specs.push(TensorSpec::new("model.layers.00.attention.v_proj.weight", vec![hidden, hidden], DType::F32, "transformer-surgery"));
    specs.push(TensorSpec::new("model.layers.00.attention.o_proj.weight", vec![hidden, hidden], DType::F32, "transformer-surgery"));
    // ExternalTransformer executes XW, therefore up/gate are [hidden, intermediate].
    specs.push(TensorSpec::new("model.layers.00.mlp.up_proj.weight", vec![hidden, intermediate], DType::F32, "transformer-surgery"));
    specs.push(TensorSpec::new("model.layers.00.mlp.gate_proj.weight", vec![hidden, intermediate], DType::F32, "transformer-surgery"));
    specs.push(TensorSpec::new("model.layers.00.mlp.down_proj.weight", vec![intermediate, hidden], DType::F32, "transformer-surgery"));
    specs.push(TensorSpec::new("model.layers.00.attention_norm.weight", vec![hidden], DType::F32, "transformer-surgery"));
    specs.push(TensorSpec::new("model.layers.00.mlp_norm.weight", vec![hidden], DType::F32, "transformer-surgery"));
    specs.push(TensorSpec::new("model.final_norm.weight", vec![hidden], DType::F32, "transformer-surgery"));
    specs.push(TensorSpec::new("model.lm_head.weight", vec![vocab, hidden], DType::F32, "transformer-surgery"));

    let manifest = WeightSetManifest::new(
        WeightSetHeader::new(WeightSetId::new("transformer-surgery-v1"), WeightSetVersion::new("0.1.0").unwrap(), ArchitectureId::new("noworodek-decoder-v0")),
        specs,
    ).unwrap();

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

    let backend = MemoryWeightBackend::with_tensor_data(manifest, names);
    let mut manager = WeightSetManager::new(ArchitectureId::new("noworodek-decoder-v0"));
    let weight_set = manager.mount(Box::new(backend)).unwrap();
    let model = ExternalTransformer::new(TinyTransformerConfig {
        vocab_size: vocab,
        hidden_size: hidden,
        intermediate_size: intermediate,
        sequence_length: 8,
        num_layers: layers,
        rms_norm_eps: 1e-5,
    }, weight_set.clone());

    let tokens = [0usize, 1, 2];
    let before = model.forward(&manager, &tokens).unwrap();
    let before_values = before.values().to_vec();

    let q_handle = ParameterHandle::new(weight_set.clone(), "model.layers.00.attention.q_proj.weight").unwrap();
    let q_before = q_handle.read(&manager).unwrap().values().to_vec();
    let mut q_edited = q_before.clone();
    q_edited[0] += 1.0;
    q_handle.write(&mut manager, &Tensor::from_vec(vec![hidden, hidden], q_edited.clone()).unwrap()).unwrap();

    let after = model.forward(&manager, &tokens).unwrap();
    let changed = before_values.iter().zip(after.values()).filter(|(a, b)| (*a - *b).abs() > 1e-8).count();
    let max_delta = before_values.iter().zip(after.values()).map(|(a, b)| (*a - *b).abs()).fold(0.0_f32, f32::max);

    q_handle.write(&mut manager, &Tensor::from_vec(vec![hidden, hidden], q_before.clone()).unwrap()).unwrap();
    let restored = model.forward(&manager, &tokens).unwrap();
    let exact_restore = restored.values() == before_values.as_slice();

    println!("NOWORODEK REAL TRANSFORMER WEIGHT SURGERY V1");
    println!("architecture=noworodek-decoder-v0 vocab={} hidden={} intermediate={} layers={}", vocab, hidden, intermediate, layers);
    println!("surgery_tensor=model.layers.00.attention.q_proj.weight");
    println!("edited_element[0] += 1.0");
    println!("changed_logits={} max_abs_logit_delta={:.9}", changed, max_delta);
    println!("restore_exact={}", exact_restore);
    println!("baseline_logits_first_row={:?}", &before_values[..vocab]);
    println!("edited_logits_first_row={:?}", &after.values()[..vocab]);
}
