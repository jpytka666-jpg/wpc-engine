use noworodek::{ArchitectureId, DType, ExternalTransformer, MemoryWeightBackend, ParameterHandle, Tensor, TensorSpec, TinyTransformerConfig, WeightSetHeader, WeightSetId, WeightSetManager, WeightSetManifest, WeightSetVersion};

#[derive(Debug, Clone)]
struct Influence {
    tensor: &'static str,
    shape: Vec<usize>,
    index: usize,
    delta: f32,
    max_delta: f32,
    changed: usize,
}

fn tensor_values(len: usize, scale: f32, phase: f32) -> Vec<f32> {
    (0..len).map(|i| scale * (((i as f32) + phase) * 0.37).sin()).collect()
}

fn build_fixture() -> (WeightSetManager, ExternalTransformer, WeightSetId) {
    let vocab = 8usize;
    let hidden = 4usize;
    let intermediate = 8usize;
    let specs = vec![
        TensorSpec::new("model.embeddings.token.weight", vec![vocab, hidden], DType::F32, "influence-scan"),
        TensorSpec::new("model.layers.00.attention.q_proj.weight", vec![hidden, hidden], DType::F32, "influence-scan"),
        TensorSpec::new("model.layers.00.attention.k_proj.weight", vec![hidden, hidden], DType::F32, "influence-scan"),
        TensorSpec::new("model.layers.00.attention.v_proj.weight", vec![hidden, hidden], DType::F32, "influence-scan"),
        TensorSpec::new("model.layers.00.attention.o_proj.weight", vec![hidden, hidden], DType::F32, "influence-scan"),
        TensorSpec::new("model.layers.00.mlp.up_proj.weight", vec![hidden, intermediate], DType::F32, "influence-scan"),
        TensorSpec::new("model.layers.00.mlp.gate_proj.weight", vec![hidden, intermediate], DType::F32, "influence-scan"),
        TensorSpec::new("model.layers.00.mlp.down_proj.weight", vec![intermediate, hidden], DType::F32, "influence-scan"),
        TensorSpec::new("model.layers.00.attention_norm.weight", vec![hidden], DType::F32, "influence-scan"),
        TensorSpec::new("model.layers.00.mlp_norm.weight", vec![hidden], DType::F32, "influence-scan"),
        TensorSpec::new("model.final_norm.weight", vec![hidden], DType::F32, "influence-scan"),
        TensorSpec::new("model.lm_head.weight", vec![vocab, hidden], DType::F32, "influence-scan"),
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
        WeightSetHeader::new(WeightSetId::new("transformer-influence-v2"), WeightSetVersion::new("0.2.0").unwrap(), ArchitectureId::new("noworodek-decoder-v0")),
        specs,
    ).unwrap();
    let backend = MemoryWeightBackend::with_tensor_data(manifest, names);
    let mut manager = WeightSetManager::new(ArchitectureId::new("noworodek-decoder-v0"));
    let weight_set = manager.mount(Box::new(backend)).unwrap();
    let model = ExternalTransformer::new(TinyTransformerConfig {
        vocab_size: vocab,
        hidden_size: hidden,
        intermediate_size: intermediate,
        sequence_length: 8,
        num_layers: 1,
        rms_norm_eps: 1e-5,
    }, weight_set.clone());
    (manager, model, weight_set)
}

fn is_internal(tensor: &str) -> bool {
    !tensor.starts_with("model.embeddings.") && tensor != "model.lm_head.weight"
}

fn index_coords(shape: &[usize], index: usize) -> String {
    if shape.len() == 1 { return format!("{}", index); }
    if shape.len() == 2 { return format!("{},{}", index / shape[1], index % shape[1]); }
    format!("index={index}")
}

fn scan_delta(
    manager: &mut WeightSetManager,
    model: &ExternalTransformer,
    weight_set: &WeightSetId,
    tokens: &[usize],
    name: &'static str,
    shape: &[usize],
    delta: f32,
    baseline: &[f32],
) -> Vec<Influence> {
    let handle = ParameterHandle::new(weight_set.clone(), name).unwrap();
    let original = handle.read(manager).unwrap().values().to_vec();
    let mut results = Vec::with_capacity(original.len());
    for index in 0..original.len() {
        let mut edited = original.clone();
        edited[index] += delta;
        handle.write(manager, &Tensor::from_vec(shape.to_vec(), edited).unwrap()).unwrap();
        let out = model.forward(manager, tokens).unwrap();
        let mut max_delta = 0.0_f32;
        let mut changed = 0usize;
        for (a, b) in baseline.iter().zip(out.values()) {
            let d = (*a - *b).abs();
            if d > 1e-8 { changed += 1; }
            max_delta = max_delta.max(d);
        }
        results.push(Influence { tensor: name, shape: shape.to_vec(), index, delta, max_delta, changed });
        handle.write(manager, &Tensor::from_vec(shape.to_vec(), original.clone()).unwrap()).unwrap();
    }
    results
}

fn run_scan(delta: f32, internal_only: bool) -> Vec<Influence> {
    let tokens = [0usize, 1, 2];
    let (mut manager, model, weight_set) = build_fixture();
    let baseline = model.forward(&manager, &tokens).unwrap().values().to_vec();
    let tensors: [(&'static str, Vec<usize>); 12] = [
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
    let mut all = Vec::new();
    for (name, shape) in tensors {
        if internal_only && !is_internal(name) { continue; }
        all.extend(scan_delta(&mut manager, &model, &weight_set, &tokens, name, &shape, delta, &baseline));
    }
    all
}

fn print_top(title: &str, results: &[Influence], limit: usize) {
    println!("{title}");
    println!("rank,tensor,index,coords,delta,max_abs_logit_delta,influence_per_weight,changed_logits");
    for (rank, r) in results.iter().take(limit).enumerate() {
        println!("{},{},{},{},{},{:.9},{:.9},{}", rank + 1, r.tensor, r.index, index_coords(&r.shape, r.index), r.delta, r.max_delta, r.max_delta / r.delta.abs(), r.changed);
    }
}

fn main() {
    let delta = 0.1_f32;
    let full = run_scan(delta, false);
    let mut full_sorted = full.clone();
    full_sorted.sort_by(|a, b| b.max_delta.total_cmp(&a.max_delta));
    let mut internal = run_scan(delta, true);
    internal.sort_by(|a, b| b.max_delta.total_cmp(&a.max_delta));

    println!("NOWORODEK TRANSFORMER INFLUENCE SCAN V2");
    println!("delta={} full_elements={} internal_elements={}", delta, full.len(), internal.len());
    print_top("TOP FULL MODEL", &full_sorted, 20);
    print_top("TOP INTERNAL ONLY (embedding/lm_head excluded)", &internal, 20);

    let mut deltas = Vec::new();
    for d in [0.01_f32, 0.1_f32, 1.0_f32] {
        let mut r = run_scan(d, true);
        r.sort_by(|a, b| b.max_delta.total_cmp(&a.max_delta));
        if let Some(top) = r.first() {
            deltas.push((d, top.tensor, top.index, top.max_delta, top.max_delta / d));
        }
    }
    println!("DELTA SENSITIVITY INTERNAL TOP-1");
    println!("delta,tensor,index,max_abs_logit_delta,influence_per_weight");
    for (d, t, i, md, inf) in deltas { println!("{d},{t},{i},{md:.9},{inf:.9}"); }
}
