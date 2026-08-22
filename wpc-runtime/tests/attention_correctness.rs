use rand::Rng;
use wpc_runtime::forward_batch::{BatchEngine, KvLayer, MmapF32};

fn flatten(xs: &[Vec<f32>]) -> Vec<f32> {
    xs.iter().flat_map(|row| row.iter().copied()).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn optimized_attention_matches_reference() {
    let dim = 64;
    let batch = 3;
    let past = 10;
    let total = past + batch;
    let mut rng = rand::thread_rng();

    let k: Vec<f32> = (0..total * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let v: Vec<f32> = (0..total * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let q: Vec<f32> = (0..batch * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let q_rows: Vec<Vec<f32>> = (0..batch).map(|i| q[i * dim..(i + 1) * dim].to_vec()).collect();

    let mut kv = KvLayer::with_capacity(dim, total).unwrap();
    kv.append_batch(&k, &v, total).unwrap();
    let engine = BatchEngine::new(dim);

    let reference = flatten(&engine.reference_attention_batch(&q_rows, &kv).unwrap());
    let optimized = engine
        .optimized_attention_batch(&q, kv.keys_ptr(), kv.vals_ptr(), batch, total, past)
        .unwrap();

    let max_diff = max_abs_diff(&reference, &optimized);
    assert!(max_diff < 1e-4, "max abs diff = {max_diff}");
}

#[test]
fn mmap_reallocation_preserves_data() {
    let mut map = MmapF32::new(4).unwrap();
    map.as_mut_slice().copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    map.ensure_capacity(8).unwrap();
    assert_eq!(&map.as_slice()[..4], &[1.0, 2.0, 3.0, 4.0]);
}
