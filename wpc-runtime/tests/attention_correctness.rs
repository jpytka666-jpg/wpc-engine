use rand::Rng;
use wpc_runtime::forward_batch::{BatchEngine, KvLayer, MmapF32};

fn flatten(xs: &[Vec<f32>]) -> Vec<f32> {
    xs.iter().flat_map(|row| row.iter().copied()).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn optimized_attention_matches_reference_across_shapes() {
    let mut rng = rand::thread_rng();
    for &(dim, batch, past) in &[(8, 1, 0), (16, 2, 3), (32, 4, 7), (64, 3, 10)] {
        let total = past + batch;
        let k: Vec<f32> = (0..total * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let v: Vec<f32> = (0..total * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let q: Vec<f32> = (0..batch * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let q_rows: Vec<Vec<f32>> = (0..batch)
            .map(|i| q[i * dim..(i + 1) * dim].to_vec())
            .collect();

        let mut kv = KvLayer::with_capacity(dim, total).unwrap();
        kv.append_batch(&k, &v, total).unwrap();
        let engine = BatchEngine::new(dim);

        let reference = flatten(&engine.reference_attention_batch(&q_rows, &kv).unwrap());
        let optimized = engine
            .optimized_attention_from_kv(&q, &kv, batch, past)
            .unwrap();

        let max_diff = max_abs_diff(&reference, &optimized);
        assert!(
            max_diff < 1e-4,
            "dim={dim} batch={batch} past={past}, max diff={max_diff}"
        );
    }
}

#[test]
fn earlier_query_does_not_see_future_batch_token() {
    let dim = 4;
    let batch = 2;
    let past = 1;
    let total = past + batch;

    let k = vec![0.0; total * dim];
    let mut v_a = vec![0.0; total * dim];
    let mut v_b = vec![0.0; total * dim];
    v_a[(past + 1) * dim..(past + 2) * dim].fill(1.0);
    v_b[(past + 1) * dim..(past + 2) * dim].fill(100.0);
    let q = vec![0.0; batch * dim];

    let mut kv_a = KvLayer::with_capacity(dim, total).unwrap();
    let mut kv_b = KvLayer::with_capacity(dim, total).unwrap();
    kv_a.append_batch(&k, &v_a, total).unwrap();
    kv_b.append_batch(&k, &v_b, total).unwrap();
    let engine = BatchEngine::new(dim);

    let out_a = engine
        .optimized_attention_from_kv(&q, &kv_a, batch, past)
        .unwrap();
    let out_b = engine
        .optimized_attention_from_kv(&q, &kv_b, batch, past)
        .unwrap();

    let first_a = &out_a[..dim];
    let first_b = &out_b[..dim];
    assert!(max_abs_diff(first_a, first_b) < 1e-6);
}

#[test]
fn mmap_and_kv_reallocation_preserve_all_rows() {
    let dim = 4;
    let mut kv = KvLayer::with_capacity(dim, 1).unwrap();
    let mut expected_keys = Vec::new();
    let mut expected_values = Vec::new();

    for row in 0..10 {
        let keys = vec![row as f32; dim];
        let values = vec![(row as f32) * 10.0; dim];
        expected_keys.push(keys.clone());
        expected_values.push(values.clone());
        kv.append_batch(&keys, &values, 1).unwrap();
    }

    assert_eq!(kv.seq_len, 10);
    for row in 0..10 {
        assert_eq!(kv.get_key_row(row).unwrap(), expected_keys[row].as_slice());
        assert_eq!(
            kv.get_value_row(row).unwrap(),
            expected_values[row].as_slice()
        );
    }
}

#[test]
fn mmap_direct_reallocation_preserves_data() {
    let mut map = MmapF32::new(4).unwrap();
    map.as_mut_slice().copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    map.mark_used(4).unwrap();
    map.ensure_capacity(8).unwrap();
    assert_eq!(&map.as_slice()[..4], &[1.0, 2.0, 3.0, 4.0]);

    map.as_mut_slice()[4..8].copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);
    map.mark_used(8).unwrap();
    map.ensure_capacity(32).unwrap();
    assert_eq!(
        &map.as_slice()[..8],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );
}

#[test]
fn mmap_mark_used_rejects_over_capacity() {
    let mut map = MmapF32::new(4).unwrap();
    assert!(map.mark_used(5).is_err());
}

#[test]
fn invalid_attention_inputs_are_rejected() {
    let engine = BatchEngine::new(8);
    let kv = KvLayer::with_capacity(8, 3).unwrap();

    assert!(engine
        .optimized_attention_from_kv(&vec![0.0; 7], &kv, 1, 2)
        .is_err());
    assert!(engine
        .optimized_attention_batch(&vec![0.0; 8], kv.keys_ptr(), kv.vals_ptr(), 1, 3, 3)
        .is_err());
}

#[test]
fn medium_stress_case_produces_finite_outputs() {
    let dim = 256;
    let batch = 16;
    let past = 512;
    let total = past + batch;
    let mut rng = rand::thread_rng();

    let k: Vec<f32> = (0..total * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let v: Vec<f32> = (0..total * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let q: Vec<f32> = (0..batch * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let mut kv = KvLayer::with_capacity(dim, total).unwrap();
    kv.append_batch(&k, &v, total).unwrap();
    let engine = BatchEngine::new(dim);

    let out = engine
        .optimized_attention_from_kv(&q, &kv, batch, past)
        .unwrap();
    assert_eq!(out.len(), batch * dim);
    assert!(out.iter().all(|x| x.is_finite()));
}
