use rand::{rngs::StdRng, Rng, SeedableRng};
use wpc_runtime::forward_batch::{BatchEngine, KvLayer, MmapF32};

fn flatten(rows: &[Vec<f32>]) -> Vec<f32> {
    rows.iter().flat_map(|row| row.iter().copied()).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn make_case(dim: usize, batch: usize, past: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>, KvLayer) {
    let total = past + batch;
    let mut rng = StdRng::seed_from_u64(seed);
    let k: Vec<f32> = (0..total * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let v: Vec<f32> = (0..total * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let q: Vec<f32> = (0..batch * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let mut kv = KvLayer::with_capacity(dim, total).unwrap();
    kv.append_batch(&k, &v, total).unwrap();
    (q, k, v, kv)
}

#[test]
fn optimized_attention_matches_reference_across_shapes() {
    for (case_id, &(dim, batch, past)) in [
        (1usize, 1usize, 0usize),
        (7, 1, 3),
        (7, 2, 3),
        (32, 4, 0),
        (64, 3, 10),
        (128, 8, 16),
    ]
    .iter()
    .enumerate()
    {
        let (q, _, _, kv) = make_case(dim, batch, past, 0xA11CE + case_id as u64);
        let q_rows: Vec<Vec<f32>> = (0..batch)
            .map(|i| q[i * dim..(i + 1) * dim].to_vec())
            .collect();
        let engine = BatchEngine::new(dim);
        let reference = flatten(&engine.reference_attention_batch(&q_rows, &kv).unwrap());
        let optimized = engine
            .optimized_attention_from_kv(&q, &kv, batch, past)
            .unwrap();

        let max_diff = max_abs_diff(&reference, &optimized);
        assert!(
            max_diff < 1e-4,
            "case {case_id}: dim={dim} batch={batch} past={past} max_diff={max_diff}"
        );
    }
}

#[test]
fn earlier_query_does_not_see_future_batch_token() {
    let dim = 16;
    let batch = 2;
    let past = 2;
    let total = past + batch;

    let q = vec![0.25f32; batch * dim];
    let k = vec![0.0f32; total * dim];
    let mut v = vec![0.0f32; total * dim];

    for x in &mut v[0..dim] {
        *x = 1.0;
    }
    for x in &mut v[dim..2 * dim] {
        *x = 2.0;
    }
    for x in &mut v[2 * dim..3 * dim] {
        *x = 3.0;
    }
    for x in &mut v[3 * dim..4 * dim] {
        *x = 1000.0;
    }

    let mut kv = KvLayer::with_capacity(dim, total).unwrap();
    kv.append_batch(&k, &v, total).unwrap();

    let engine = BatchEngine::new(dim);
    let out_a = engine
        .optimized_attention_from_kv(&q, &kv, batch, past)
        .unwrap();

    let mut v_changed = v.clone();
    for x in &mut v_changed[3 * dim..4 * dim] {
        *x = -5000.0;
    }
    let mut kv_changed = KvLayer::with_capacity(dim, total).unwrap();
    kv_changed.append_batch(&k, &v_changed, total).unwrap();
    let out_b = engine
        .optimized_attention_from_kv(&q, &kv_changed, batch, past)
        .unwrap();

    let first_a = &out_a[0..dim];
    let first_b = &out_b[0..dim];
    assert!(max_abs_diff(first_a, first_b) < 1e-5);
}

#[test]
fn mmap_and_kv_reallocation_preserve_all_rows() {
    let dim = 4;
    let mut kv = KvLayer::with_capacity(dim, 1).unwrap();

    let expected_keys: Vec<Vec<f32>> = (0..10)
        .map(|row| (0..dim).map(|j| (row * dim + j) as f32).collect())
        .collect();
    let expected_values: Vec<Vec<f32>> = (0..10)
        .map(|row| (0..dim).map(|j| 1000.0 + (row * dim + j) as f32).collect())
        .collect();

    for row in 0..10 {
        kv.append_batch(&expected_keys[row], &expected_values[row], 1).unwrap();
    }

    assert_eq!(kv.seq_len, 10);
    for row in 0..10 {
        assert_eq!(kv.get_key_row(row).unwrap(), expected_keys[row].as_slice());
        assert_eq!(kv.get_value_row(row).unwrap(), expected_values[row].as_slice());
    }
}

#[test]
fn mmap_direct_reallocation_preserves_data() {
    let mut map = MmapF32::new(4).unwrap();
    map.as_mut_slice().copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    map.ensure_capacity(8).unwrap();
    assert_eq!(&map.as_slice()[..4], &[1.0, 2.0, 3.0, 4.0]);
    map.as_mut_slice()[4..8].copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);
    map.ensure_capacity(32).unwrap();
    assert_eq!(&map.as_slice()[..8], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn invalid_attention_inputs_are_rejected() {
    let engine = BatchEngine::new(8);
    let kv = KvLayer::with_capacity(8, 3).unwrap();

    assert!(engine
        .optimized_attention_from_kv(&vec![0.0; 7], &kv, 1, 2)
        .is_err());
    assert!(engine
        .optimized_attention_batch(&vec![0.0; 8], kv.keys_ptr(), kv.vals_ptr(), 1, 3, 1)
        .is_err());
}

#[test]
fn medium_stress_case_produces_finite_outputs() {
    let dim = 256;
    let batch = 16;
    let past = 512;
    let (q, _, _, kv) = make_case(dim, batch, past, 0x5157);
    let engine = BatchEngine::new(dim);
    let out = engine
        .optimized_attention_from_kv(&q, &kv, batch, past)
        .unwrap();

    assert_eq!(out.len(), batch * dim);
    assert!(out.iter().all(|x| x.is_finite()));
}
