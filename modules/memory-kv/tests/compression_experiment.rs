use aions_memory_kv::{run_wpc_kv, WpcKvInput};

fn synthetic_kv(tokens: usize, width: usize) -> (Vec<f32>, Vec<f32>) {
    let mut keys = Vec::with_capacity(tokens * width);
    let mut values = Vec::with_capacity(tokens * width);
    for token in 0..tokens {
        for dim in 0..width {
            let x = (token as f32 * 0.13 + dim as f32 * 0.07).sin();
            keys.push(x);
            values.push((x * 0.83 + 0.11).cos());
        }
    }
    (keys, values)
}

#[test]
fn wpc_kv_gate_reports_real_metrics() {
    let (keys, values) = synthetic_kv(128, 64);
    let metrics = run_wpc_kv(WpcKvInput {
        session_id: "compression-gate".into(),
        keys,
        values,
        vector_width: 64,
        pattern_count: 16,
        residual_count: 256,
        train_iters: 5,
    })
    .expect("WPC KV compression gate");

    assert!(!metrics.generation_critical);
    assert!(metrics.original_bytes_f16 > metrics.compressed_bytes);
    assert!(metrics.compression_ratio_vs_f16 > 1.0);
    assert!(metrics.key_rmse < 0.25, "key RMSE: {}", metrics.key_rmse);
    assert!(metrics.value_rmse < 0.25, "value RMSE: {}", metrics.value_rmse);
    assert!(
        metrics.attention_output_rmse < 0.25,
        "attention output RMSE: {}",
        metrics.attention_output_rmse
    );
}
