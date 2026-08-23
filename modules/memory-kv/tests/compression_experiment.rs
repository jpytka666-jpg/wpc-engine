use aions_memory_kv::{CompressionExperiment, CompressionInput, CompressionResult};

#[test]
fn compression_experiment_is_explicitly_outside_generation_path() {
    let input = CompressionInput::new("session-1", 1024, 4096);
    let result = CompressionExperiment::run(input).expect("experiment should produce a result");

    assert!(!result.generation_critical);
    assert!(result.original_bytes > 0);
    assert!(result.compressed_bytes > 0);
    assert!(result.compression_ratio > 0.0);
}

#[test]
fn compression_result_round_trips_without_changing_policy() {
    let result = CompressionResult {
        session_id: "session-2".into(),
        original_bytes: 8192,
        compressed_bytes: 4096,
        compression_ratio: 2.0,
        generation_critical: false,
    };

    let encoded = serde_json::to_vec(&result).expect("encode result");
    let decoded: CompressionResult = serde_json::from_slice(&encoded).expect("decode result");

    assert_eq!(decoded, result);
    assert!(!decoded.generation_critical);
}

#[test]
fn compression_experiment_rejects_zero_sized_input() {
    let input = CompressionInput::new("session-3", 0, 4096);
    assert!(CompressionExperiment::run(input).is_err());
}
