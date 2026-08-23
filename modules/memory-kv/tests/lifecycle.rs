use aions_memory_kv::{is_compatible, snapshot_round_trip};
use serde_json::json;

#[test]
fn compatible_hot_snapshot_round_trips() {
    let snapshot = json!({
        "session_id": "session-1",
        "model_fingerprint": "model-a",
        "config_fingerprint": "config-a",
        "layer": 2,
        "sequence_start": 8,
        "sequence_end": 11,
        "residency": "hot",
        "generation_critical": true,
        "dtype": "f32",
        "payload": [1, 2, 3, 4]
    });

    assert_eq!(snapshot_round_trip(&snapshot), snapshot);
}

#[test]
fn incompatible_model_fingerprint_is_rejected() {
    let snapshot = json!({
        "model_fingerprint": "model-a",
        "config_fingerprint": "config-a"
    });

    assert!(!is_compatible(&snapshot, "model-b", "config-a"));
}
