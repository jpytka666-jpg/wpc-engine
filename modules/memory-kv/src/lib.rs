use serde_json::Value;

/// Check whether a persisted or warm KV snapshot belongs to the expected model and configuration.
pub fn is_compatible(snapshot: &Value, model_fingerprint: &str, config_fingerprint: &str) -> bool {
    snapshot
        .get("model_fingerprint")
        .and_then(Value::as_str)
        == Some(model_fingerprint)
        && snapshot
            .get("config_fingerprint")
            .and_then(Value::as_str)
            == Some(config_fingerprint)
}

/// Serialize and deserialize a snapshot deterministically through the module boundary.
pub fn snapshot_round_trip(snapshot: &Value) -> Value {
    let encoded = serde_json::to_vec(snapshot).expect("snapshot must be serializable JSON");
    serde_json::from_slice(&encoded).expect("module must decode its own snapshot format")
}

#[cfg(test)]
mod tests {
    use super::{is_compatible, snapshot_round_trip};
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
}
