#[cfg(test)]
mod tests {
    use serde_json::Value;

    #[test]
    fn compatible_hot_snapshot_round_trips() {
        let snapshot = serde_json::json!({
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

        let encoded = serde_json::to_vec(&snapshot).expect("encode snapshot");
        let decoded: Value = serde_json::from_slice(&encoded).expect("decode snapshot");
        assert_eq!(decoded, snapshot);
    }

    #[test]
    fn incompatible_model_fingerprint_is_rejected() {
        let snapshot = serde_json::json!({
            "model_fingerprint": "model-a",
            "config_fingerprint": "config-a"
        });

        assert!(!is_compatible(&snapshot, "model-b", "config-a"));
    }

    fn is_compatible(snapshot: &Value, model: &str, config: &str) -> bool {
        snapshot.get("model_fingerprint").and_then(Value::as_str) == Some(model)
            && snapshot.get("config_fingerprint").and_then(Value::as_str) == Some(config)
    }
}
