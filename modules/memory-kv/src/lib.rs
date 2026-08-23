use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KvEncoding {
    F32,
    F16,
    Bf16,
    Wpc,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvEnvelope {
    pub model_fingerprint: String,
    pub session_id: String,
    pub dimension: usize,
    pub sequence_length: usize,
    pub encoding: KvEncoding,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload_ref: Option<String>,
}

/// Check whether an envelope belongs to the expected model and active session.
pub fn envelope_is_compatible(
    envelope: &KvEnvelope,
    model_fingerprint: &str,
    session_id: &str,
) -> bool {
    envelope.model_fingerprint == model_fingerprint && envelope.session_id == session_id
}

/// Backward-compatible compatibility check for JSON snapshots.
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
    use super::{envelope_is_compatible, is_compatible, snapshot_round_trip, KvEncoding, KvEnvelope};
    use serde_json::json;

    #[test]
    fn typed_envelope_round_trips() {
        let envelope = KvEnvelope {
            model_fingerprint: "model-12345678".into(),
            session_id: "session-1".into(),
            dimension: 4096,
            sequence_length: 128,
            encoding: KvEncoding::F16,
            payload_ref: Some("hot:session-1:layer-0".into()),
        };
        let encoded = serde_json::to_vec(&envelope).expect("encode envelope");
        let decoded: KvEnvelope = serde_json::from_slice(&encoded).expect("decode envelope");
        assert_eq!(decoded, envelope);
    }

    #[test]
    fn typed_envelope_rejects_wrong_model_or_session() {
        let envelope = KvEnvelope {
            model_fingerprint: "model-12345678".into(),
            session_id: "session-1".into(),
            dimension: 4096,
            sequence_length: 128,
            encoding: KvEncoding::F16,
            payload_ref: None,
        };
        assert!(envelope_is_compatible(&envelope, "model-12345678", "session-1"));
        assert!(!envelope_is_compatible(&envelope, "model-87654321", "session-1"));
        assert!(!envelope_is_compatible(&envelope, "model-12345678", "session-2"));
    }

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
