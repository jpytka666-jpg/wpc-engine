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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SequenceError {
    Gap { expected: usize, actual: usize },
    Overlap { expected: usize, actual: usize },
    InvalidRange { start: usize, end: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidencyMetrics {
    pub entries: usize,
    pub sequence_length: usize,
    pub payload_bytes: usize,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct HotKvBuffer {
    next_sequence: usize,
    payload_bytes: usize,
    entries: Vec<Vec<u8>>,
}

impl HotKvBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn next_sequence(&self) -> usize {
        self.next_sequence
    }

    /// Return deterministic, allocation-free residency counters for the hot layer.
    pub fn residency_metrics(&self) -> ResidencyMetrics {
        ResidencyMetrics {
            entries: self.entries.len(),
            sequence_length: self.next_sequence,
            payload_bytes: self.payload_bytes,
        }
    }

    /// Append an owned contiguous sequence range. The caller must start exactly
    /// at the next unowned sequence position; gaps and overlaps are rejected.
    pub fn append(
        &mut self,
        sequence_start: usize,
        entries: Vec<Vec<u8>>,
    ) -> Result<(), SequenceError> {
        if entries.is_empty() {
            return Err(SequenceError::InvalidRange {
                start: sequence_start,
                end: sequence_start,
            });
        }

        if sequence_start > self.next_sequence {
            return Err(SequenceError::Gap {
                expected: self.next_sequence,
                actual: sequence_start,
            });
        }

        if sequence_start < self.next_sequence {
            return Err(SequenceError::Overlap {
                expected: self.next_sequence,
                actual: sequence_start,
            });
        }

        self.payload_bytes += entries.iter().map(Vec::len).sum::<usize>();
        self.next_sequence += entries.len();
        self.entries.extend(entries);
        Ok(())
    }

    /// Read an owned half-open sequence range [start, end).
    pub fn read(&self, start: usize, end: usize) -> Result<Vec<Vec<u8>>, SequenceError> {
        if start > end || end > self.next_sequence {
            return Err(SequenceError::InvalidRange { start, end });
        }
        Ok(self.entries[start..end].to_vec())
    }
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
    snapshot.get("model_fingerprint").and_then(Value::as_str) == Some(model_fingerprint)
        && snapshot.get("config_fingerprint").and_then(Value::as_str) == Some(config_fingerprint)
}

/// Serialize and deserialize a snapshot deterministically through the module boundary.
pub fn snapshot_round_trip(snapshot: &Value) -> Value {
    let encoded = serde_json::to_vec(snapshot).expect("snapshot must be serializable JSON");
    serde_json::from_slice(&encoded).expect("module must decode its own snapshot format")
}

#[cfg(test)]
mod tests {
    use super::{
        envelope_is_compatible, is_compatible, snapshot_round_trip, HotKvBuffer, KvEncoding,
        KvEnvelope, ResidencyMetrics, SequenceError,
    };
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
        assert!(envelope_is_compatible(
            &envelope,
            "model-12345678",
            "session-1"
        ));
        assert!(!envelope_is_compatible(
            &envelope,
            "model-87654321",
            "session-1"
        ));
        assert!(!envelope_is_compatible(
            &envelope,
            "model-12345678",
            "session-2"
        ));
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

    #[test]
    fn append_assigns_contiguous_sequence_ownership() {
        let mut buffer = HotKvBuffer::new();
        buffer
            .append(0, vec![vec![1], vec![2]])
            .expect("first batch");
        buffer
            .append(2, vec![vec![3], vec![4]])
            .expect("second batch");
        assert_eq!(buffer.next_sequence(), 4);
        assert_eq!(
            buffer.read(1, 4).expect("read owned range"),
            vec![vec![2], vec![3], vec![4]]
        );
    }

    #[test]
    fn append_rejects_gaps_and_overlaps() {
        let mut buffer = HotKvBuffer::new();
        buffer
            .append(0, vec![vec![1], vec![2]])
            .expect("initial batch");

        assert_eq!(
            buffer.append(4, vec![vec![5]]),
            Err(SequenceError::Gap {
                expected: 2,
                actual: 4
            })
        );
        assert_eq!(
            buffer.append(1, vec![vec![9]]),
            Err(SequenceError::Overlap {
                expected: 2,
                actual: 1
            })
        );
    }

    #[test]
    fn read_rejects_unowned_range() {
        let mut buffer = HotKvBuffer::new();
        buffer.append(0, vec![vec![1]]).expect("initial batch");
        assert_eq!(
            buffer.read(0, 2),
            Err(SequenceError::InvalidRange { start: 0, end: 2 })
        );
    }

    #[test]
    fn residency_metrics_match_hot_payload() {
        let mut buffer = HotKvBuffer::new();
        assert_eq!(
            buffer.residency_metrics(),
            ResidencyMetrics {
                entries: 0,
                sequence_length: 0,
                payload_bytes: 0
            }
        );

        buffer
            .append(0, vec![vec![1, 2], vec![3, 4, 5]])
            .expect("append payload");
        assert_eq!(
            buffer.residency_metrics(),
            ResidencyMetrics {
                entries: 2,
                sequence_length: 2,
                payload_bytes: 5
            }
        );
    }
}
