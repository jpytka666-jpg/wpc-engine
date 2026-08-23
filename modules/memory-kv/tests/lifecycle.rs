use aions_memory_kv::{
    envelope_is_compatible, is_compatible, snapshot_round_trip, HotKvBuffer, KvEncoding,
    KvEnvelope, SequenceError,
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

#[test]
fn sequence_batches_are_owned_contiguously() {
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
fn sequence_gaps_and_overlaps_are_rejected() {
    let mut buffer = HotKvBuffer::new();
    buffer
        .append(0, vec![vec![1], vec![2]])
        .expect("initial batch");

    assert_eq!(
        buffer.append(4, vec![vec![5]]),
        Err(SequenceError::Gap {
            expected: 2,
            actual: 4,
        })
    );
    assert_eq!(
        buffer.append(1, vec![vec![9]]),
        Err(SequenceError::Overlap {
            expected: 2,
            actual: 1,
        })
    );
}
