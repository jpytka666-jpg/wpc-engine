// AIONS Qwen KV → Memory/KV bridge
// AUTHOR: M. SZUL
// AI MODEL: GPT-5.6 Luna
// TIMESTAMP: 2026-08-24 21:42:00 +01:00
// REASON FOR CREATION: Reuse the existing aions-memory-kv envelope contract for Qwen3-MoE resident-KV snapshots.
// MECHANICS: Writes a typed KvEnvelope sidecar that references the existing AIONSKV1 payload; never copies or rewrites model weights.
// SYSTEM PART: WPC runtime ↔ Memory/KV persistence boundary
// ARCHITECTURE FUNCTION: Connects hot resident KV capture to the canonical AIONS Memory/KV metadata contract.
// DEPENDENCIES/LINKS: aions-memory-kv::KvEnvelope, Qwen StatsKvProbe AIONSKV1 payload
// TECH STACK: Rust 2021; selected because wpc-runtime and memory-kv are Rust crates sharing the workspace contract.
// LOCAL WORKSPACE: C:\\temp\\aions-qwen-kv-probe-rebuild-2026-08-24
// GIT COMMIT: PENDING
// GITHUB METADATA: jpytka666-jpg/wpc-engine / feature/memory-kv-real-model-bridge

use aions_memory_kv::{KvEncoding, KvEnvelope};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

pub fn write_envelope_sidecar(
    payload_path: impl AsRef<Path>,
    dimension: usize,
    sequence_length: usize,
) -> io::Result<PathBuf> {
    let model_fingerprint = std::env::var("AIONS_KV_MODEL_FINGERPRINT").map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "AIONS_KV_MODEL_FINGERPRINT is required",
        )
    })?;
    let session_id = std::env::var("AIONS_KV_SESSION_ID").map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "AIONS_KV_SESSION_ID is required",
        )
    })?;

    let payload_path = payload_path.as_ref();
    let envelope = KvEnvelope {
        model_fingerprint,
        session_id,
        dimension,
        sequence_length,
        encoding: KvEncoding::F32,
        payload_ref: Some(payload_path.to_string_lossy().into_owned()),
    };

    let mut sidecar = payload_path.to_path_buf();
    sidecar.set_extension("envelope.json");
    let bytes = serde_json::to_vec_pretty(&envelope)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    fs::write(&sidecar, bytes)?;
    Ok(sidecar)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_sidecar_uses_canonical_memory_kv_type() {
        let envelope = KvEnvelope {
            model_fingerprint: "qwen3-coder-30b-a3b:wpc-v4".into(),
            session_id: "session-test".into(),
            dimension: 128,
            sequence_length: 4,
            encoding: KvEncoding::F32,
            payload_ref: Some("capture.aionskv".into()),
        };
        let json = serde_json::to_string(&envelope).expect("envelope serializes");
        let decoded: KvEnvelope = serde_json::from_str(&json).expect("envelope decodes");
        assert_eq!(decoded, envelope);
    }
}
