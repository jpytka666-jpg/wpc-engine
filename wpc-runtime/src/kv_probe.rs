// AIONS Qwen3-MoE KV probe
// 2026-08-24 — maintained by ChatGPT in this session.
// Reason: activate the already-defined read-only KV observation contract for
// real-model Qwen3-Coder-30B-A3B statistics and optionally persist a bounded
// resident-KV snapshot for the next reload/restore stage, without changing
// normal generation when capture is disabled.

use crate::kv_memory_bridge;
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

const SNAPSHOT_MAGIC: &[u8; 8] = b"AIONSKV1";
const SNAPSHOT_VERSION: u32 = 1;
const SNAPSHOT_FLAG_TRUNCATED: u32 = 1;
const DEFAULT_CAPTURE_MAX_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_CAPTURE_MAX_POSITIONS: usize = 128;

/// Read-only observation hook for K/V states produced by the Qwen3-MoE runtime.
///
/// The hook is disabled by default and therefore adds no callback work to the
/// normal generation path. Implementations may copy or compress the borrowed
/// vectors, but they must not mutate them.
pub trait KvProbe: Send + Sync {
    fn observe(&self, layer: usize, position: usize, kv_head: usize, key: &[f32], value: &[f32]);
    fn record_logits(&self, _logits: &[f32]) {}
}

pub type KvProbeHandle = Arc<dyn KvProbe>;

#[derive(Clone, Debug, PartialEq)]
pub struct KvSnapshotRecord {
    pub layer: u32,
    pub position: u64,
    pub kv_head: u32,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KvSnapshot {
    pub version: u32,
    pub truncated: bool,
    pub records: Vec<KvSnapshotRecord>,
}

impl KvSnapshot {
    pub fn read_from_path(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        decode_snapshot(&bytes)
            .map_err(|msg| std::io::Error::new(std::io::ErrorKind::InvalidData, msg))
    }
}

struct CaptureState {
    path: PathBuf,
    max_bytes: usize,
    max_positions: usize,
    bytes: usize,
    truncated: bool,
    records: Vec<KvSnapshotRecord>,
}

impl CaptureState {
    fn from_env() -> Option<Self> {
        let path = env::var("AIONS_KV_CAPTURE").ok()?;
        if path.trim().is_empty() {
            return None;
        }
        let max_bytes = env::var("AIONS_KV_CAPTURE_MAX_BYTES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(DEFAULT_CAPTURE_MAX_BYTES);
        let max_positions = env::var("AIONS_KV_CAPTURE_MAX_POSITIONS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(DEFAULT_CAPTURE_MAX_POSITIONS);
        Some(Self {
            path: PathBuf::from(path),
            max_bytes,
            max_positions,
            bytes: 0,
            truncated: false,
            records: Vec::new(),
        })
    }

    fn observe(
        &mut self,
        layer: usize,
        position: usize,
        kv_head: usize,
        key: &[f32],
        value: &[f32],
    ) {
        if self.truncated || position >= self.max_positions {
            self.truncated = true;
            return;
        }
        if key.len() != value.len() {
            self.truncated = true;
            return;
        }
        let payload_bytes = (key.len() + value.len()) * std::mem::size_of::<f32>();
        let record_bytes = 4 + 8 + 4 + 4 + payload_bytes;
        if self.bytes.saturating_add(record_bytes) > self.max_bytes {
            self.truncated = true;
            return;
        }
        self.bytes += record_bytes;
        self.records.push(KvSnapshotRecord {
            layer: layer as u32,
            position: position as u64,
            kv_head: kv_head as u32,
            key: key.to_vec(),
            value: value.to_vec(),
        });
    }

    fn flush(&self) -> std::io::Result<()> {
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let file = File::create(&self.path)?;
        let mut writer = BufWriter::new(file);
        let bytes = encode_snapshot(&KvSnapshot {
            version: SNAPSHOT_VERSION,
            truncated: self.truncated,
            records: self.records.clone(),
        });
        writer.write_all(&bytes)?;
        writer.flush()
    }
}

/// Bounded resident-KV statistics collector.
///
/// When `AIONS_KV_PROBE=1` is enabled by the Qwen runtime, every K/V vector is
/// counted and its raw f32 byte volume is accumulated exactly. Value samples
/// are scanned only at the configured stride (default 64), keeping the probe
/// cheap enough to leave attached during a real Qwen generation run.
///
/// Set `AIONS_KV_CAPTURE=<path>` to persist a bounded binary snapshot of the
/// observed resident K/V vectors. The capture is opt-in, capped by
/// `AIONS_KV_CAPTURE_MAX_BYTES` and `AIONS_KV_CAPTURE_MAX_POSITIONS`, and never
/// mutates the resident cache. Final statistics are emitted to stderr when the
/// probe is dropped.
pub struct StatsKvProbe {
    calls: AtomicU64,
    raw_bytes: AtomicU64,
    positions: AtomicU64,
    sampled_values: AtomicU64,
    stride: usize,
    stats: Mutex<SampleStats>,
    capture: Mutex<Option<CaptureState>>,
    final_logits: Mutex<Option<Vec<f32>>>,
}

#[derive(Default)]
struct SampleStats {
    sum: f64,
    sumsq: f64,
    min: f32,
    max: f32,
}

impl StatsKvProbe {
    pub fn from_env() -> KvProbeHandle {
        let stride = env::var("AIONS_KV_PROBE_STRIDE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(64);

        Arc::new(Self {
            calls: AtomicU64::new(0),
            raw_bytes: AtomicU64::new(0),
            positions: AtomicU64::new(0),
            sampled_values: AtomicU64::new(0),
            stride,
            stats: Mutex::new(SampleStats {
                min: f32::INFINITY,
                max: f32::NEG_INFINITY,
                ..Default::default()
            }),
            capture: Mutex::new(CaptureState::from_env()),
            final_logits: Mutex::new(None),
        })
    }

    fn observe_values(&self, values: &[f32]) {
        let mut stats = self.stats.lock().expect("KV probe stats mutex poisoned");
        for &v in values {
            stats.sum += v as f64;
            stats.sumsq += (v as f64) * (v as f64);
            stats.min = stats.min.min(v);
            stats.max = stats.max.max(v);
        }
        self.sampled_values
            .fetch_add(values.len() as u64, Ordering::Relaxed);
    }
}

impl KvProbe for StatsKvProbe {
    fn record_logits(&self, logits: &[f32]) {
        *self.final_logits.lock().expect("KV logits mutex poisoned") = Some(logits.to_vec());
    }

    fn observe(&self, layer: usize, position: usize, kv_head: usize, key: &[f32], value: &[f32]) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.raw_bytes.fetch_add(
            ((key.len() + value.len()) * std::mem::size_of::<f32>()) as u64,
            Ordering::Relaxed,
        );
        self.positions
            .fetch_max((position + 1) as u64, Ordering::Relaxed);

        if position % self.stride == 0 {
            self.observe_values(key);
            self.observe_values(value);
        }

        if let Some(capture) = self
            .capture
            .lock()
            .expect("KV capture mutex poisoned")
            .as_mut()
        {
            capture.observe(layer, position, kv_head, key, value);
        }
    }
}

impl Drop for StatsKvProbe {
    fn drop(&mut self) {
        if let Some(capture) = self
            .capture
            .get_mut()
            .expect("KV capture mutex poisoned")
            .take()
        {
            if let Err(err) = capture.flush() {
                eprintln!(
                    "kv-capture: failed to write {}: {err}",
                    capture.path.display()
                );
            } else {
                eprintln!(
                    "kv-capture: wrote {} records to {}{}",
                    capture.records.len(),
                    capture.path.display(),
                    if capture.truncated {
                        " (TRUNCATED)"
                    } else {
                        ""
                    }
                );

                if capture.truncated {
                    eprintln!("kv-memory: envelope suppressed because capture is truncated");
                } else {
                    let logits = self
                        .final_logits
                        .get_mut()
                        .expect("KV logits mutex poisoned")
                        .take();
                    if let Some(logits) = logits {
                        let mut lp = capture.path.clone();
                        lp.set_extension("logits.f32");
                        let mut bytes = Vec::with_capacity(logits.len() * 4);
                        for v in logits {
                            bytes.extend_from_slice(&v.to_le_bytes());
                        }
                        if let Err(err) = std::fs::write(&lp, bytes) {
                            eprintln!("kv-capture: failed to write {}: {err}", lp.display());
                        } else {
                            eprintln!("kv-capture: wrote final logits to {}", lp.display());
                        }
                    }
                }
                if !capture.truncated {
                    if let Some(first) = capture.records.first() {
                        let sequence_length = capture
                            .records
                            .iter()
                            .map(|record| record.position as usize + 1)
                            .max()
                            .unwrap_or(0);
                        match kv_memory_bridge::write_envelope_sidecar(
                            &capture.path,
                            first.key.len(),
                            sequence_length,
                        ) {
                            Ok(path) => eprintln!(
                                "kv-memory: canonical envelope written to {}",
                                path.display()
                            ),
                            Err(err) => {
                                eprintln!("kv-memory: canonical envelope not written: {err}")
                            }
                        }
                    }
                }
            }
        }

        let calls = self.calls.load(Ordering::Relaxed);
        let positions = self.positions.load(Ordering::Relaxed);
        let raw_bytes = self.raw_bytes.load(Ordering::Relaxed);
        let sampled = self.sampled_values.load(Ordering::Relaxed);
        let stats = self.stats.get_mut().expect("KV probe stats mutex poisoned");

        let mean = if sampled == 0 {
            0.0
        } else {
            stats.sum / sampled as f64
        };
        let variance = if sampled == 0 {
            0.0
        } else {
            (stats.sumsq / sampled as f64) - (mean * mean)
        };
        let rms = (variance.max(0.0) + mean * mean).sqrt();

        eprintln!(
            "kv-probe: calls={} positions={} raw_kv_bytes={} sampled_values={} sample_min={:.6} sample_max={:.6} sample_mean={:.6} sample_rms={:.6} stride={}",
            calls,
            positions,
            raw_bytes,
            sampled,
            stats.min,
            stats.max,
            mean,
            rms,
            self.stride
        );
    }
}

fn encode_snapshot(snapshot: &KvSnapshot) -> Vec<u8> {
    let mut out = Vec::with_capacity(32);
    out.extend_from_slice(SNAPSHOT_MAGIC);
    out.extend_from_slice(&snapshot.version.to_le_bytes());
    let flags = if snapshot.truncated {
        SNAPSHOT_FLAG_TRUNCATED
    } else {
        0
    };
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&(snapshot.records.len() as u64).to_le_bytes());
    for record in &snapshot.records {
        out.extend_from_slice(&record.layer.to_le_bytes());
        out.extend_from_slice(&record.position.to_le_bytes());
        out.extend_from_slice(&record.kv_head.to_le_bytes());
        out.extend_from_slice(&(record.key.len() as u32).to_le_bytes());
        for value in &record.key {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for value in &record.value {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    out
}

fn decode_snapshot(bytes: &[u8]) -> Result<KvSnapshot, String> {
    let mut cursor = 0usize;
    fn take<'a>(bytes: &'a [u8], cursor: &mut usize, n: usize) -> Result<&'a [u8], String> {
        let end = cursor
            .checked_add(n)
            .ok_or_else(|| "snapshot length overflow".to_string())?;
        let slice = bytes
            .get(*cursor..end)
            .ok_or_else(|| "truncated snapshot".to_string())?;
        *cursor = end;
        Ok(slice)
    }
    fn u32le(bytes: &[u8], cursor: &mut usize) -> Result<u32, String> {
        Ok(u32::from_le_bytes(
            take(bytes, cursor, 4)?
                .try_into()
                .map_err(|_| "u32 decode")?,
        ))
    }
    fn u64le(bytes: &[u8], cursor: &mut usize) -> Result<u64, String> {
        Ok(u64::from_le_bytes(
            take(bytes, cursor, 8)?
                .try_into()
                .map_err(|_| "u64 decode")?,
        ))
    }

    if take(bytes, &mut cursor, SNAPSHOT_MAGIC.len())? != SNAPSHOT_MAGIC {
        return Err("invalid AIONS KV snapshot magic".to_string());
    }
    let version = u32le(bytes, &mut cursor)?;
    let flags = u32le(bytes, &mut cursor)?;
    let count = u64le(bytes, &mut cursor)? as usize;
    let mut records = Vec::with_capacity(count);

    for _ in 0..count {
        let layer = u32le(bytes, &mut cursor)?;
        let position = u64le(bytes, &mut cursor)?;
        let kv_head = u32le(bytes, &mut cursor)?;
        let len = u32le(bytes, &mut cursor)? as usize;
        let mut key = Vec::with_capacity(len);
        let mut value = Vec::with_capacity(len);
        for _ in 0..len {
            key.push(f32::from_le_bytes(
                take(bytes, &mut cursor, 4)?
                    .try_into()
                    .map_err(|_| "key decode")?,
            ));
        }
        for _ in 0..len {
            value.push(f32::from_le_bytes(
                take(bytes, &mut cursor, 4)?
                    .try_into()
                    .map_err(|_| "value decode")?,
            ));
        }
        records.push(KvSnapshotRecord {
            layer,
            position,
            kv_head,
            key,
            value,
        });
    }

    if cursor != bytes.len() {
        return Err("snapshot has trailing bytes".to_string());
    }

    Ok(KvSnapshot {
        version,
        truncated: flags & SNAPSHOT_FLAG_TRUNCATED != 0,
        records,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_roundtrip_is_binary_stable() {
        let snapshot = KvSnapshot {
            version: SNAPSHOT_VERSION,
            truncated: true,
            records: vec![KvSnapshotRecord {
                layer: 47,
                position: 3,
                kv_head: 2,
                key: vec![1.25, -2.5, 3.75],
                value: vec![-4.0, 5.5, -6.25],
            }],
        };
        let bytes = encode_snapshot(&snapshot);
        let decoded = decode_snapshot(&bytes).expect("snapshot decodes");
        assert_eq!(decoded, snapshot);
        assert_eq!(encode_snapshot(&decoded), bytes);
    }

    #[test]
    fn snapshot_rejects_trailing_bytes() {
        let snapshot = KvSnapshot {
            version: SNAPSHOT_VERSION,
            truncated: false,
            records: Vec::new(),
        };
        let mut bytes = encode_snapshot(&snapshot);
        bytes.push(0xFF);
        assert!(decode_snapshot(&bytes).is_err());
    }
}
