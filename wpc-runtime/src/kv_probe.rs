// AIONS KV probe
// 2026-08-24 — maintained by ChatGPT in this session.
// Reason: activate the already-defined read-only KV observation contract for
// real-model Qwen statistics, without changing the normal generation path.

use std::env;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Read-only observation hook for K/V states produced by the Gemma/Qwen runtime.
///
/// The hook is disabled by default and therefore adds no callback work to the
/// normal generation path. Implementations may copy or compress the borrowed
/// vectors, but they must not mutate them.
pub trait KvProbe: Send + Sync {
    fn observe(&self, layer: usize, position: usize, kv_head: usize, key: &[f32], value: &[f32]);
}

pub type KvProbeHandle = Arc<dyn KvProbe>;

/// Bounded read-only resident-KV statistics collector.
///
/// When `AIONS_KV_PROBE=1` is enabled by the runtime, every K/V vector is
/// counted and its raw f32 byte volume is accumulated exactly. Value samples
/// are scanned only at the configured stride (default 64), keeping the probe
/// cheap enough to leave attached during a real generation run.
///
/// The probe writes no files and never mutates the resident cache. Final
/// statistics are emitted to stderr when the probe is dropped.
pub struct StatsKvProbe {
    calls: AtomicU64,
    raw_bytes: AtomicU64,
    positions: AtomicU64,
    sampled_values: AtomicU64,
    stride: usize,
    stats: Mutex<SampleStats>,
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
    fn observe(
        &self,
        _layer: usize,
        position: usize,
        _kv_head: usize,
        key: &[f32],
        value: &[f32],
    ) {
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
    }
}

impl Drop for StatsKvProbe {
    fn drop(&mut self) {
        let calls = self.calls.load(Ordering::Relaxed);
        let positions = self.positions.load(Ordering::Relaxed);
        let raw_bytes = self.raw_bytes.load(Ordering::Relaxed);
        let sampled = self.sampled_values.load(Ordering::Relaxed);
        let stats = self.stats.lock().expect("KV probe stats mutex poisoned");

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
