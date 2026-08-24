use std::sync::Arc;

/// Read-only observation hook for K/V states produced by the Gemma runtime.
///
/// The hook is disabled by default and therefore adds no callback work to the
/// normal generation path. Implementations may copy or compress the borrowed
/// vectors, but they must not mutate them.
pub trait KvProbe: Send + Sync {
    fn observe(&self, layer: usize, position: usize, kv_head: usize, key: &[f32], value: &[f32]);
}

pub type KvProbeHandle = Arc<dyn KvProbe>;
