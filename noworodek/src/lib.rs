pub const VERSION: &str = "0.1.0";

pub mod backend;
pub mod editor;
pub mod observatory;
pub mod snapshot;
pub mod tensor;
pub mod weightset;

pub use backend::{MemoryWeightBackend, MountedWeightSet, WeightBackend, WeightSetManager};
pub use editor::{diff_tensors, snapshot_tensor, TensorDiff, WeightEditor};
pub use observatory::{TensorDeltaSummary, TrainingObservation, TrainingObservatory};
pub use snapshot::{WeightSetSnapshot, WeightSetSnapshotEntry, SNAPSHOT_SCHEMA_VERSION};
pub use tensor::Tensor;
pub use weightset::{
    ArchitectureId, DType, TensorSpec, WeightSetError, WeightSetHeader, WeightSetId,
    WeightSetManifest, WeightSetState, WeightSetVersion,
};

#[cfg(test)]
mod tests {
    #[test]
    fn noworodek_crate_is_reachable() {
        assert_eq!(crate::VERSION, "0.1.0");
    }
}
