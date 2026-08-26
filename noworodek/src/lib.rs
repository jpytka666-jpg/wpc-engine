pub const VERSION: &str = "0.1.0";

pub mod backend;
pub mod device;
pub mod editor;
pub mod evaluator;
pub mod experience;
pub mod observation_bus;
pub mod observatory;
pub mod snapshot;
pub mod teacher;
pub mod teachers;
pub mod tensor;
pub mod trace;
pub mod trace_store;
pub mod training;
pub mod weightset;

pub use backend::{MemoryWeightBackend, MountedWeightSet, WeightBackend, WeightSetManager};
pub use device::{ComputeDevice, DeviceBackend, DeviceMemoryReport, ResidencyPolicy, ResidencyViolation};
pub use editor::{diff_tensors, snapshot_tensor, TensorDiff, WeightEditor};
pub use evaluator::{Evidence, EvidenceEvaluator, EvaluationResult, Evaluator, OutcomeClass};
pub use experience::{ExperienceError, ExperienceNormalizer, NormalizedExperience, ProvenanceRef};
pub use observation_bus::{DiagnosticsSink, ObservationBus, ObservationError, ObservationEvent, ObservationSink, RawTraceSink, TrainingObservatorySink};
pub use observatory::{TensorDeltaSummary, TrainingObservation, TrainingObservatory};
pub use snapshot::{WeightSetSnapshot, WeightSetSnapshotEntry, SNAPSHOT_SCHEMA_VERSION};
pub use teacher::{BehaviourTrace, ExperienceId, ObservationId, ObservationRecord, TeacherDelta, TeacherId, TeacherManifest, TeacherObserver, TeacherSnapshot, TeacherTensor, TensorStats};
pub use teachers::ClaudeCodeAdapter;
pub use tensor::Tensor;
pub use trace::{SessionId, TraceError, TraceEvent, TraceEventKind, TracePayload, TraceSequenceGuard};
pub use trace_store::{RawTraceStore, TraceStoreError};
pub use training::BigramLanguageModel;
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
