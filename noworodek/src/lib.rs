pub const VERSION: &str = "0.1.0";

pub mod backend;
pub mod device;
pub mod editor;
pub mod evaluator;
pub mod experience;
pub mod model;
pub mod observation_bus;
pub mod observatory;
pub mod parameter_handle;
pub mod qwen;
pub mod snapshot;
pub mod teacher;
pub mod teachers;
pub mod tensor;
pub mod tokenizer;
pub mod trace;
pub mod trace_store;
pub mod training;
pub mod weightset;

pub use backend::{MemoryWeightBackend, MountedWeightSet, WeightBackend, WeightSetManager};
pub use device::{ComputeDevice, DeviceBackend, DeviceMemoryReport, ResidencyPolicy, ResidencyViolation};
pub use editor::{diff_tensors, snapshot_tensor, TensorDiff, WeightEditor};
pub use evaluator::{Evidence, EvidenceEvaluator, EvaluationResult, Evaluator, OutcomeClass};
pub use experience::{ExperienceError, ExperienceNormalizer, NormalizedExperience, ProvenanceRef};
pub use model::{ExternalTransformer, ParameterRegistration, ParameterRegistry, RegistryError, TinyTransformerConfig, TransformerTensorRole};
pub use observation_bus::{DiagnosticsSink, ObservationBus, ObservationError, ObservationEvent, ObservationSink, RawTraceSink, TrainingObservatorySink};
pub use observatory::{TensorDeltaSummary, TrainingObservation, TrainingObservatory};
pub use parameter_handle::ParameterHandle;
pub use qwen::qwen3_coder_registry;
pub use snapshot::{WeightSetSnapshot, WeightSetSnapshotEntry, SNAPSHOT_SCHEMA_VERSION};
pub use teacher::{BehaviourTrace, ExperienceId, ObservationId, ObservationRecord, TeacherDelta, TeacherId, TeacherManifest, TeacherObserver, TeacherSnapshot, TeacherTensor, TensorStats};
pub use teachers::ClaudeCodeAdapter;
pub use tensor::Tensor;
pub use tokenizer::{format_chat_turn, format_tool_call, Qwen3CoderTokenizer, TokenizerError, EOS_ID, EOS_TOKEN, IM_END_ID, IM_START_ID, MODEL_ID as QWEN3_CODER_TOKENIZER_MODEL_ID, MODEL_REVISION as QWEN3_CODER_TOKENIZER_REVISION, PAD_ID, PAD_TOKEN, VOCAB_SIZE as QWEN3_CODER_VOCAB_SIZE, MAX_POSITION_TOKENS as QWEN3_CODER_MAX_POSITION_TOKENS};
pub use trace::{SessionId, TraceError, TraceEvent, TraceEventKind, TracePayload, TraceSequenceGuard};
pub use trace_store::{RawTraceStore, TraceStoreError};
pub use training::BigramLanguageModel;
pub use weightset::{ArchitectureId, DType, TensorSpec, WeightSetError, WeightSetHeader, WeightSetId, WeightSetManifest, WeightSetState, WeightSetVersion};

#[cfg(test)]
mod tests {
    #[test]
    fn noworodek_crate_is_reachable() {
        assert_eq!(crate::VERSION, "0.1.0");
    }
}
