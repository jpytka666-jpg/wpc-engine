//! Read-only teacher-model observation contracts.
//!
//! The protocol records what was observed without assuming teacher tensors can
//! be copied directly into a differently shaped student model.

use crate::weights::{ArchitectureId, TensorSpec, WeightSetId};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TeacherId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExperienceId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ObservationId(pub String);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TeacherTensor {
    pub name: String,
    pub spec: TensorSpec,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TeacherManifest {
    pub teacher_id: TeacherId,
    pub architecture: ArchitectureId,
    pub tensors: Vec<TeacherTensor>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorStats {
    pub element_count: usize,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub l2: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TeacherSnapshot {
    pub teacher_id: TeacherId,
    pub observation_id: ObservationId,
    pub step: u64,
    pub weight_set: WeightSetId,
    pub tensors: Vec<TensorStats>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TeacherDelta {
    pub teacher_id: TeacherId,
    pub observation_id: ObservationId,
    pub experience_id: Option<ExperienceId>,
    pub before_step: u64,
    pub after_step: u64,
    pub changed_elements: usize,
    pub l1: f32,
    pub l2: f32,
    pub max_abs: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BehaviourTrace {
    pub teacher_id: TeacherId,
    pub experience_id: ExperienceId,
    pub task: String,
    pub actions: Vec<String>,
    pub tool_results: Vec<String>,
    pub final_output: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ObservationRecord {
    Structure(TeacherManifest),
    Snapshot(TeacherSnapshot),
    Delta(TeacherDelta),
    Behaviour(BehaviourTrace),
}

/// Read-only boundary for a teacher model.
///
/// Implementations expose metadata and observations; they do not expose a
/// mutable handle to teacher parameters.
pub trait TeacherObserver {
    fn manifest(&self) -> &TeacherManifest;
    fn observe(&self, observation_id: ObservationId, step: u64) -> TeacherSnapshot;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn teacher_delta_can_be_linked_to_an_experience() {
        let delta = TeacherDelta {
            teacher_id: TeacherId("teacher-a".into()),
            observation_id: ObservationId("obs-1".into()),
            experience_id: Some(ExperienceId("exp-42".into())),
            before_step: 10,
            after_step: 11,
            changed_elements: 3,
            l1: 1.0,
            l2: 0.5,
            max_abs: 0.25,
        };

        assert_eq!(delta.experience_id, Some(ExperienceId("exp-42".into())));
        assert_eq!(delta.after_step - delta.before_step, 1);
    }

    #[test]
    fn behaviour_trace_is_read_only_data() {
        let trace = BehaviourTrace {
            teacher_id: TeacherId("teacher-a".into()),
            experience_id: ExperienceId("exp-1".into()),
            task: "inspect Rust error".into(),
            actions: vec!["inspect_code".into()],
            tool_results: vec!["compile failed".into()],
            final_output: "diagnose error".into(),
        };

        assert_eq!(trace.actions.len(), 1);
        assert_eq!(trace.tool_results.len(), 1);
    }
}
