//! Read-only teacher-model observation contracts.
//!
//! The protocol records what was observed without assuming teacher tensors can
//! be copied directly into a differently shaped student model.

use crate::weightset::{ArchitectureId, TensorSpec, WeightSetId};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TeacherId(String);

impl TeacherId {
    pub fn new(value: impl Into<String>) -> Result<Self, &'static str> {
        let value = value.into();
        if value.is_empty() { return Err("teacher id must not be empty"); }
        Ok(Self(value))
    }
    pub fn as_str(&self) -> &str { &self.0 }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExperienceId(String);

impl ExperienceId {
    pub fn new(value: impl Into<String>) -> Result<Self, &'static str> {
        let value = value.into();
        if value.is_empty() { return Err("experience id must not be empty"); }
        Ok(Self(value))
    }
    pub fn as_str(&self) -> &str { &self.0 }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ObservationId(String);

impl ObservationId {
    pub fn new(value: impl Into<String>) -> Result<Self, &'static str> {
        let value = value.into();
        if value.is_empty() { return Err("observation id must not be empty"); }
        Ok(Self(value))
    }
    pub fn as_str(&self) -> &str { &self.0 }
}

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

impl TeacherManifest {
    pub fn new(teacher_id: TeacherId, architecture: ArchitectureId, tensors: Vec<TeacherTensor>) -> Result<Self, &'static str> {
        if architecture.as_str().is_empty() { return Err("teacher architecture must not be empty"); }
        Ok(Self { teacher_id, architecture, tensors })
    }
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

impl TeacherDelta {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.after_step < self.before_step { return Err("teacher delta step range is reversed"); }
        if !self.l1.is_finite() || !self.l2.is_finite() || !self.max_abs.is_finite() { return Err("teacher delta statistics must be finite"); }
        Ok(())
    }
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
    fn ids_must_be_non_empty() {
        assert!(TeacherId::new("").is_err());
        assert!(ExperienceId::new("").is_err());
        assert!(ObservationId::new("").is_err());
    }

    #[test]
    fn manifest_preserves_architecture_and_tensor_metadata() {
        let teacher_id = TeacherId::new("claude-code").unwrap();
        let architecture = ArchitectureId::new("teacher-v1");
        let tensor = TeacherTensor {
            name: "layer.0.q".into(),
            spec: TensorSpec::new("layer.0.q", vec![2, 2], crate::DType::F32, "checksum"),
        };
        let manifest = TeacherManifest::new(teacher_id.clone(), architecture.clone(), vec![tensor.clone()]).unwrap();
        assert_eq!(manifest.teacher_id, teacher_id);
        assert_eq!(manifest.architecture, architecture);
        assert_eq!(manifest.tensors[0], tensor);
    }

    #[test]
    fn teacher_delta_can_be_linked_to_an_experience_and_step_range() {
        let delta = TeacherDelta {
            teacher_id: TeacherId::new("teacher-a").unwrap(),
            observation_id: ObservationId::new("obs-1").unwrap(),
            experience_id: Some(ExperienceId::new("exp-42").unwrap()),
            before_step: 10,
            after_step: 11,
            changed_elements: 3,
            l1: 1.0,
            l2: 0.5,
            max_abs: 0.25,
        };
        assert!(delta.validate().is_ok());
        assert_eq!(delta.experience_id.unwrap().as_str(), "exp-42");
    }

    #[test]
    fn reversed_teacher_delta_is_rejected() {
        let delta = TeacherDelta {
            teacher_id: TeacherId::new("teacher-a").unwrap(),
            observation_id: ObservationId::new("obs-1").unwrap(),
            experience_id: None,
            before_step: 11,
            after_step: 10,
            changed_elements: 0,
            l1: 0.0,
            l2: 0.0,
            max_abs: 0.0,
        };
        assert!(delta.validate().is_err());
    }

    #[test]
    fn behaviour_trace_is_read_only_data() {
        let trace = BehaviourTrace {
            teacher_id: TeacherId::new("teacher-a").unwrap(),
            experience_id: ExperienceId::new("exp-1").unwrap(),
            task: "inspect Rust error".into(),
            actions: vec!["inspect_code".into()],
            tool_results: vec!["compile failed".into()],
            final_output: "diagnose error".into(),
        };
        assert_eq!(trace.actions.len(), 1);
        assert_eq!(trace.tool_results.len(), 1);
    }
}
