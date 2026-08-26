use crate::teacher::ExperienceId;
use crate::trace::{TraceEvent, TraceEventKind};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvenanceRef {
    pub sequence: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedExperience {
    pub id: ExperienceId,
    pub session: String,
    pub task: String,
    pub context: Vec<String>,
    pub observations: Vec<String>,
    pub actions: Vec<String>,
    pub tool_results: Vec<String>,
    pub corrections: Vec<String>,
    pub outcome: Option<String>,
    pub evidence: Vec<String>,
    pub provenance: Vec<ProvenanceRef>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExperienceError {
    EmptyTrace,
    MissingTask,
    NonContiguousSession,
}

pub struct ExperienceNormalizer;

impl ExperienceNormalizer {
    pub fn normalize(events: &[TraceEvent], id: ExperienceId) -> Result<NormalizedExperience, ExperienceError> {
        if events.is_empty() { return Err(ExperienceError::EmptyTrace); }
        let session = events[0].session_id.as_str().to_owned();
        if events.iter().any(|event| event.session_id.as_str() != session) {
            return Err(ExperienceError::NonContiguousSession);
        }

        let mut task = None;
        let mut context = Vec::new();
        let mut observations = Vec::new();
        let mut actions = Vec::new();
        let mut tool_results = Vec::new();
        let mut corrections = Vec::new();
        let mut outcome = None;
        let mut evidence = Vec::new();
        let mut provenance = Vec::with_capacity(events.len());

        for event in events {
            provenance.push(ProvenanceRef { sequence: event.sequence });
            match event.kind {
                TraceEventKind::UserIntent => task = Some(event.payload.data.clone()),
                TraceEventKind::ContextSnapshot => context.push(event.payload.data.clone()),
                TraceEventKind::ToolCall => actions.push(event.payload.data.clone()),
                TraceEventKind::ToolResult => tool_results.push(event.payload.data.clone()),
                TraceEventKind::TeacherMessage | TraceEventKind::FileRead | TraceEventKind::GitState => {
                    observations.push(event.payload.data.clone())
                }
                TraceEventKind::PatchApplied | TraceEventKind::FileWrite => corrections.push(event.payload.data.clone()),
                TraceEventKind::TestResult | TraceEventKind::CiResult | TraceEventKind::SessionFinished => {
                    evidence.push(event.payload.data.clone());
                    outcome = Some(event.payload.data.clone());
                }
                TraceEventKind::SessionStarted | TraceEventKind::CommandExecuted => {}
            }
        }

        let task = task.ok_or(ExperienceError::MissingTask)?;
        Ok(NormalizedExperience { id, session, task, context, observations, actions, tool_results, corrections, outcome, evidence, provenance })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::teacher::{ExperienceId, TeacherId};
    use crate::trace::{SessionId, TracePayload};

    fn event(kind: TraceEventKind, sequence: u64, payload: &str) -> TraceEvent {
        TraceEvent { session_id: SessionId::new("s1").unwrap(), sequence, timestamp_unix_nanos: 1, source: TeacherId::new("claude-code").unwrap(), kind, payload: TracePayload::new(payload), parent_observation: None, experience_id: None }
    }

    #[test]
    fn normalization_is_deterministic_and_preserves_provenance() {
        let events = vec![
            event(TraceEventKind::UserIntent, 1, "fix parser"),
            event(TraceEventKind::ToolCall, 2, "Read src/lib.rs"),
            event(TraceEventKind::ToolResult, 3, "ok"),
            event(TraceEventKind::TestResult, 4, "PASS"),
        ];
        let id = ExperienceId::new("exp-1").unwrap();
        let left = ExperienceNormalizer::normalize(&events, id.clone()).unwrap();
        let right = ExperienceNormalizer::normalize(&events, id).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.task, "fix parser");
        assert_eq!(left.actions, vec!["Read src/lib.rs"]);
        assert_eq!(left.provenance.iter().map(|p| p.sequence).collect::<Vec<_>>(), vec![1,2,3,4]);
    }

    #[test]
    fn missing_task_is_rejected() {
        let events = vec![event(TraceEventKind::ToolCall, 1, "Read")];
        let result = ExperienceNormalizer::normalize(&events, ExperienceId::new("exp-1").unwrap());
        assert_eq!(result, Err(ExperienceError::MissingTask));
    }
}
