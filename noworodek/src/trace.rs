use crate::teacher::{ExperienceId, ObservationId, TeacherId};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionId(String);

impl SessionId {
    pub fn new(value: impl Into<String>) -> Result<Self, TraceError> {
        let value = value.into();
        if value.is_empty() { return Err(TraceError::EmptyId("session")); }
        Ok(Self(value))
    }
    pub fn as_str(&self) -> &str { &self.0 }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceEventKind {
    SessionStarted,
    UserIntent,
    ContextSnapshot,
    ToolCall,
    ToolResult,
    FileRead,
    FileWrite,
    PatchApplied,
    CommandExecuted,
    TestResult,
    CiResult,
    GitState,
    TeacherMessage,
    SessionFinished,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TracePayload {
    pub data: String,
}

impl TracePayload {
    pub fn new(data: impl Into<String>) -> Self { Self { data: data.into() } }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceEvent {
    pub session_id: SessionId,
    pub sequence: u64,
    pub timestamp_unix_nanos: u128,
    pub source: TeacherId,
    pub kind: TraceEventKind,
    pub payload: TracePayload,
    pub parent_observation: Option<ObservationId>,
    pub experience_id: Option<ExperienceId>,
}

impl TraceEvent {
    pub fn validate(&self) -> Result<(), TraceError> {
        if self.payload.data.is_empty() { return Err(TraceError::EmptyPayload); }
        Ok(())
    }

    pub fn jsonl_line(&self) -> String {
        format!(
            "{{\"session_id\":\"{}\",\"sequence\":{},\"timestamp_unix_nanos\":{},\"source\":\"{}\",\"kind\":\"{:?}\",\"payload\":\"{}\"}}",
            escape(self.session_id.as_str()),
            self.sequence,
            self.timestamp_unix_nanos,
            escape(self.source.as_str()),
            self.kind,
            escape(&self.payload.data),
        )
    }
}

fn escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n").replace('\r', "\\r")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceError {
    EmptyId(&'static str),
    EmptyPayload,
    SequenceRegression,
}

#[derive(Default, Debug, Clone)]
pub struct TraceSequenceGuard {
    last_sequence: Option<u64>,
}

impl TraceSequenceGuard {
    pub fn accept(&mut self, event: &TraceEvent) -> Result<(), TraceError> {
        if let Some(last) = self.last_sequence {
            if event.sequence <= last { return Err(TraceError::SequenceRegression); }
        }
        self.last_sequence = Some(event.sequence);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(sequence: u64) -> TraceEvent {
        TraceEvent {
            session_id: SessionId::new("session-1").unwrap(),
            sequence,
            timestamp_unix_nanos: 10,
            source: TeacherId::new("claude-code").unwrap(),
            kind: TraceEventKind::ToolCall,
            payload: TracePayload::new("inspect_code"),
            parent_observation: None,
            experience_id: None,
        }
    }

    #[test]
    fn trace_event_is_losslessly_described_as_jsonl() {
        let event = event(1);
        let line = event.jsonl_line();
        assert!(line.contains("session-1"));
        assert!(line.contains("ToolCall"));
        assert!(line.contains("inspect_code"));
    }

    #[test]
    fn sequence_guard_rejects_regression() {
        let mut guard = TraceSequenceGuard::default();
        guard.accept(&event(1)).unwrap();
        assert_eq!(guard.accept(&event(1)), Err(TraceError::SequenceRegression));
        assert_eq!(guard.accept(&event(0)), Err(TraceError::SequenceRegression));
        guard.accept(&event(2)).unwrap();
    }

    #[test]
    fn empty_payload_is_rejected() {
        let mut event = event(1);
        event.payload = TracePayload::new("");
        assert_eq!(event.validate(), Err(TraceError::EmptyPayload));
    }
}
