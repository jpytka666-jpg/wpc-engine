use crate::teacher::ExperienceId;
use crate::trace::{SessionId, TraceError, TraceEvent, TraceSequenceGuard};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceStoreError {
    InvalidEvent(TraceError),
    SessionMismatch,
}

#[derive(Default, Debug, Clone)]
pub struct RawTraceStore {
    events: Vec<TraceEvent>,
    sequence: TraceSequenceGuard,
}

impl RawTraceStore {
    pub fn new() -> Self { Self::default() }

    pub fn append(&mut self, event: TraceEvent) -> Result<(), TraceStoreError> {
        event.validate().map_err(TraceStoreError::InvalidEvent)?;
        if let Some(first) = self.events.first() {
            if first.session_id != event.session_id { return Err(TraceStoreError::SessionMismatch); }
        }
        self.sequence.accept(&event).map_err(TraceStoreError::InvalidEvent)?;
        self.events.push(event);
        Ok(())
    }

    pub fn events(&self) -> &[TraceEvent] { &self.events }

    pub fn session(&self) -> Option<&SessionId> { self.events.first().map(|event| &event.session_id) }

    pub fn experience_slice(&self, experience_id: &ExperienceId) -> Vec<&TraceEvent> {
        self.events.iter().filter(|event| event.experience_id.as_ref() == Some(experience_id)).collect()
    }

    pub fn replay_jsonl(&self) -> impl Iterator<Item = String> + '_ {
        self.events.iter().map(TraceEvent::jsonl_line)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::teacher::{ExperienceId, TeacherId};
    use crate::trace::{TraceEventKind, TracePayload};

    fn event(sequence: u64, experience: Option<&str>) -> TraceEvent {
        TraceEvent {
            session_id: SessionId::new("session-1").unwrap(),
            sequence,
            timestamp_unix_nanos: sequence as u128,
            source: TeacherId::new("claude-code").unwrap(),
            kind: TraceEventKind::ToolCall,
            payload: TracePayload::new(format!("call-{sequence}")),
            parent_observation: None,
            experience_id: experience.map(|id| ExperienceId::new(id).unwrap()),
        }
    }

    #[test]
    fn store_preserves_event_order_and_replays_identically() {
        let mut store = RawTraceStore::new();
        store.append(event(1, Some("exp-1"))).unwrap();
        store.append(event(2, Some("exp-1"))).unwrap();
        let sequences: Vec<u64> = store.events().iter().map(|event| event.sequence).collect();
        assert_eq!(sequences, vec![1, 2]);
        assert_eq!(store.replay_jsonl().count(), 2);
    }

    #[test]
    fn store_returns_experience_slice() {
        let mut store = RawTraceStore::new();
        store.append(event(1, Some("exp-1"))).unwrap();
        store.append(event(2, Some("exp-2"))).unwrap();
        assert_eq!(store.experience_slice(&ExperienceId::new("exp-2").unwrap()).len(), 1);
    }

    #[test]
    fn store_rejects_a_second_session() {
        let mut store = RawTraceStore::new();
        store.append(event(1, None)).unwrap();
        let mut second = event(2, None);
        second.session_id = SessionId::new("session-2").unwrap();
        assert_eq!(store.append(second), Err(TraceStoreError::SessionMismatch));
    }
}
