use crate::teacher::{ExperienceId, ObservationId, TeacherId};
use crate::trace::{SessionId, TraceEvent, TraceEventKind, TracePayload};

#[derive(Debug, Clone)]
pub struct ClaudeCodeAdapter {
    teacher_id: TeacherId,
}

impl ClaudeCodeAdapter {
    pub fn new() -> Self {
        Self { teacher_id: TeacherId::new("claude-code").expect("static teacher id is valid") }
    }

    pub fn teacher_id(&self) -> &TeacherId { &self.teacher_id }

    pub fn map_hook(
        &self,
        hook: &str,
        session_id: SessionId,
        sequence: u64,
        timestamp_unix_nanos: u128,
        payload: impl Into<String>,
    ) -> Result<TraceEvent, &'static str> {
        let kind = match hook {
            "SessionStart" => TraceEventKind::SessionStarted,
            "UserPromptSubmit" => TraceEventKind::UserIntent,
            "PreToolUse" => TraceEventKind::ToolCall,
            "PostToolUse" | "PostToolUseFailure" => TraceEventKind::ToolResult,
            "PreCompact" => TraceEventKind::ContextSnapshot,
            "SessionEnd" | "Stop" | "StopFailure" => TraceEventKind::SessionFinished,
            "SubagentStart" | "SubagentStop" | "Notification" | "MessageDisplay" => TraceEventKind::TeacherMessage,
            _ => return Err("unsupported Claude Code hook"),
        };

        let event = TraceEvent {
            session_id,
            sequence,
            timestamp_unix_nanos,
            source: self.teacher_id.clone(),
            kind,
            payload: TracePayload::new(payload),
            parent_observation: None,
            experience_id: None,
        };
        event.validate().map_err(|_| "invalid trace event")?;
        Ok(event)
    }

    pub fn link_event(
        &self,
        mut event: TraceEvent,
        observation_id: ObservationId,
        experience_id: Option<ExperienceId>,
    ) -> TraceEvent {
        event.parent_observation = Some(observation_id);
        event.experience_id = experience_id;
        event
    }
}

impl Default for ClaudeCodeAdapter {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_user_prompt_to_user_intent() {
        let adapter = ClaudeCodeAdapter::new();
        let event = adapter
            .map_hook("UserPromptSubmit", SessionId::new("s1").unwrap(), 1, 100, "fix rust")
            .unwrap();
        assert_eq!(event.kind, TraceEventKind::UserIntent);
        assert_eq!(event.source.as_str(), "claude-code");
    }

    #[test]
    fn maps_tool_lifecycle_to_tool_events() {
        let adapter = ClaudeCodeAdapter::new();
        let before = adapter.map_hook("PreToolUse", SessionId::new("s1").unwrap(), 1, 100, "Read").unwrap();
        let after = adapter.map_hook("PostToolUse", SessionId::new("s1").unwrap(), 2, 101, "ok").unwrap();
        assert_eq!(before.kind, TraceEventKind::ToolCall);
        assert_eq!(after.kind, TraceEventKind::ToolResult);
    }

    #[test]
    fn unsupported_hooks_fail_closed() {
        let adapter = ClaudeCodeAdapter::new();
        assert!(adapter.map_hook("UnknownFutureHook", SessionId::new("s1").unwrap(), 1, 100, "x").is_err());
    }

    #[test]
    fn event_can_be_linked_to_observation_and_experience() {
        let adapter = ClaudeCodeAdapter::new();
        let event = adapter.map_hook("PostToolUseFailure", SessionId::new("s1").unwrap(), 1, 100, "failed").unwrap();
        let linked = adapter.link_event(event, ObservationId::new("obs-1").unwrap(), Some(ExperienceId::new("exp-1").unwrap()));
        assert_eq!(linked.parent_observation.unwrap().as_str(), "obs-1");
        assert_eq!(linked.experience_id.unwrap().as_str(), "exp-1");
    }
}
