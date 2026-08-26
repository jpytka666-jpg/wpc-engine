use crate::observatory::{TrainingObservation, TrainingObservatory, TensorDeltaSummary};
use crate::teacher::ObservationRecord;
use crate::trace::TraceEvent;

#[derive(Debug, Clone, PartialEq)]
pub enum ObservationEvent {
    Trace(TraceEvent),
    Teacher(ObservationRecord),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObservationError {
    SinkRejected(String),
}

pub trait ObservationSink {
    fn accept(&mut self, event: &ObservationEvent) -> Result<(), ObservationError>;
}

#[derive(Debug, Default)]
pub struct RawTraceSink {
    events: Vec<TraceEvent>,
}

impl RawTraceSink {
    pub fn events(&self) -> &[TraceEvent] { &self.events }
}

impl ObservationSink for RawTraceSink {
    fn accept(&mut self, event: &ObservationEvent) -> Result<(), ObservationError> {
        if let ObservationEvent::Trace(trace) = event { self.events.push(trace.clone()); }
        Ok(())
    }
}

#[derive(Debug)]
pub struct TrainingObservatorySink {
    observatory: TrainingObservatory,
}

impl TrainingObservatorySink {
    pub fn new() -> Self { Self { observatory: TrainingObservatory::new() } }
    pub fn observatory(&self) -> &TrainingObservatory { &self.observatory }
}

impl Default for TrainingObservatorySink {
    fn default() -> Self { Self::new() }
}

impl ObservationSink for TrainingObservatorySink {
    fn accept(&mut self, event: &ObservationEvent) -> Result<(), ObservationError> {
        if let ObservationEvent::Teacher(ObservationRecord::Delta(delta)) = event {
            if let Some(experience_id) = &delta.experience_id {
                let summary = TensorDeltaSummary {
                    tensor_name: "teacher-observed".into(),
                    changed_elements: delta.changed_elements,
                    l1: delta.l1,
                    l2: delta.l2,
                    max_abs: delta.max_abs,
                };
                self.observatory.record(TrainingObservation {
                    step: delta.after_step,
                    experience_id: experience_id.as_str().to_owned(),
                    weight_set: delta.weight_set.clone(),
                    loss: None,
                    deltas: vec![summary],
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct DiagnosticsSink {
    accepted: usize,
}

impl DiagnosticsSink {
    pub fn accepted(&self) -> usize { self.accepted }
}

impl ObservationSink for DiagnosticsSink {
    fn accept(&mut self, _event: &ObservationEvent) -> Result<(), ObservationError> {
        self.accepted += 1;
        Ok(())
    }
}

pub struct ObservationBus {
    sinks: Vec<Box<dyn ObservationSink>>,
}

impl ObservationBus {
    pub fn new() -> Self { Self { sinks: Vec::new() } }

    pub fn add_sink(&mut self, sink: Box<dyn ObservationSink>) {
        self.sinks.push(sink);
    }

    pub fn publish(&mut self, event: ObservationEvent) -> Result<(), ObservationError> {
        for sink in &mut self.sinks { sink.accept(&event)?; }
        Ok(())
    }

    pub fn sink_count(&self) -> usize { self.sinks.len() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::teacher::{ExperienceId, ObservationId, TeacherDelta, TeacherId};
    use crate::trace::{SessionId, TraceEventKind, TracePayload};
    use crate::{WeightSetId};

    #[test]
    fn one_event_reaches_multiple_sinks() {
        let mut bus = ObservationBus::new();
        bus.add_sink(Box::new(RawTraceSink::default()));
        bus.add_sink(Box::new(DiagnosticsSink::default()));
        assert_eq!(bus.sink_count(), 2);
        let event = ObservationEvent::Trace(TraceEvent {
            session_id: SessionId::new("s1").unwrap(),
            sequence: 1,
            timestamp_unix_nanos: 1,
            source: TeacherId::new("claude-code").unwrap(),
            kind: TraceEventKind::ToolCall,
            payload: TracePayload::new("inspect_code"),
            parent_observation: None,
            experience_id: None,
        });
        bus.publish(event).unwrap();
    }

    #[test]
    fn teacher_delta_can_feed_training_observatory_sink() {
        let mut sink = TrainingObservatorySink::new();
        let event = ObservationEvent::Teacher(ObservationRecord::Delta(TeacherDelta {
            teacher_id: TeacherId::new("teacher").unwrap(),
            observation_id: ObservationId::new("obs").unwrap(),
            experience_id: Some(ExperienceId::new("exp").unwrap()),
            before_step: 1,
            after_step: 2,
            changed_elements: 4,
            l1: 1.0,
            l2: 0.5,
            max_abs: 0.25,
            weight_set: WeightSetId::new("teacher-coding"),
        }));
        sink.accept(&event).unwrap();
        assert_eq!(sink.observatory().observations().len(), 1);
    }
}
