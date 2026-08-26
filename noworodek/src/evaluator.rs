use crate::experience::NormalizedExperience;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutcomeClass { Success, Partial, Failure }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Evidence {
    pub kind: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationResult {
    pub class: OutcomeClass,
    pub score: f32,
    pub evidence: Vec<Evidence>,
}

impl EvaluationResult {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.score.is_finite() || !(0.0..=1.0).contains(&self.score) {
            return Err("evaluation score must be finite and within [0,1]");
        }
        Ok(())
    }
}

pub trait Evaluator {
    fn evaluate(&self, experience: &NormalizedExperience) -> EvaluationResult;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct EvidenceEvaluator;

impl Evaluator for EvidenceEvaluator {
    fn evaluate(&self, experience: &NormalizedExperience) -> EvaluationResult {
        let mut evidence = Vec::new();
        let mut score = 0.0f32;
        let mut class = OutcomeClass::Failure;

        for item in &experience.evidence {
            let normalized = item.trim().to_ascii_uppercase();
            if normalized.contains("PASS") || normalized.contains("SUCCESS") || normalized.contains("GREEN") {
                score = score.max(1.0);
                class = OutcomeClass::Success;
                evidence.push(Evidence { kind: "observable_result".into(), value: item.clone() });
            } else if normalized.contains("FAIL") || normalized.contains("ERROR") || normalized.contains("RED") {
                if class != OutcomeClass::Success { class = OutcomeClass::Partial; score = score.max(0.25); }
                evidence.push(Evidence { kind: "observable_result".into(), value: item.clone() });
            }
        }

        if class == OutcomeClass::Failure && !experience.actions.is_empty() {
            class = OutcomeClass::Partial;
            score = score.max(0.1);
        }

        EvaluationResult { class, score, evidence }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experience::{ExperienceNormalizer, NormalizedExperience};
    use crate::teacher::{ExperienceId, TeacherId};
    use crate::trace::{SessionId, TraceEvent, TraceEventKind, TracePayload};

    fn event(kind: TraceEventKind, sequence: u64, payload: &str) -> TraceEvent {
        TraceEvent { session_id: SessionId::new("s1").unwrap(), sequence, timestamp_unix_nanos: 1, source: TeacherId::new("claude-code").unwrap(), kind, payload: TracePayload::new(payload), parent_observation: None, experience_id: None }
    }

    #[test]
    fn passing_test_evidence_scores_success() {
        let events = vec![event(TraceEventKind::UserIntent, 1, "fix parser"), event(TraceEventKind::ToolCall, 2, "edit"), event(TraceEventKind::TestResult, 3, "PASS")];
        let experience = ExperienceNormalizer::normalize(&events, ExperienceId::new("exp-1").unwrap()).unwrap();
        let result = EvidenceEvaluator.evaluate(&experience);
        assert_eq!(result.class, OutcomeClass::Success);
        assert_eq!(result.score, 1.0);
        assert!(result.validate().is_ok());
    }

    #[test]
    fn failed_evidence_never_claims_success() {
        let experience = NormalizedExperience { id: ExperienceId::new("exp-1").unwrap(), session: "s1".into(), task: "broken".into(), context: vec![], observations: vec![], actions: vec!["edit".into()], tool_results: vec![], corrections: vec![], outcome: Some("FAIL".into()), evidence: vec!["TEST FAIL".into()], provenance: vec![] };
        let result = EvidenceEvaluator.evaluate(&experience);
        assert_ne!(result.class, OutcomeClass::Success);
        assert!(result.score < 1.0);
    }
}
