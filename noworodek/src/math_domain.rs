//! Mathematical domain examples and deterministic evaluation contracts.
//!
//! This module intentionally starts with exact symbolic/numeric tasks rather
//! than free-form prose. Each example keeps the problem, expected form, and
//! verification evidence together so training can distinguish reasoning from
//! a lucky final token sequence.

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MathDomain {
    Arithmetic,
    Algebra,
    LinearAlgebra,
    Calculus,
    Discrete,
    Geometry,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MathExample {
    pub id: String,
    pub domain: MathDomain,
    pub prompt: String,
    pub canonical_answer: String,
    pub difficulty: u8,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MathEvaluation {
    pub correct: bool,
    pub exact_match: bool,
    pub score: f32,
    pub evidence: String,
}

pub fn evaluate_exact(example: &MathExample, answer: &str) -> MathEvaluation {
    let exact_match = answer.trim() == example.canonical_answer.trim();
    MathEvaluation {
        correct: exact_match,
        exact_match,
        score: if exact_match { 1.0 } else { 0.0 },
        evidence: format!("canonical-answer-check:{}", example.id),
    }
}

pub fn starter_curriculum() -> Vec<MathExample> {
    vec![
        MathExample { id: "arith-001".into(), domain: MathDomain::Arithmetic, prompt: "17 * 19 = ?".into(), canonical_answer: "323".into(), difficulty: 1 },
        MathExample { id: "alg-001".into(), domain: MathDomain::Algebra, prompt: "Solve 3x + 7 = 22 for x.".into(), canonical_answer: "5".into(), difficulty: 2 },
        MathExample { id: "la-001".into(), domain: MathDomain::LinearAlgebra, prompt: "For A=[[1,2],[0,3]], what is det(A)?".into(), canonical_answer: "3".into(), difficulty: 3 },
        MathExample { id: "calc-001".into(), domain: MathDomain::Calculus, prompt: "d/dx (x^3 + 2x) = ?".into(), canonical_answer: "3x^2 + 2".into(), difficulty: 3 },
        MathExample { id: "disc-001".into(), domain: MathDomain::Discrete, prompt: "How many subsets does a set with 5 elements have?".into(), canonical_answer: "32".into(), difficulty: 2 },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_evaluator_accepts_canonical_answer() {
        let example = MathExample {
            id: "test".into(),
            domain: MathDomain::Arithmetic,
            prompt: "2 + 2".into(),
            canonical_answer: "4".into(),
            difficulty: 1,
        };
        let result = evaluate_exact(&example, "4");
        assert!(result.correct);
        assert_eq!(result.score, 1.0);
    }

    #[test]
    fn exact_evaluator_rejects_wrong_answer() {
        let example = MathExample {
            id: "test".into(),
            domain: MathDomain::Arithmetic,
            prompt: "2 + 2".into(),
            canonical_answer: "4".into(),
            difficulty: 1,
        };
        let result = evaluate_exact(&example, "5");
        assert!(!result.correct);
        assert_eq!(result.score, 0.0);
    }

    #[test]
    fn starter_curriculum_has_multiple_domains() {
        let curriculum = starter_curriculum();
        assert!(curriculum.len() >= 5);
        assert!(curriculum.iter().any(|item| item.domain == MathDomain::Calculus));
        assert!(curriculum.iter().any(|item| item.domain == MathDomain::LinearAlgebra));
    }
}
