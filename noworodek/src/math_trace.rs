//! Observable mathematical solution traces.
//!
//! A mathematical training example is represented as a sequence of explicit
//! operations and intermediate states. This is intentionally not a hidden
//! reasoning trace: it contains only structured, externally defined steps
//! suitable for training and deterministic evaluation.

use crate::math_domain::MathDomain;

#[derive(Clone, Debug, PartialEq)]
pub enum MathOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Substitute,
    Differentiate,
    Determine,
    Rewrite,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MathStep {
    pub index: u32,
    pub op: MathOp,
    pub expression_before: String,
    pub expression_after: String,
    pub justification: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MathTrace {
    pub example_id: String,
    pub domain: MathDomain,
    pub steps: Vec<MathStep>,
    pub final_answer: String,
}

impl MathTrace {
    pub fn validate(&self) -> Result<(), String> {
        if self.example_id.is_empty() {
            return Err("example id is empty".into());
        }
        if self.steps.is_empty() {
            return Err("math trace must contain at least one step".into());
        }
        for (expected, step) in self.steps.iter().enumerate() {
            if step.index as usize != expected {
                return Err(format!("step index mismatch at position {expected}"));
            }
            if step.expression_after.is_empty() {
                return Err(format!("step {expected} has empty resulting expression"));
            }
        }
        if self.final_answer.is_empty() {
            return Err("final answer is empty".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_algebra_trace_is_accepted() {
        let trace = MathTrace {
            example_id: "alg-001".into(),
            domain: MathDomain::Algebra,
            steps: vec![
                MathStep { index: 0, op: MathOp::Subtract, expression_before: "3x + 7 = 22".into(), expression_after: "3x = 15".into(), justification: "subtract 7".into() },
                MathStep { index: 1, op: MathOp::Divide, expression_before: "3x = 15".into(), expression_after: "x = 5".into(), justification: "divide by 3".into() },
            ],
            final_answer: "5".into(),
        };
        assert!(trace.validate().is_ok());
    }

    #[test]
    fn invalid_trace_with_gap_is_rejected() {
        let trace = MathTrace {
            example_id: "x".into(),
            domain: MathDomain::Arithmetic,
            steps: vec![
                MathStep { index: 0, op: MathOp::Add, expression_before: "1+1".into(), expression_after: "2".into(), justification: "addition".into() },
                MathStep { index: 2, op: MathOp::Multiply, expression_before: "2*2".into(), expression_after: "4".into(), justification: "multiplication".into() },
            ],
            final_answer: "4".into(),
        };
        assert!(trace.validate().is_err());
    }
}
