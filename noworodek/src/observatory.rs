use crate::weightset::WeightSetId;

#[derive(Clone, Debug, PartialEq)]
pub struct TensorDeltaSummary {
    pub tensor_name: String,
    pub changed_elements: usize,
    pub l1: f32,
    pub l2: f32,
    pub max_abs: f32,
}

impl TensorDeltaSummary {
    pub fn from_delta(tensor_name: impl Into<String>, delta: &[f32]) -> Self {
        let mut l1: f32 = 0.0;
        let mut l2: f32 = 0.0;
        let mut max_abs: f32 = 0.0;
        let mut changed_elements = 0;
        for &value in delta {
            let abs = value.abs();
            if abs != 0.0 { changed_elements += 1; }
            l1 += abs;
            l2 += value * value;
            max_abs = max_abs.max(abs);
        }
        Self { tensor_name: tensor_name.into(), changed_elements, l1, l2: l2.sqrt(), max_abs }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainingObservation {
    pub step: u64,
    pub experience_id: String,
    pub weight_set: WeightSetId,
    pub loss: Option<f32>,
    pub deltas: Vec<TensorDeltaSummary>,
}

#[derive(Default, Debug)]
pub struct TrainingObservatory {
    observations: Vec<TrainingObservation>,
}

impl TrainingObservatory {
    pub fn new() -> Self { Self::default() }

    pub fn record(&mut self, observation: TrainingObservation) {
        self.observations.push(observation);
    }

    pub fn observations(&self) -> &[TrainingObservation] { &self.observations }

    pub fn latest(&self) -> Option<&TrainingObservation> { self.observations.last() }

    pub fn for_experience<'a>(&'a self, experience_id: &'a str) -> impl Iterator<Item = &'a TrainingObservation> + 'a {
        self.observations.iter().filter(move |item| item.experience_id == experience_id)
    }

    pub fn clear(&mut self) { self.observations.clear(); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_summary_captures_parameter_change() {
        let summary = TensorDeltaSummary::from_delta("layer.0.weight", &[0.0, 1.0, -2.0]);
        assert_eq!(summary.changed_elements, 2);
        assert_eq!(summary.l1, 3.0);
        assert_eq!(summary.max_abs, 2.0);
    }

    #[test]
    fn observations_are_indexed_by_experience() {
        let mut observatory = TrainingObservatory::new();
        observatory.record(TrainingObservation {
            step: 7,
            experience_id: "rust-result".into(),
            weight_set: WeightSetId::new("coding"),
            loss: Some(0.42),
            deltas: Vec::new(),
        });
        assert_eq!(observatory.for_experience("rust-result").count(), 1);
        assert_eq!(observatory.latest().unwrap().step, 7);
    }
}
