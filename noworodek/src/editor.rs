use crate::backend::{WeightBackend, WeightSetManager};
use crate::observatory::TensorDeltaSummary;
use crate::weightset::{WeightSetError, WeightSetId};

#[derive(Clone, Debug, PartialEq)]
pub struct TensorDiff {
    pub tensor_name: String,
    pub changed_elements: usize,
    pub l1: f32,
    pub l2: f32,
    pub max_abs: f32,
}

pub fn diff_tensors(name: impl Into<String>, before: &[f32], after: &[f32]) -> Result<TensorDiff, WeightSetError> {
    if before.len() != after.len() {
        return Err(WeightSetError::Backend("tensor length mismatch during diff".into()));
    }
    let mut delta = Vec::with_capacity(before.len());
    for (&old, &new) in before.iter().zip(after) { delta.push(new - old); }
    let summary = TensorDeltaSummary::from_delta(name, &delta);
    Ok(TensorDiff {
        tensor_name: summary.tensor_name,
        changed_elements: summary.changed_elements,
        l1: summary.l1,
        l2: summary.l2,
        max_abs: summary.max_abs,
    })
}

pub struct WeightEditor<'a> {
    manager: &'a mut WeightSetManager,
}

impl<'a> WeightEditor<'a> {
    pub fn new(manager: &'a mut WeightSetManager) -> Self { Self { manager } }

    pub fn scale_tensor(&mut self, id: &WeightSetId, tensor: &str, factor: f32) -> Result<(), WeightSetError> {
        let mounted = self.manager.active_mut(id).ok_or_else(|| WeightSetError::NotMounted(id.clone()))?;
        let values = mounted.backend_mut().tensor_mut(tensor)?;
        for value in values { *value *= factor; }
        Ok(())
    }

    pub fn add_delta(&mut self, id: &WeightSetId, tensor: &str, delta: &[f32]) -> Result<(), WeightSetError> {
        let mounted = self.manager.active_mut(id).ok_or_else(|| WeightSetError::NotMounted(id.clone()))?;
        let values = mounted.backend_mut().tensor_mut(tensor)?;
        if values.len() != delta.len() {
            return Err(WeightSetError::Backend("tensor length mismatch during edit".into()));
        }
        for (value, change) in values.iter_mut().zip(delta) { *value += change; }
        Ok(())
    }

    pub fn replace_tensor(&mut self, id: &WeightSetId, tensor: &str, replacement: &[f32]) -> Result<(), WeightSetError> {
        let mounted = self.manager.active_mut(id).ok_or_else(|| WeightSetError::NotMounted(id.clone()))?;
        let values = mounted.backend_mut().tensor_mut(tensor)?;
        if values.len() != replacement.len() {
            return Err(WeightSetError::Backend("tensor length mismatch during replacement".into()));
        }
        values.copy_from_slice(replacement);
        Ok(())
    }
}

// Keep the editor generic: this helper is intentionally independent of a concrete storage format.
pub fn snapshot_tensor(backend: &dyn WeightBackend, tensor: &str) -> Result<Vec<f32>, WeightSetError> {
    backend.tensor(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::MemoryWeightBackend;
    use crate::weightset::{ArchitectureId, DType, TensorSpec, WeightSetHeader, WeightSetManifest, WeightSetVersion};

    fn manifest() -> WeightSetManifest {
        WeightSetManifest::new(
            WeightSetHeader::new(WeightSetId::new("coding"), WeightSetVersion::new("1.0.0").unwrap(), ArchitectureId::new("noworodek-v0")),
            vec![TensorSpec::new("x", vec![2], DType::F32, "x")],
        ).unwrap()
    }

    #[test]
    fn diff_reports_parameter_change() {
        let diff = diff_tensors("x", &[1.0, 2.0], &[2.0, 1.0]).unwrap();
        assert_eq!(diff.changed_elements, 2);
        assert_eq!(diff.l1, 2.0);
    }

    #[test]
    fn editor_can_perform_controlled_weight_surgery() {
        let mut manager = WeightSetManager::new(ArchitectureId::new("noworodek-v0"));
        let backend = MemoryWeightBackend::with_tensor_data(manifest(), [("x", vec![1.0, 2.0])]);
        let id = manager.mount(Box::new(backend)).unwrap();
        let mut editor = WeightEditor::new(&mut manager);
        editor.scale_tensor(&id, "x", 2.0).unwrap();
        assert_eq!(snapshot_tensor(manager.active(&id).unwrap().backend(), "x").unwrap(), vec![2.0, 4.0]);
    }
}
