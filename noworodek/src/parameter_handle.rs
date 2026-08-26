use crate::backend::WeightSetManager;
use crate::{Tensor, WeightSetError};
use crate::weightset::WeightSetId;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParameterHandle {
    weight_set: WeightSetId,
    tensor_name: String,
}

impl ParameterHandle {
    pub fn new(weight_set: WeightSetId, tensor_name: impl Into<String>) -> Result<Self, WeightSetError> {
        let tensor_name = tensor_name.into();
        if tensor_name.is_empty() {
            return Err(WeightSetError::EmptyTensorName);
        }
        Ok(Self { weight_set, tensor_name })
    }

    pub fn weight_set(&self) -> &WeightSetId { &self.weight_set }
    pub fn tensor_name(&self) -> &str { &self.tensor_name }

    pub fn read(&self, manager: &WeightSetManager) -> Result<Tensor, WeightSetError> {
        let mounted = manager.active(&self.weight_set)
            .ok_or_else(|| WeightSetError::NotMounted(self.weight_set.clone()))?;
        let spec = mounted.manifest().tensor(&self.tensor_name)
            .ok_or_else(|| WeightSetError::Backend(format!("tensor not found in manifest: {}", self.tensor_name)))?;
        let values = mounted.backend().tensor(&self.tensor_name)?;
        Tensor::from_vec(spec.shape.clone(), values)
    }

    pub fn write(&self, manager: &mut WeightSetManager, tensor: &Tensor) -> Result<(), WeightSetError> {
        let mounted = manager.active_mut(&self.weight_set)
            .ok_or_else(|| WeightSetError::NotMounted(self.weight_set.clone()))?;
        let spec = mounted.manifest().tensor(&self.tensor_name)
            .ok_or_else(|| WeightSetError::Backend(format!("tensor not found in manifest: {}", self.tensor_name)))?;
        if spec.shape != tensor.shape() {
            return Err(WeightSetError::Backend(format!("shape mismatch for {}", self.tensor_name)));
        }
        let target = mounted.backend_mut().tensor_mut(&self.tensor_name)?;
        target.clone_from_slice(tensor.values());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArchitectureId, DType, MemoryWeightBackend, TensorSpec, WeightSetHeader, WeightSetManifest, WeightSetVersion};

    fn manifest() -> WeightSetManifest {
        WeightSetManifest::new(
            WeightSetHeader::new(WeightSetId::new("student"), WeightSetVersion::new("0.1.0").unwrap(), ArchitectureId::new("noworodek-v0")),
            vec![TensorSpec::new("x", vec![2, 2], DType::F32, "x")],
        ).unwrap()
    }

    #[test]
    fn handle_reads_external_tensor_from_weightset() {
        let mut manager = WeightSetManager::new(ArchitectureId::new("noworodek-v0"));
        let id = manager.mount(Box::new(MemoryWeightBackend::with_tensor_data(manifest(), [("x", vec![1.0,2.0,3.0,4.0])]))) .unwrap();
        let handle = ParameterHandle::new(id, "x").unwrap();
        assert_eq!(handle.read(&manager).unwrap().values(), &[1.0,2.0,3.0,4.0]);
    }

    #[test]
    fn handle_writes_back_without_owning_model_parameters() {
        let mut manager = WeightSetManager::new(ArchitectureId::new("noworodek-v0"));
        let id = manager.mount(Box::new(MemoryWeightBackend::from_manifest(manifest()))).unwrap();
        let handle = ParameterHandle::new(id.clone(), "x").unwrap();
        let tensor = Tensor::from_vec(vec![2,2], vec![9.0,8.0,7.0,6.0]).unwrap();
        handle.write(&mut manager, &tensor).unwrap();
        assert_eq!(handle.read(&manager).unwrap().values(), &[9.0,8.0,7.0,6.0]);
        assert!(manager.active(&id).unwrap().backend().tensor("x").is_ok());
    }
}
