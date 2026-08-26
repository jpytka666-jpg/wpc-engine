use crate::weightset::{DType, TensorSpec, WeightSetError, WeightSetId, WeightSetManifest, WeightSetVersion, ArchitectureId, WeightSetHeader};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParameterRole {
    TokenEmbedding,
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionOutput,
    AttentionNorm,
    MlpGate,
    MlpUp,
    MlpDown,
    MlpNorm,
    FinalNorm,
    LmHead,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainableParameter {
    pub name: String,
    pub role: ParameterRole,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ParameterRegistry {
    parameters: Vec<TrainableParameter>,
}

impl ParameterRegistry {
    pub fn new() -> Self { Self::default() }

    pub fn register(&mut self, parameter: TrainableParameter) -> Result<(), WeightSetError> {
        if parameter.name.is_empty() { return Err(WeightSetError::EmptyTensorName); }
        if self.parameters.iter().any(|p| p.name == parameter.name) {
            return Err(WeightSetError::DuplicateTensor(parameter.name));
        }
        self.parameters.push(parameter);
        Ok(())
    }

    pub fn parameters(&self) -> &[TrainableParameter] { &self.parameters }

    pub fn find(&self, name: &str) -> Option<&TrainableParameter> {
        self.parameters.iter().find(|p| p.name == name)
    }

    pub fn to_manifest(
        &self,
        weight_set_id: WeightSetId,
        version: WeightSetVersion,
        architecture: ArchitectureId,
        provenance: impl Into<String>,
    ) -> Result<WeightSetManifest, WeightSetError> {
        let header = WeightSetHeader::new(weight_set_id, version, architecture)
            .with_capabilities(["externalized", "addressable", "editable", "observable"])
            .with_provenance(provenance);
        let tensors = self
            .parameters
            .iter()
            .map(|p| TensorSpec::new(&p.name, p.shape.clone(), p.dtype, "unmaterialized"))
            .collect();
        WeightSetManifest::new(header, tensors)
    }

    pub fn tiny_transformer(num_layers: usize, vocab: usize, hidden: usize, intermediate: usize) -> Result<Self, WeightSetError> {
        let mut registry = Self::new();
        registry.register(TrainableParameter { name: "model.embeddings.token.weight".into(), role: ParameterRole::TokenEmbedding, shape: vec![vocab, hidden], dtype: DType::F32 })?;
        for layer in 0..num_layers {
            let p = |suffix: &str| format!("model.layers.{layer:02}.{suffix}");
            registry.register(TrainableParameter { name: p("attention.q_proj.weight"), role: ParameterRole::AttentionQ, shape: vec![hidden, hidden], dtype: DType::F32 })?;
            registry.register(TrainableParameter { name: p("attention.k_proj.weight"), role: ParameterRole::AttentionK, shape: vec![hidden, hidden], dtype: DType::F32 })?;
            registry.register(TrainableParameter { name: p("attention.v_proj.weight"), role: ParameterRole::AttentionV, shape: vec![hidden, hidden], dtype: DType::F32 })?;
            registry.register(TrainableParameter { name: p("attention.o_proj.weight"), role: ParameterRole::AttentionOutput, shape: vec![hidden, hidden], dtype: DType::F32 })?;
            registry.register(TrainableParameter { name: p("attention.norm.weight"), role: ParameterRole::AttentionNorm, shape: vec![hidden], dtype: DType::F32 })?;
            registry.register(TrainableParameter { name: p("mlp.gate_proj.weight"), role: ParameterRole::MlpGate, shape: vec![intermediate, hidden], dtype: DType::F32 })?;
            registry.register(TrainableParameter { name: p("mlp.up_proj.weight"), role: ParameterRole::MlpUp, shape: vec![intermediate, hidden], dtype: DType::F32 })?;
            registry.register(TrainableParameter { name: p("mlp.down_proj.weight"), role: ParameterRole::MlpDown, shape: vec![hidden, intermediate], dtype: DType::F32 })?;
            registry.register(TrainableParameter { name: p("mlp.norm.weight"), role: ParameterRole::MlpNorm, shape: vec![hidden], dtype: DType::F32 })?;
        }
        registry.register(TrainableParameter { name: "model.final_norm.weight".into(), role: ParameterRole::FinalNorm, shape: vec![hidden], dtype: DType::F32 })?;
        registry.register(TrainableParameter { name: "model.lm_head.weight".into(), role: ParameterRole::LmHead, shape: vec![vocab, hidden], dtype: DType::F32 })?;
        Ok(registry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_transformer_registers_full_parameter_family() {
        let registry = ParameterRegistry::tiny_transformer(2, 32, 16, 64).unwrap();
        assert_eq!(registry.parameters().len(), 21);
        assert!(registry.find("model.layers.00.attention.q_proj.weight").is_some());
        assert!(registry.find("model.layers.01.mlp.down_proj.weight").is_some());
        assert!(registry.find("model.lm_head.weight").is_some());
    }

    #[test]
    fn duplicate_parameter_ids_are_rejected() {
        let mut registry = ParameterRegistry::new();
        let p = TrainableParameter { name: "x".into(), role: ParameterRole::FinalNorm, shape: vec![4], dtype: DType::F32 };
        registry.register(p.clone()).unwrap();
        assert!(matches!(registry.register(p), Err(WeightSetError::DuplicateTensor(_))));
    }

    #[test]
    fn manifest_declares_externalized_contract() {
        let registry = ParameterRegistry::tiny_transformer(1, 16, 8, 32).unwrap();
        let manifest = registry.to_manifest(WeightSetId::new("student"), WeightSetVersion::new("0.1.0").unwrap(), ArchitectureId::new("noworodek-tiny-v0"), "registry-test").unwrap();
        assert!(manifest.capabilities().iter().any(|c| c == "externalized"));
        assert!(manifest.capabilities().iter().any(|c| c == "observable"));
        assert_eq!(manifest.tensors().len(), 12);
    }
}
