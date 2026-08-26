//! Externalized parameter registry for the future decoder Transformer.
//!
//! The model owns architecture and execution logic, but trainable tensors are
//! identified by stable names and registered in a WeightSet manifest. This
//! module is intentionally independent of tensor storage so memory, mmap and
//! future WPC backends can satisfy the same contract.

use crate::{ArchitectureId, DType, TensorSpec, WeightSetId, WeightSetManifest, WeightSetVersion};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformerTensorRole {
    TokenEmbedding,
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionOutput,
    MlpUp,
    MlpGate,
    MlpDown,
    AttentionNorm,
    MlpNorm,
    FinalNorm,
    LanguageModelHead,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterRegistration {
    pub name: String,
    pub role: TransformerTensorRole,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub trainable: bool,
}

impl ParameterRegistration {
    pub fn new(
        name: impl Into<String>,
        role: TransformerTensorRole,
        shape: Vec<usize>,
        dtype: DType,
    ) -> Result<Self, RegistryError> {
        let name = name.into();
        if name.is_empty() {
            return Err(RegistryError::EmptyName);
        }
        if shape.is_empty() || shape.iter().any(|d| *d == 0) {
            return Err(RegistryError::InvalidShape(name));
        }
        Ok(Self { name, role, shape, dtype, trainable: true })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    EmptyName,
    InvalidShape(String),
    DuplicateName(String),
    EmptyArchitecture,
    EmptyWeightSet,
}

#[derive(Debug, Clone, Default)]
pub struct ParameterRegistry {
    parameters: Vec<ParameterRegistration>,
}

impl ParameterRegistry {
    pub fn new() -> Self { Self::default() }

    pub fn register(&mut self, parameter: ParameterRegistration) -> Result<(), RegistryError> {
        if self.parameters.iter().any(|p| p.name == parameter.name) {
            return Err(RegistryError::DuplicateName(parameter.name));
        }
        self.parameters.push(parameter);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&ParameterRegistration> {
        self.parameters.iter().find(|p| p.name == name)
    }

    pub fn len(&self) -> usize { self.parameters.len() }
    pub fn is_empty(&self) -> bool { self.parameters.is_empty() }
    pub fn iter(&self) -> impl Iterator<Item = &ParameterRegistration> { self.parameters.iter() }

    pub fn to_manifest(
        &self,
        weight_set_id: WeightSetId,
        version: WeightSetVersion,
        architecture: ArchitectureId,
    ) -> Result<WeightSetManifest, RegistryError> {
        if weight_set_id.as_str().is_empty() { return Err(RegistryError::EmptyWeightSet); }
        if architecture.as_str().is_empty() { return Err(RegistryError::EmptyArchitecture); }

        let specs = self
            .parameters
            .iter()
            .map(|p| TensorSpec::new(p.name.clone(), p.shape.clone(), p.dtype, "UNCOMMITTED"))
            .collect();

        WeightSetManifest::new(
            crate::weightset::WeightSetHeader::new(weight_set_id, version, architecture)
                .with_capabilities(["externalized-parameters", "observable", "editable"])
                .with_provenance("noworodek-transformer-parameter-registry"),
            specs,
        ).map_err(|e| RegistryError::InvalidShape(e.to_string()))
    }

    pub fn register_decoder_transformer(
        &mut self,
        layers: usize,
        vocab_size: usize,
        hidden_size: usize,
        intermediate_size: usize,
        dtype: DType,
    ) -> Result<(), RegistryError> {
        self.register(ParameterRegistration::new(
            "model.embeddings.token.weight",
            TransformerTensorRole::TokenEmbedding,
            vec![vocab_size, hidden_size],
            dtype,
        )?)?;

        for layer in 0..layers {
            let prefix = format!("model.layers.{layer:02}");
            let q = format!("{prefix}.attention.q_proj.weight");
            let k = format!("{prefix}.attention.k_proj.weight");
            let v = format!("{prefix}.attention.v_proj.weight");
            let o = format!("{prefix}.attention.o_proj.weight");
            let up = format!("{prefix}.mlp.up_proj.weight");
            let gate = format!("{prefix}.mlp.gate_proj.weight");
            let down = format!("{prefix}.mlp.down_proj.weight");
            let an = format!("{prefix}.attention_norm.weight");
            let mn = format!("{prefix}.mlp_norm.weight");

            self.register(ParameterRegistration::new(q, TransformerTensorRole::AttentionQ, vec![hidden_size, hidden_size], dtype)?)?;
            self.register(ParameterRegistration::new(k, TransformerTensorRole::AttentionK, vec![hidden_size, hidden_size], dtype)?)?;
            self.register(ParameterRegistration::new(v, TransformerTensorRole::AttentionV, vec![hidden_size, hidden_size], dtype)?)?;
            self.register(ParameterRegistration::new(o, TransformerTensorRole::AttentionOutput, vec![hidden_size, hidden_size], dtype)?)?;
            self.register(ParameterRegistration::new(up, TransformerTensorRole::MlpUp, vec![intermediate_size, hidden_size], dtype)?)?;
            self.register(ParameterRegistration::new(gate, TransformerTensorRole::MlpGate, vec![intermediate_size, hidden_size], dtype)?)?;
            self.register(ParameterRegistration::new(down, TransformerTensorRole::MlpDown, vec![hidden_size, intermediate_size], dtype)?)?;
            self.register(ParameterRegistration::new(an, TransformerTensorRole::AttentionNorm, vec![hidden_size], dtype)?)?;
            self.register(ParameterRegistration::new(mn, TransformerTensorRole::MlpNorm, vec![hidden_size], dtype)?)?;
        }

        self.register(ParameterRegistration::new(
            "model.final_norm.weight",
            TransformerTensorRole::FinalNorm,
            vec![hidden_size],
            dtype,
        )?)?;
        self.register(ParameterRegistration::new(
            "model.lm_head.weight",
            TransformerTensorRole::LanguageModelHead,
            vec![vocab_size, hidden_size],
            dtype,
        )?)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registration_rejects_duplicate_names() {
        let mut registry = ParameterRegistry::new();
        let p = ParameterRegistration::new("x", TransformerTensorRole::MlpUp, vec![2, 2], DType::F32).unwrap();
        registry.register(p.clone()).unwrap();
        assert_eq!(registry.register(p), Err(RegistryError::DuplicateName("x".into())));
    }

    #[test]
    fn transformer_registry_contains_every_trainable_family() {
        let mut registry = ParameterRegistry::new();
        registry.register_decoder_transformer(2, 128, 32, 64, DType::F32).unwrap();
        assert!(registry.get("model.embeddings.token.weight").is_some());
        assert!(registry.get("model.layers.00.attention.q_proj.weight").is_some());
        assert!(registry.get("model.layers.00.attention.k_proj.weight").is_some());
        assert!(registry.get("model.layers.00.attention.v_proj.weight").is_some());
        assert!(registry.get("model.layers.00.attention.o_proj.weight").is_some());
        assert!(registry.get("model.layers.00.mlp.up_proj.weight").is_some());
        assert!(registry.get("model.layers.00.mlp.gate_proj.weight").is_some());
        assert!(registry.get("model.layers.00.mlp.down_proj.weight").is_some());
        assert!(registry.get("model.layers.00.attention_norm.weight").is_some());
        assert!(registry.get("model.layers.00.mlp_norm.weight").is_some());
        assert!(registry.get("model.final_norm.weight").is_some());
        assert!(registry.get("model.lm_head.weight").is_some());
        assert_eq!(registry.len(), 20);
    }

    #[test]
    fn registry_exports_external_weight_manifest() {
        let mut registry = ParameterRegistry::new();
        registry.register_decoder_transformer(1, 64, 16, 32, DType::F32).unwrap();
        let manifest = registry.to_manifest(
            WeightSetId::new("core"),
            WeightSetVersion::new("0.1.0").unwrap(),
            ArchitectureId::new("noworodek-decoder-v1"),
        ).unwrap();
        assert_eq!(manifest.architecture().as_str(), "noworodek-decoder-v1");
        assert_eq!(manifest.tensors().len(), registry.len());
        assert!(manifest.capabilities().contains(&"externalized-parameters".to_string()));
    }
}
