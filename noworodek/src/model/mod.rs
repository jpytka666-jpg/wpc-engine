pub mod transformer;

//! Externalized parameter registry for the decoder Transformer.

use crate::{ArchitectureId, DType, TensorSpec, WeightSetId, WeightSetManifest, WeightSetVersion};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformerTensorRole { TokenEmbedding, AttentionQ, AttentionK, AttentionV, AttentionOutput, MlpUp, MlpGate, MlpDown, AttentionNorm, MlpNorm, FinalNorm, LanguageModelHead }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterRegistration { pub name: String, pub role: TransformerTensorRole, pub shape: Vec<usize>, pub dtype: DType, pub trainable: bool }
impl ParameterRegistration {
    pub fn new(name: impl Into<String>, role: TransformerTensorRole, shape: Vec<usize>, dtype: DType) -> Result<Self, RegistryError> {
        let name=name.into(); if name.is_empty(){return Err(RegistryError::EmptyName);} if shape.is_empty()||shape.iter().any(|d|*d==0){return Err(RegistryError::InvalidShape(name));} Ok(Self{name,role,shape,dtype,trainable:true})
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError { EmptyName, InvalidShape(String), DuplicateName(String), EmptyArchitecture, EmptyWeightSet }
#[derive(Debug, Clone, Default)]
pub struct ParameterRegistry { parameters: Vec<ParameterRegistration> }
impl ParameterRegistry {
    pub fn new()->Self{Self::default()}
    pub fn register(&mut self,parameter:ParameterRegistration)->Result<(),RegistryError>{if self.parameters.iter().any(|p|p.name==parameter.name){return Err(RegistryError::DuplicateName(parameter.name));}self.parameters.push(parameter);Ok(())}
    pub fn get(&self,name:&str)->Option<&ParameterRegistration>{self.parameters.iter().find(|p|p.name==name)}
    pub fn len(&self)->usize{self.parameters.len()}
    pub fn is_empty(&self)->bool{self.parameters.is_empty()}
    pub fn iter(&self)->impl Iterator<Item=&ParameterRegistration>{self.parameters.iter()}
    pub fn to_manifest(&self,weight_set_id:WeightSetId,version:WeightSetVersion,architecture:ArchitectureId)->Result<WeightSetManifest,RegistryError>{if weight_set_id.as_str().is_empty(){return Err(RegistryError::EmptyWeightSet);}if architecture.as_str().is_empty(){return Err(RegistryError::EmptyArchitecture);}let specs=self.parameters.iter().map(|p|TensorSpec::new(p.name.clone(),p.shape.clone(),p.dtype,"UNCOMMITTED")).collect();WeightSetManifest::new(crate::weightset::WeightSetHeader::new(weight_set_id,version,architecture).with_capabilities(["externalized-parameters","observable","editable"]).with_provenance("noworodek-transformer-parameter-registry"),specs).map_err(|e|RegistryError::InvalidShape(e.to_string()))}
    pub fn register_decoder_transformer(&mut self,layers:usize,vocab_size:usize,hidden_size:usize,intermediate_size:usize,dtype:DType)->Result<(),RegistryError>{self.register(ParameterRegistration::new("model.embeddings.token.weight",TransformerTensorRole::TokenEmbedding,vec![vocab_size,hidden_size],dtype)?)?;for layer in 0..layers{let p=|suffix:&str|format!("model.layers.{layer:02}.{suffix}");self.register(ParameterRegistration::new(p("attention.q_proj.weight"),TransformerTensorRole::AttentionQ,vec![hidden_size,hidden_size],dtype)?)?;self.register(ParameterRegistration::new(p("attention.k_proj.weight"),TransformerTensorRole::AttentionK,vec![hidden_size,hidden_size],dtype)?)?;self.register(ParameterRegistration::new(p("attention.v_proj.weight"),TransformerTensorRole::AttentionV,vec![hidden_size,hidden_size],dtype)?)?;self.register(ParameterRegistration::new(p("attention.o_proj.weight"),TransformerTensorRole::AttentionOutput,vec![hidden_size,hidden_size],dtype)?)?;self.register(ParameterRegistration::new(p("mlp.up_proj.weight"),TransformerTensorRole::MlpUp,vec![intermediate_size,hidden_size],dtype)?)?;self.register(ParameterRegistration::new(p("mlp.gate_proj.weight"),TransformerTensorRole::MlpGate,vec![intermediate_size,hidden_size],dtype)?)?;self.register(ParameterRegistration::new(p("mlp.down_proj.weight"),TransformerTensorRole::MlpDown,vec![hidden_size,intermediate_size],dtype)?)?;self.register(ParameterRegistration::new(p("attention_norm.weight"),TransformerTensorRole::AttentionNorm,vec![hidden_size],dtype)?)?;self.register(ParameterRegistration::new(p("mlp_norm.weight"),TransformerTensorRole::MlpNorm,vec![hidden_size],dtype)?)?;}self.register(ParameterRegistration::new("model.final_norm.weight",TransformerTensorRole::FinalNorm,vec![hidden_size],dtype)?)?;self.register(ParameterRegistration::new("model.lm_head.weight",TransformerTensorRole::LanguageModelHead,vec![vocab_size,hidden_size],dtype)?)?;Ok(())}
}
#[cfg(test)]
mod tests{use super::*;#[test]fn registration_rejects_duplicate_names(){let mut r=ParameterRegistry::new();let p=ParameterRegistration::new("x",TransformerTensorRole::MlpUp,vec![2,2],DType::F32).unwrap();r.register(p.clone()).unwrap();assert_eq!(r.register(p),Err(RegistryError::DuplicateName("x".into())));}#[test]fn transformer_registry_contains_every_trainable_family(){let mut r=ParameterRegistry::new();r.register_decoder_transformer(2,128,32,64,DType::F32).unwrap();for n in ["model.embeddings.token.weight","model.layers.00.attention.q_proj.weight","model.layers.00.attention.k_proj.weight","model.layers.00.attention.v_proj.weight","model.layers.00.attention.o_proj.weight","model.layers.00.mlp.up_proj.weight","model.layers.00.mlp.gate_proj.weight","model.layers.00.mlp.down_proj.weight","model.layers.00.attention_norm.weight","model.layers.00.mlp_norm.weight","model.final_norm.weight","model.lm_head.weight"]{assert!(r.get(n).is_some());}assert_eq!(r.len(),20);}#[test]fn registry_exports_external_weight_manifest(){let mut r=ParameterRegistry::new();r.register_decoder_transformer(1,64,16,32,DType::F32).unwrap();let m=r.to_manifest(WeightSetId::new("core"),WeightSetVersion::new("0.1.0").unwrap(),ArchitectureId::new("noworodek-decoder-v1")).unwrap();assert_eq!(m.architecture().as_str(),"noworodek-decoder-v1");assert_eq!(m.tensors().len(),r.len());assert!(m.capabilities().contains(&"externalized-parameters".to_string()));}}

pub use transformer::{ExternalTransformer,TinyTransformerConfig};
