use crate::{ParameterHandle, Tensor, WeightSetError, WeightSetId, WeightSetManager};

#[derive(Clone, Debug)]
pub struct TinyTransformerConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub sequence_length: usize,
}

#[derive(Debug)]
pub struct ExternalTransformer {
    pub config: TinyTransformerConfig,
    pub weight_set: WeightSetId,
}

impl ExternalTransformer {
    pub fn new(config: TinyTransformerConfig, weight_set: WeightSetId) -> Self {
        Self { config, weight_set }
    }

    fn handle(&self, name: String) -> Result<ParameterHandle, WeightSetError> {
        ParameterHandle::new(self.weight_set.clone(), name)
    }

    pub fn embedding(&self, manager: &WeightSetManager, token_ids: &[usize]) -> Result<Tensor, WeightSetError> {
        if token_ids.len() > self.config.sequence_length {
            return Err(WeightSetError::Backend("sequence length exceeds transformer config".into()));
        }
        let embedding = self.handle("model.embeddings.token.weight".into())?.read(manager)?;
        if embedding.shape() != [self.config.vocab_size, self.config.hidden_size] {
            return Err(WeightSetError::Backend("embedding shape does not match config".into()));
        }
        let mut values = Vec::with_capacity(token_ids.len() * self.config.hidden_size);
        for &token in token_ids {
            if token >= self.config.vocab_size {
                return Err(WeightSetError::Backend(format!("token id {token} out of range")));
            }
            let start = token * self.config.hidden_size;
            values.extend_from_slice(&embedding.values()[start..start + self.config.hidden_size]);
        }
        Tensor::from_vec(vec![token_ids.len(), self.config.hidden_size], values)
    }

    pub fn forward_single_layer(&self, manager: &WeightSetManager, token_ids: &[usize]) -> Result<Tensor, WeightSetError> {
        let hidden = self.embedding(manager, token_ids)?;
        let prefix = "model.layers.00";
        let q = hidden.matmul(&self.handle(format!("{prefix}.attention.q_proj.weight"))?.read(manager)?)?;
        let k = hidden.matmul(&self.handle(format!("{prefix}.attention.k_proj.weight"))?.read(manager)?)?;
        let v = hidden.matmul(&self.handle(format!("{prefix}.attention.v_proj.weight"))?.read(manager)?)?;
        let attended = causal_attention(&q, &k, &v)?;
        let projected = attended.matmul(&self.handle(format!("{prefix}.attention.o_proj.weight"))?.read(manager)?)?;
        hidden.add(&projected)
    }

    pub fn logits(&self, manager: &WeightSetManager, token_ids: &[usize]) -> Result<Tensor, WeightSetError> {
        let hidden = self.forward_single_layer(manager, token_ids)?;
        let lm_head = self.handle("model.lm_head.weight".into())?.read(manager)?;
        hidden.matmul(&transpose_2d(&lm_head)?)
    }
}

fn transpose_2d(input: &Tensor) -> Result<Tensor, WeightSetError> {
    if input.shape().len() != 2 { return Err(WeightSetError::Backend("transpose requires rank-2 tensor".into())); }
    let (rows, cols) = (input.shape()[0], input.shape()[1]);
    let mut values = vec![0.0; rows * cols];
    for r in 0..rows { for c in 0..cols { values[c * rows + r] = input.values()[r * cols + c]; } }
    Tensor::from_vec(vec![cols, rows], values)
}

fn causal_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor, WeightSetError> {
    if q.shape() != k.shape() || k.shape() != v.shape() || q.shape().len() != 2 {
        return Err(WeightSetError::Backend("attention tensors must have equal rank-2 shapes".into()));
    }
    let seq = q.shape()[0];
    let hidden = q.shape()[1];
    let scale = (hidden as f32).sqrt();
    let mut output = vec![0.0; seq * hidden];

    for i in 0..seq {
        let begin = 0usize;
        let end = i + 1;
        let mut scores = Vec::with_capacity(end);
        for j in begin..end {
            let mut dot = 0.0;
            for d in 0..hidden { dot += q.values()[i * hidden + d] * k.values()[j * hidden + d]; }
            scores.push(dot / scale);
        }
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut weights = scores.iter().map(|s| (s - max_score).exp()).collect::<Vec<_>>();
        let total: f32 = weights.iter().sum();
        for w in &mut weights { *w /= total.max(f32::MIN_POSITIVE); }
        for (offset, &weight) in weights.iter().enumerate() {
            let j = begin + offset;
            for d in 0..hidden { output[i * hidden + d] += weight * v.values()[j * hidden + d]; }
        }
    }
    Tensor::from_vec(vec![seq, hidden], output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArchitectureId, DType, MemoryWeightBackend, ParameterRegistry, TensorSpec, WeightSetHeader, WeightSetManifest, WeightSetVersion};

    fn manifest() -> WeightSetManifest {
        let mut registry = ParameterRegistry::new();
        registry.register_decoder_transformer(1, 8, 4, 8, DType::F32).unwrap();
        registry.to_manifest(WeightSetId::new("student"), WeightSetVersion::new("0.1.0").unwrap(), ArchitectureId::new("noworodek-v0")).unwrap()
    }

    fn backend() -> MemoryWeightBackend {
        let manifest = manifest();
        let mut data = Vec::new();
        data.push(("model.embeddings.token.weight", vec![0.1; 32]));
        for name in [
            "model.layers.00.attention.q_proj.weight",
            "model.layers.00.attention.k_proj.weight",
            "model.layers.00.attention.v_proj.weight",
            "model.layers.00.attention.o_proj.weight",
        ] { data.push((name, vec![0.2; 16])); }
        data.push(("model.lm_head.weight", vec![0.3; 32]));
        MemoryWeightBackend::with_tensor_data(manifest, data)
    }

    #[test]
    fn external_transformer_reads_embedding_from_weightset() {
        let mut manager = WeightSetManager::new(ArchitectureId::new("noworodek-v0"));
        let id = manager.mount(Box::new(backend())).unwrap();
        let model = ExternalTransformer::new(TinyTransformerConfig { vocab_size: 8, hidden_size: 4, sequence_length: 4 }, id);
        let hidden = model.embedding(&manager, &[0, 3]).unwrap();
        assert_eq!(hidden.shape(), &[2, 4]);
    }

    #[test]
    fn single_layer_forward_keeps_sequence_and_hidden_shape() {
        let mut manager = WeightSetManager::new(ArchitectureId::new("noworodek-v0"));
        let id = manager.mount(Box::new(backend())).unwrap();
        let model = ExternalTransformer::new(TinyTransformerConfig { vocab_size: 8, hidden_size: 4, sequence_length: 4 }, id);
        let hidden = model.forward_single_layer(&manager, &[0, 1, 2]).unwrap();
        assert_eq!(hidden.shape(), &[3, 4]);
    }

    #[test]
    fn logits_are_produced_from_external_lm_head() {
        let mut manager = WeightSetManager::new(ArchitectureId::new("noworodek-v0"));
        let id = manager.mount(Box::new(backend())).unwrap();
        let model = ExternalTransformer::new(TinyTransformerConfig { vocab_size: 8, hidden_size: 4, sequence_length: 4 }, id);
        let logits = model.logits(&manager, &[0, 1]).unwrap();
        assert_eq!(logits.shape(), &[2, 8]);
    }
}
