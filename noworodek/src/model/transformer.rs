use crate::{ParameterHandle, Tensor, WeightSetError, WeightSetId, WeightSetManager};

#[derive(Clone, Debug)]
pub struct TinyTransformerConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub sequence_length: usize,
    pub num_layers: usize,
    pub rms_norm_eps: f32,
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

    fn handle(&self, name: impl Into<String>) -> Result<ParameterHandle, WeightSetError> {
        ParameterHandle::new(self.weight_set.clone(), name.into())
    }

    pub fn embedding(&self, manager: &WeightSetManager, token_ids: &[usize]) -> Result<Tensor, WeightSetError> {
        self.validate_sequence(token_ids.len())?;
        let embedding = self.handle("model.embeddings.token.weight")?.read(manager)?;
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

    /// Full decoder path using only external WeightSet tensors.
    pub fn forward(&self, manager: &WeightSetManager, token_ids: &[usize]) -> Result<Tensor, WeightSetError> {
        self.validate_sequence(token_ids.len())?;
        let mut hidden = self.embedding(manager, token_ids)?;
        for layer in 0..self.config.num_layers {
            hidden = self.forward_block(manager, &hidden, layer)?;
        }
        hidden = self.rms_norm(manager, &hidden, "model.final_norm.weight")?;
        let lm_head = self.handle("model.lm_head.weight")?.read(manager)?;
        if lm_head.shape() != [self.config.vocab_size, self.config.hidden_size] {
            return Err(WeightSetError::Backend("lm head shape does not match config".into()));
        }
        hidden.matmul(&transpose_2d(&lm_head)?)
    }

    pub fn forward_single_layer(&self, manager: &WeightSetManager, token_ids: &[usize]) -> Result<Tensor, WeightSetError> {
        self.validate_sequence(token_ids.len())?;
        let hidden = self.embedding(manager, token_ids)?;
        self.forward_block(manager, &hidden, 0)
    }

    pub fn logits(&self, manager: &WeightSetManager, token_ids: &[usize]) -> Result<Tensor, WeightSetError> {
        self.forward(manager, token_ids)
    }

    fn forward_block(&self, manager: &WeightSetManager, hidden: &Tensor, layer: usize) -> Result<Tensor, WeightSetError> {
        if layer >= self.config.num_layers {
            return Err(WeightSetError::Backend(format!("layer {layer} out of range")));
        }
        let prefix = format!("model.layers.{layer:02}");
        let normed = self.rms_norm(manager, hidden, &format!("{prefix}.attention_norm.weight"))?;
        let q = normed.matmul(&self.handle(format!("{prefix}.attention.q_proj.weight"))?.read(manager)?)?;
        let k = normed.matmul(&self.handle(format!("{prefix}.attention.k_proj.weight"))?.read(manager)?)?;
        let v = normed.matmul(&self.handle(format!("{prefix}.attention.v_proj.weight"))?.read(manager)?)?;
        let attended = causal_attention(&q, &k, &v)?;
        let projected = attended.matmul(&self.handle(format!("{prefix}.attention.o_proj.weight"))?.read(manager)?)?;
        let residual = hidden.add(&projected)?;

        let mlp_normed = self.rms_norm(manager, &residual, &format!("{prefix}.mlp_norm.weight"))?;
        let gate = mlp_normed.matmul(&self.handle(format!("{prefix}.mlp.gate_proj.weight"))?.read(manager)?)?;
        let up = mlp_normed.matmul(&self.handle(format!("{prefix}.mlp.up_proj.weight"))?.read(manager)?)?;
        let gated = silu(&gate)?.hadamard(&up)?;
        let down = gated.matmul(&self.handle(format!("{prefix}.mlp.down_proj.weight"))?.read(manager)?)?;
        residual.add(&down)
    }

    fn rms_norm(&self, manager: &WeightSetManager, input: &Tensor, weight_name: &str) -> Result<Tensor, WeightSetError> {
        let weight = self.handle(weight_name.to_string())?.read(manager)?;
        if weight.shape() != [self.config.hidden_size] {
            return Err(WeightSetError::Backend(format!("norm shape does not match hidden size: {weight_name}")));
        }
        if input.shape().len() != 2 || input.shape()[1] != self.config.hidden_size {
            return Err(WeightSetError::Backend("rms norm requires [sequence, hidden]".into()));
        }
        let seq = input.shape()[0];
        let hidden = input.shape()[1];
        let mut out = vec![0.0; seq * hidden];
        for row in 0..seq {
            let offset = row * hidden;
            let mut mean_sq = 0.0;
            for d in 0..hidden {
                let x = input.values()[offset + d];
                mean_sq += x * x;
            }
            mean_sq /= hidden as f32;
            let inv = (mean_sq + self.config.rms_norm_eps).sqrt().recip();
            for d in 0..hidden {
                out[offset + d] = input.values()[offset + d] * inv * weight.values()[d];
            }
        }
        Tensor::from_vec(vec![seq, hidden], out)
    }

    fn validate_sequence(&self, len: usize) -> Result<(), WeightSetError> {
        if len == 0 {
            return Err(WeightSetError::Backend("empty token sequence".into()));
        }
        if len > self.config.sequence_length {
            return Err(WeightSetError::Backend("sequence length exceeds transformer config".into()));
        }
        Ok(())
    }
}

fn transpose_2d(input: &Tensor) -> Result<Tensor, WeightSetError> {
    if input.shape().len() != 2 { return Err(WeightSetError::Backend("transpose requires rank-2 tensor".into())); }
    let (rows, cols) = (input.shape()[0], input.shape()[1]);
    let mut values = vec![0.0; rows * cols];
    for r in 0..rows { for c in 0..cols { values[c * rows + r] = input.values()[r * cols + c]; } }
    Tensor::from_vec(vec![cols, rows], values)
}

fn silu(input: &Tensor) -> Result<Tensor, WeightSetError> {
    let values = input.values().iter().map(|x| *x / (1.0 + (-*x).exp())).collect();
    Tensor::from_vec(input.shape().to_vec(), values)
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
        let mut scores = Vec::with_capacity(i + 1);
        for j in 0..=i {
            let mut dot = 0.0;
            for d in 0..hidden { dot += q.values()[i * hidden + d] * k.values()[j * hidden + d]; }
            scores.push(dot / scale);
        }
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut weights = scores.iter().map(|s| (s - max_score).exp()).collect::<Vec<_>>();
        let total: f32 = weights.iter().sum();
        for w in &mut weights { *w /= total.max(f32::MIN_POSITIVE); }
        for (j, &weight) in weights.iter().enumerate() {
            for d in 0..hidden { output[i * hidden + d] += weight * v.values()[j * hidden + d]; }
        }
    }
    Tensor::from_vec(vec![seq, hidden], output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArchitectureId, DType, MemoryWeightBackend, ParameterRegistry, WeightSetHeader, WeightSetManifest, WeightSetVersion};

    fn manifest() -> WeightSetManifest {
        let mut registry = ParameterRegistry::new();
        registry.register_decoder_transformer(2, 8, 4, 8, DType::F32).unwrap();
        registry.to_manifest(WeightSetId::new("student"), WeightSetVersion::new("0.1.0").unwrap(), ArchitectureId::new("noworodek-v0")).unwrap()
    }

    fn backend() -> MemoryWeightBackend {
        let manifest = manifest();
        let mut data = Vec::new();
        data.push(("model.embeddings.token.weight", vec![0.1; 32]));
        for layer in 0..2 {
            let prefix = format!("model.layers.{layer:02}");
            for name in [
                format!("{prefix}.attention.q_proj.weight"),
                format!("{prefix}.attention.k_proj.weight"),
                format!("{prefix}.attention.v_proj.weight"),
                format!("{prefix}.attention.o_proj.weight"),
                format!("{prefix}.mlp.gate_proj.weight"),
                format!("{prefix}.mlp.up_proj.weight"),
                format!("{prefix}.mlp.down_proj.weight"),
            ] {
                let len = if name.contains("down_proj") { 32 } else if name.contains("gate_proj") || name.contains("up_proj") { 32 } else { 16 };
                data.push((Box::leak(name.into_boxed_str()) as &'static str, vec![0.02; len]));
            }
            data.push((Box::leak(format!("{prefix}.attention_norm.weight").into_boxed_str()) as &'static str, vec![1.0; 4]));
            data.push((Box::leak(format!("{prefix}.mlp_norm.weight").into_boxed_str()) as &'static str, vec![1.0; 4]));
        }
        data.push(("model.final_norm.weight", vec![1.0; 4]));
        data.push(("model.lm_head.weight", vec![0.03; 32]));
        MemoryWeightBackend::with_tensor_data(manifest, data)
    }

    fn manager_and_model() -> (WeightSetManager, ExternalTransformer) {
        let mut manager = WeightSetManager::new(ArchitectureId::new("noworodek-v0"));
        let id = manager.mount(Box::new(backend())).unwrap();
        let model = ExternalTransformer::new(TinyTransformerConfig {
            vocab_size: 8,
            hidden_size: 4,
            intermediate_size: 8,
            sequence_length: 8,
            num_layers: 2,
            rms_norm_eps: 1e-5,
        }, id);
        (manager, model)
    }

    #[test]
    fn full_decoder_forward_produces_logits() {
        let (manager, model) = manager_and_model();
        let logits = model.forward(&manager, &[0, 1, 2]).unwrap();
        assert_eq!(logits.shape(), &[3, 8]);
        assert!(logits.values().iter().all(|x| x.is_finite()));
    }

    #[test]
    fn zero_sequence_is_rejected() {
        let (manager, model) = manager_and_model();
        assert!(model.forward(&manager, &[]).is_err());
    }

    #[test]
    fn changing_external_weight_changes_logits() {
        let (mut manager, model) = manager_and_model();
        let before = model.forward(&manager, &[0, 1]).unwrap();
        let id = model.weight_set.clone();
        let mounted = manager.active_mut(&id).unwrap();
        let weight = mounted.backend_mut().tensor_mut("model.lm_head.weight").unwrap();
        weight[0] += 1.0;
        let after = model.forward(&manager, &[0, 1]).unwrap();
        assert_ne!(before.values(), after.values());
    }
}
