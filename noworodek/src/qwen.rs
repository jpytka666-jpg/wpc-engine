use crate::{DType, ParameterRegistry, RegistryError};

/// Build an externalized Transformer registry whose vocabulary is exactly the
/// pinned Qwen3-Coder-30B-A3B tokenizer vocabulary.
pub fn qwen3_coder_registry(
    layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
    dtype: DType,
) -> Result<ParameterRegistry, RegistryError> {
    let mut registry = ParameterRegistry::new();
    registry.register_decoder_transformer(
        layers,
        crate::tokenizer::VOCAB_SIZE as usize,
        hidden_size,
        intermediate_size,
        dtype,
    )?;
    Ok(registry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_bridge_uses_pinned_vocab() {
        let registry = qwen3_coder_registry(1, 32, 64, DType::F32).unwrap();
        assert_eq!(
            registry.get("model.embeddings.token.weight").unwrap().shape[0],
            crate::tokenizer::VOCAB_SIZE as usize
        );
        assert_eq!(
            registry.get("model.lm_head.weight").unwrap().shape[0],
            crate::tokenizer::VOCAB_SIZE as usize
        );
    }
}
