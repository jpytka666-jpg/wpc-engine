# Noworodek — Qwen3-Coder tokenizer

Noworodek v1 uses the **Qwen/Qwen3-Coder-30B-A3B-Instruct** tokenizer as its tokenizer contract only. No Qwen model weights are used by this component.

## Pin

- Model: `Qwen/Qwen3-Coder-30B-A3B-Instruct`
- Revision: `573fa3901e5799703b1e60825b0ec024a4c0f1d3`
- Vocabulary size: `151936`
- EOS: `<|im_end|>` = `151645`
- PAD: `<|endoftext|>` = `151643`
- `<|im_start|>` = `151644`
- Maximum declared token length: `1048576`

Qwen's current tokenizer configuration defines these special tokens and the chat/tool serialization used by the model family. Qwen also explicitly notes that the updated tokenizer must be used for Qwen3-Coder function calling.

## Runtime

The Rust crate uses Hugging Face `tokenizers` `0.23.1`. The tokenizer is loaded from a local `tokenizer.json` so experiments remain reproducible and do not silently switch tokenizer revisions.

Example preparation:

```text
huggingface-cli download Qwen/Qwen3-Coder-30B-A3B-Instruct \\
  tokenizer.json tokenizer_config.json \\
  --revision 573fa3901e5799703b1e60825b0ec024a4c0f1d3 \\
  --local-dir ./artifacts/tokenizer/qwen3-coder-30b
```

The application should then construct `Qwen3CoderTokenizer::from_file(".../tokenizer.json")` and verify the vocabulary size before training.

## Architectural rule

The tokenizer defines the input vocabulary and serialization contract. The embedding matrix, Transformer blocks, LM head, and every other trainable parameter remain Noworodek-owned external `WeightSet` tensors.
