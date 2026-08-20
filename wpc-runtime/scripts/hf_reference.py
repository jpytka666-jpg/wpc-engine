#!/usr/bin/env python3
"""Ground-truth greedy decode for Qwen2.5-0.5B, run with the real HF
transformers implementation (CPU, torch 2.13). Used to validate the Rust
wpc-runtime engine's forward pass token-for-token.

Usage (with /home/owner/.aions/venv activated or invoked directly):
    /home/owner/.aions/venv/bin/python3 hf_reference.py \
        --model /home/owner/models/qwen2.5-0.5b \
        --prompt "The capital of France is" \
        --max-tokens 20
"""
import argparse
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=20)
    args = ap.parse_args()

    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=config, torch_dtype=torch.float32
    )
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt")
    prompt_ids = inputs["input_ids"][0].tolist()
    print(f"prompt tokens ({len(prompt_ids)}): {prompt_ids}")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    generated_ids = out[0][len(prompt_ids):].tolist()
    print(f"generated token ids: {generated_ids}")
    print("generated text:")
    print(tokenizer.decode(generated_ids, skip_special_tokens=True))

    # Also dump raw logits for the very first generated-token step, useful
    # for float-tolerance comparison beyond just argmax ids.
    with torch.no_grad():
        first_logits = model(**inputs).logits[0, -1, :]
    top5 = torch.topk(first_logits, 5)
    print("first-step top5 logits:", list(zip(top5.indices.tolist(), top5.values.tolist())))


if __name__ == "__main__":
    main()
