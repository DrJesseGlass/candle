#!/usr/bin/env python3
"""
Validation script for SmolLM3 - outputs token IDs for debugging.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM3-3B",
                        help="Model ID")
    parser.add_argument("--prompt", default="Write a haiku about debugging",
                        help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature")
    parser.add_argument("--seed", type=int, default=299792458,
                        help="Random seed")
    parser.add_argument("--no-chat", action="store_true",
                        help="Skip chat template (raw prompt)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU (avoid CUDA OOM)")

    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    print(f"Loading {args.model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        device_map=device,
    )

    # Format prompt
    if args.no_chat:
        final_prompt = args.prompt
        print(f"\n=== RAW PROMPT (no chat template) ===")
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": args.prompt}
        ]
        final_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"\n=== WITH CHAT TEMPLATE ===")
        print(f"Formatted:\n{repr(final_prompt)}\n")

    print(f"Prompt: {final_prompt}\n")

    # Tokenize
    inputs = tokenizer(final_prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"][0]

    print(f"Input token IDs ({len(input_ids)} tokens):")
    print(input_ids.tolist())
    print()

    # Generate
    print(f"Generating {args.max_tokens} tokens (temp={args.temperature})...\n")
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=args.max_tokens,
            do_sample=(args.temperature > 0),
            temperature=args.temperature if args.temperature > 0 else 1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode
    generated_ids = outputs[0][len(input_ids):]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("=" * 80)
    print("OUTPUT")
    print("=" * 80)
    print(f"\nFull text:\n{full_text}\n")
    print(f"Generated text:\n{generated_text}\n")
    print(f"Generated token IDs ({len(generated_ids)} tokens):")
    print(generated_ids.tolist())
    print()

    # For debugging - show first few tokens
    print("First 10 generated tokens:")
    for i, tok_id in enumerate(generated_ids[:10].tolist()):
        tok_text = tokenizer.decode([tok_id])
        print(f"  {i}: {tok_id:6d} -> {repr(tok_text)}")

    print("\n" + "=" * 80)
    print("COPY THIS FOR RUST COMPARISON:")
    print("=" * 80)
    print(f"\nInput tokens: vec![{', '.join(map(str, input_ids.tolist()))}];")
    print(f"\nExpected output tokens: vec![{', '.join(map(str, generated_ids.tolist()))}];")
    print()


if __name__ == "__main__":
    main()