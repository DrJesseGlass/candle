#!/usr/bin/env python3
"""
Validation script for SmolLM3 candle implementation.
Compares outputs between HuggingFace transformers and candle.

Usage:
    python validate_smollm3.py --prompt "The capital of France is" --max-tokens 20
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


def generate_reference(
        model_id: str,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 0.0,  # Deterministic by default
        top_p: float = 1.0,
        seed: int = 299792458,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Generate reference output using HuggingFace transformers."""

    print(f"Loading model: {model_id}")
    print(f"Device: {device}")

    # Set seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
    )

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    print(f"\nPrompt: {prompt}")
    print(f"Input tokens: {input_ids[0].tolist()}")
    print(f"Input length: {len(input_ids[0])}")

    # Generate
    print(f"\nGenerating with:")
    print(f"  - max_new_tokens: {max_new_tokens}")
    print(f"  - temperature: {temperature}")
    print(f"  - top_p: {top_p}")
    print(f"  - seed: {seed}")

    with torch.no_grad():
        if temperature == 0.0:
            # Greedy decoding for deterministic output
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            # Sampling
            outputs = model.generate(
                input_id,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    # Decode output
    generated_ids = outputs[0][len(input_ids[0]):]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n" + "=" * 80)
    print("REFERENCE OUTPUT (Python/Transformers)")
    print("=" * 80)
    print(f"\nFull output:\n{full_text}")
    print(f"\nGenerated only:\n{generated_text}")
    print(f"\nGenerated token IDs: {generated_ids.tolist()}")
    print(f"Number of tokens generated: {len(generated_ids)}")
    print("=" * 80)

    return {
        "full_text": full_text,
        "generated_text": generated_text,
        "input_ids": input_ids[0].tolist(),
        "generated_ids": generated_ids.tolist(),
        "all_ids": outputs[0].tolist(),
    }


def compare_outputs(reference_ids: list, candle_ids: list):
    """Compare token IDs from reference and candle implementations."""

    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    min_len = min(len(reference_ids), len(candle_ids))
    max_len = max(len(reference_ids), len(candle_ids))

    matches = 0
    first_mismatch = None

    print(f"\nReference length: {len(reference_ids)}")
    print(f"Candle length: {len(candle_ids)}")

    print("\nToken-by-token comparison:")
    print(f"{'Pos':<5} {'Reference':<10} {'Candle':<10} {'Match':<10}")
    print("-" * 45)

    for i in range(max_len):
        ref_tok = reference_ids[i] if i < len(reference_ids) else "N/A"
        can_tok = candle_ids[i] if i < len(candle_ids) else "N/A"

        if ref_tok == can_tok:
            match = "✓"
            matches += 1
        else:
            match = "✗"
            if first_mismatch is None:
                first_mismatch = i

        print(f"{i:<5} {str(ref_tok):<10} {str(can_tok):<10} {match:<10}")

    accuracy = matches / max_len * 100 if max_len > 0 else 0

    print("\n" + "-" * 45)
    print(f"Matching tokens: {matches}/{max_len} ({accuracy:.1f}%)")

    if first_mismatch is not None:
        print(f"First mismatch at position: {first_mismatch}")
    else:
        print("✓ All tokens match!")

    print("=" * 80)

    return matches == max_len


def main():
    parser = argparse.ArgumentParser(description="Validate SmolLM3 candle implementation")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM3-3B-Base",
                        help="Model ID on HuggingFace")
    parser.add_argument("--prompt", default="The capital of France is",
                        help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=20,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0.0 for greedy)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-p sampling")
    parser.add_argument("--seed", type=int, default=299792458,
                        help="Random seed")
    parser.add_argument("--candle-output", type=str,
                        help="Path to candle output file (optional)")

    args = parser.parse_args()

    # Generate reference output
    result = generate_reference(
        model_id=args.model_id,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    print("\n" + "=" * 80)
    print("INSTRUCTIONS FOR CANDLE")
    print("=" * 80)
    print("\nRun your candle implementation with these EXACT parameters:")
    print(f"""
cargo run --release --example smollm3 -- \\
  --model 3b-base \\
  --prompt "{args.prompt}" \\
  --sample-len {args.max_tokens} \\
  --temperature {args.temperature} \\
  --top-p {args.top_p} \\
  --seed {args.seed} \\
  --repeat-penalty 1.0
""")

    if args.temperature == 0.0:
        print("\nNOTE: Temperature is 0.0, so we're using greedy decoding.")
        print("      The outputs should be EXACTLY the same (token-for-token).")
    else:
        print(f"\nNOTE: Temperature is {args.temperature}, so outputs may differ due to sampling.")
        print("      Check that the outputs are reasonable and similar in quality.")

    print("\nExpected generated token IDs:")
    print(result["generated_ids"])

    # If candle output provided, compare
    if args.candle_output:
        print(f"\nReading candle output from: {args.candle_output}")
        with open(args.candle_output, 'r') as f:
            candle_ids = [int(x.strip()) for x in f.read().strip().split(',')]
        compare_outputs(result["generated_ids"], candle_ids)


if __name__ == "__main__":
    main()