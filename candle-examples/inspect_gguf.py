#!/usr/bin/env python3
"""
Inspect GGUF metadata keys and values.
Install: pip install gguf
"""

import sys
from gguf import GGUFReader


def inspect_gguf(filename):
    reader = GGUFReader(filename)

    print(f"=== GGUF Metadata for {filename} ===\n")

    # Get all metadata
    metadata = {}
    for field in reader.fields.values():
        if hasattr(field, 'parts'):
            for part in field.parts:
                metadata[part.name] = part.data

    # Sort keys for easier reading
    for key in sorted(metadata.keys()):
        value = metadata[key]
        print(f"{key}: {value}")

    print(f"\n=== Architecture-specific keys ===")
    # Filter for architecture-specific keys
    for key in sorted(metadata.keys()):
        if any(arch in key.lower() for arch in ['qwen', 'smol', 'llama', 'mistral']):
            print(f"{key}: {metadata[key]}")

    print(f"\n=== Attention-related keys ===")
    for key in sorted(metadata.keys()):
        if 'attention' in key.lower() or 'head' in key.lower():
            print(f"{key}: {metadata[key]}")

    print(f"\n=== RoPE-related keys ===")
    for key in sorted(metadata.keys()):
        if 'rope' in key.lower() or 'position' in key.lower():
            print(f"{key}: {metadata[key]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_gguf.py <model.gguf>")
        sys.exit(1)

    inspect_gguf(sys.argv[1])