# SmolLM Model Family

This directory contains implementations for the SmolLM family of models
developed by HuggingFace.

## Models

### SmolLM2 (see `models/llama`)
SmolLM2 models (135M, 360M, 1.7B) use the standard Llama3 architecture 
and are implemented in `models/llama.rs`. No separate implementation 
is needed.

**Variants:**
- HuggingFaceTB/SmolLM2-135M
- HuggingFaceTB/SmolLM2-360M  
- HuggingFaceTB/SmolLM2-1.7B

### SmolLM3
SmolLM3-3B introduces NoPE (No Positional Encoding) which requires
a custom implementation in `smollm3.rs`.

**Key innovations:**
- Hybrid RoPE/NoPE (3:1 ratio)
- GQA with 4 groups
- Very high rope_theta (5M)
- Long context support (64k-128k)

### SmolVLM (planned)
Vision-language model variant, to be implemented.

## Related Models

### Granite-Docling
Document understanding VLM that originally used SmolLM-2 but now uses 
Granite 165M as its language backbone. See IBM's Docling project.