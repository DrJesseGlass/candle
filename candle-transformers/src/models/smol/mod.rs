//! SmolLM model family implementations.
//!
//! Note: SmolLM2 models are implemented as part of the Llama architecture
//! (see `models::llama`) since they follow the standard Llama design.
//!
//! SmolLM3 introduces NoPE (No Positional Encoding) which requires
//! a separate implementation.

pub mod smollm3;
pub mod quantized_smollm3;