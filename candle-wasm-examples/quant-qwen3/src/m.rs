use candle::{DType, Device, Tensor};
use candle::quantized::gguf_file;
use candle_transformers::generation::LogitsProcessor;
use js_sys::Date;
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;
use std::io::Cursor;

use crate::console_log;
use candle_transformers::models::quantized_qwen3::{ModelWeights as QuantizedQwen3, ComputeMode};

#[wasm_bindgen]
pub struct Model {
    model: QuantizedQwen3,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    tokens: Vec<u32>,
    repeat_penalty: f32,
    repeat_last_n: usize,
    eos_token: u32,
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn load(
        weights: Vec<u8>,
        tokenizer: Vec<u8>,
        _config: Vec<u8>,  // Not used for GGUF, but keep for compatibility
    ) -> Result<Model, JsError> {
        console_error_panic_hook::set_once();

        // Check if SIMD is enabled
        #[cfg(feature = "simd-flash-attn")]
        console_log!("Loading quantized Qwen3 model (Q8_0) with SIMD flash attention");

        #[cfg(not(feature = "simd-flash-attn"))]
        console_log!("Loading quantized Qwen3 model (Q8_0) - standard mode");

        let device = Device::Cpu;

        console_log!("Loading tokenizer...");
        let tokenizer =
            Tokenizer::from_bytes(&tokenizer).map_err(|m| JsError::new(&m.to_string()))?;

        // Get EOS token
        let eos_token = match tokenizer.get_vocab(true).get("<|endoftext|>") {
            Some(&token) => token,
            None => match tokenizer.get_vocab(true).get("<|im_end|>") {
                Some(&token) => token,
                None => {
                    console_log!("⚠️  Warning: no EOS token found, using 0");
                    0
                }
            }
        };

        let start = Date::now();
        console_log!("Weights size: {} bytes ({:.2} MB)",
            weights.len(),
            weights.len() as f64 / 1_048_576.0
        );

        // Load GGUF quantized model with SIMD optimizations
        let mut cursor = Cursor::new(weights);
        let content = gguf_file::Content::read(&mut cursor)
            .map_err(|e| JsError::new(&format!("Failed to read GGUF: {}", e)))?;

        console_log!("GGUF file parsed, loading model weights...");

        // Use the new integrated API with optimizations
        let model = QuantizedQwen3::from_gguf_with_config(
            content,
            &mut cursor,
            &device,
            ComputeMode::ForceF32,  // Best for SIMD on WASM
            true,                    // use_flash_attn
            false,                   // cache_masks (not needed with SIMD)
        )?;

        let load_time = (Date::now() - start) / 1000.0;
        console_log!("✅ Quantized model loaded in {:.2}s", load_time);

        let logits_processor = LogitsProcessor::new(299792458, None, None);

        Ok(Self {
            model,
            tokenizer,
            tokens: vec![],
            logits_processor,
            repeat_penalty: 1.,
            repeat_last_n: 64,
            eos_token,
        })
    }

    #[wasm_bindgen]
    pub fn init_with_prompt(
        &mut self,
        prompt: String,
        temp: f64,
        top_p: f64,
        repeat_penalty: f32,
        repeat_last_n: usize,
        seed: f64,
    ) -> Result<String, JsError> {
        // Clear KV cache
        self.model.clear_kv_cache();

        let temp = if temp <= 0. { None } else { Some(temp) };
        let top_p = if top_p <= 0. || top_p >= 1. {
            None
        } else {
            Some(top_p)
        };

        let seed = seed as u64;
        self.logits_processor = LogitsProcessor::new(seed, temp, top_p);
        self.repeat_penalty = repeat_penalty;
        self.repeat_last_n = repeat_last_n;
        self.tokens.clear();

        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|m| JsError::new(&m.to_string()))?
            .get_ids()
            .to_vec();

        console_log!("Prompt encoded to {} tokens", tokens.len());

        let text = self
            .process(&tokens)
            .map_err(|m| JsError::new(&m.to_string()))?;

        Ok(text)
    }

    #[wasm_bindgen]
    pub fn next_token(&mut self) -> Result<String, JsError> {
        let last_token = *self.tokens.last().unwrap();
        let text = self
            .process(&[last_token])
            .map_err(|m| JsError::new(&m.to_string()))?;
        Ok(text)
    }

    #[wasm_bindgen]
    pub fn is_eos(&self) -> bool {
        self.tokens.last().map_or(false, |&t| t == self.eos_token)
    }

    #[wasm_bindgen]
    pub fn get_token_count(&self) -> usize {
        self.tokens.len()
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.tokens.clear();
        self.model.clear_kv_cache();
    }
}

impl Model {
    fn process(&mut self, tokens: &[u32]) -> candle::Result<String> {
        let dev = Device::Cpu;
        let input = Tensor::new(tokens, &dev)?.unsqueeze(0)?;

        // Calculate offset (position in sequence)
        let offset = self.tokens.len();

        let logits = self.model.forward(&input, offset)?;
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;

        // Apply repeat penalty if enabled
        let logits = if self.repeat_penalty == 1. {
            logits
        } else {
            let start_at = self.tokens.len().saturating_sub(self.repeat_last_n);
            let context = &self.tokens[start_at..];
            candle_transformers::utils::apply_repeat_penalty(&logits, self.repeat_penalty, context)?
        };

        let next_token = self.logits_processor.sample(&logits)?;
        self.tokens.push(next_token);

        let token = match self.tokenizer.decode(&[next_token], false) {
            Ok(token) => token,
            Err(e) => {
                console_log!("Error decoding token: {:?}", e);
                "".to_string()
            }
        };

        Ok(token)
    }
}