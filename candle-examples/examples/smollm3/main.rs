#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle_transformers::models::smol::smollm3::{Config, ModelForCausalLM};

use candle::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

struct TextGeneration {
    model: ModelForCausalLM,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    eos_token_id: Option<u32>,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: ModelForCausalLM,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
        eos_token_id: Option<u32>,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
            eos_token_id,
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)  // true = add special tokens (matches quantized version)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        //let mut tokens = vec![
        //    128011, 9125, 198, 567, 34689, 271, 81434, 356, 28540, 2696, 25, 5651, 220, 2366, 20,
        // //   198, 15724, 2696, 25, 220, 2304, 6841, 220, 2366, 20, 198, 26197, 287, 14904, 25, 611,
        //    2201, 5978, 771, 271, 567, 8572, 39397, 271, 2675, 527, 264, 11190, 15592, 18328, 7086,
        //    4487, 337, 11237, 11, 16572, 555, 473, 36368, 19109, 382, 128011, 882, 198, 2323, 128012,
        // /   198, 128011, 78191, 198, 128002, 271, 128003, 198
        //];
        println!("Input token IDs: {:?}", tokens);
        println!("Number of input tokens: {}", tokens.len());
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let mut generated_token_ids = Vec::new();

        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;

            // Debug: Print top 5 logits for first token
            if generated_tokens == 0 {
                let logits_vec = logits.to_vec1::<f32>()?;
                let mut indexed: Vec<(usize, f32)> = logits_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                println!("\nTop 5 logits for first token:");
                for (idx, val) in indexed.iter().take(5) {
                    println!("  Token {}: {:.4}", idx, val);
                }
            }

            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            generated_token_ids.push(next_token);

            // Check for EOS token
            if let Some(eos_id) = self.eos_token_id {
                if next_token == eos_id {
                    break;
                }
            }

            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n\n{} tokens generated ({:.2} token/s)",
            generated_tokens,
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        println!("Generated token IDs: {:?}", generated_token_ids);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, clap::ValueEnum, PartialEq, Eq)]
enum WhichModel {
    #[value(name = "3b")]
    W3b,
    #[value(name = "3b-base")]
    W3bBase,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    use_flash_attn: bool,

    #[arg(long)]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 10000)]
    sample_len: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    weight_files: Option<String>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    #[arg(long, default_value = "3b")]
    model: WhichModel,

    /// Disable chat template (for testing chat model with raw prompts)
    #[arg(long)]
    no_chat_template: bool,

    /// Data type to use (f32, f16, bf16, or auto)
    #[arg(long, default_value = "auto")]
    dtype: String,
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.),
        args.repeat_penalty,
        args.repeat_last_n
    );

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let model_id = match args.model_id {
        Some(model_id) => model_id,
        None => {
            let model_name = match args.model {
                WhichModel::W3b => "SmolLM3-3B",
                WhichModel::W3bBase => "SmolLM3-3B-Base",
            };
            format!("HuggingFaceTB/{}", model_name)
        }
    };
    println!("DEBUG: Loading model from: {}", model_id);
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision,
    ));
    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };
    let filenames = match args.weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => {
            // SmolLM3-3B uses sharded safetensors
            candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?
        }
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    // Check if chat template tokens are in vocabulary (for debugging)
    let vocab = tokenizer.get_vocab(true);
    let has_im_start = vocab.contains_key("<|im_start|>");
    let has_im_end = vocab.contains_key("<|im_end|>");
    println!("Chat template tokens in vocab: <|im_start|>={}, <|im_end|>={}", has_im_start, has_im_end);

    let start = std::time::Instant::now();
    let config_file = repo.get("config.json")?;
    let device = candle_examples::device(false)?;  // false = use GPU if available

    let dtype = match args.dtype.as_str() {
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        "f32" => DType::F32,
        "auto" => {
            // Auto-select best dtype based on device capability
            // - BF16: Preferred for modern GPUs (Ampere+: RTX 30xx/40xx, A100, H100)
            //         More stable, wider range, matches training dtype
            // - F16:  Fallback for older GPUs (Pascal/Turing: GTX 10xx, RTX 20xx)
            //         Still fast, compatible with more hardware
            // - F32:  CPU default (BF16 not well supported on CPU)
            //
            // Note: BF16 will error on GTX 10xx series. Use --dtype f16 for those GPUs.
            if device.is_cuda() || device.is_metal() {
                DType::BF16  // Prefer BF16 on GPU
            } else {
                DType::F32   // CPU uses F32
            }
        }
        other => anyhow::bail!("Unsupported dtype: {}, use f16, bf16, f32, or auto", other),
    };

    println!("Using dtype: {:?}", dtype);

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    let config: Config = serde_json::from_slice(&std::fs::read(config_file)?)?;

    println!("DEBUG Config:");
    println!("  vocab_size: {}", config.vocab_size);
    println!("  hidden_size: {}", config.hidden_size);
    println!("  num_attention_heads: {}", config.num_attention_heads);
    println!("  num_key_value_heads: {}", config.num_key_value_heads);
    println!("  max_position_embeddings: {}", config.max_position_embeddings);
    println!("  rope_theta: {}", config.rope_theta);
    println!("  intermediate_size: {}", config.intermediate_size);

    let model = ModelForCausalLM::new(&config, vb)?;

    println!("loaded the model in {:?}", start.elapsed());
    println!("SmolLM3 Config:");

    if let Some(interval) = config.no_rope_layer_interval {
        // Every 4th layer uses NoPE, others use RoPE
        let num_nope_layers = config.num_hidden_layers / interval;
        let num_rope_layers = config.num_hidden_layers - num_nope_layers;
        println!("  - {} layers total", config.num_hidden_layers);
        println!("  - RoPE: {} layers ({}%)", num_rope_layers, num_rope_layers * 100 / config.num_hidden_layers);
        println!("  - NoPE: {} layers ({}%)", num_nope_layers, num_nope_layers * 100 / config.num_hidden_layers);
        println!("  - Pattern: NoPE on every {}th layer (indices 3, 7, 11, ...)", interval);
    } else {
        println!("  - {} layers with standard RoPE", config.num_hidden_layers);
    }

    println!("  - GQA: {} attention heads, {} KV heads",
        config.num_attention_heads,
        config.num_key_value_heads
    );
    println!("  - rope_theta: {}", config.rope_theta);

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        &device,
        config.eos_token_id,
    );

    println!("EOS token ID: {:?}", config.eos_token_id);

    // Format prompt with chat template for 3b (chat-tuned) model
    let formatted_prompt = if matches!(args.model, WhichModel::W3b) && !args.no_chat_template {
        // SmolLM3-3B requires the full chat template with metadata and instructions
        // This matches what tokenizer.apply_chat_template() produces in Python
        let chat_template_str = format!(
            "<|im_start|>system\n\
## Metadata\n\
\n\
Knowledge Cutoff Date: June 2025\n\
Today Date: 05 November 2025\n\
Reasoning Mode: /no_think\n\
\n\
## Custom Instructions\n\
\n\
You are a helpful AI assistant named SmolLM, trained by Hugging Face.\n\
\n\
<|im_start|>user\n\
{}<|im_end|>\n\
<|im_start|>assistant\n\
<think>\n\
\n\
</think>\n",
            args.prompt
        );
        println!("Using full chat template for SmolLM3-3B");
        chat_template_str
    } else {
        // 3b-base or no chat template: use raw prompt
        println!("Using raw prompt (no chat template)");
        args.prompt.clone()
    };

    pipeline.run(&formatted_prompt, args.sample_len)?;
    Ok(())
}