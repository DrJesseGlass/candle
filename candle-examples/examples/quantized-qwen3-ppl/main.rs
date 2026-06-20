//! Perplexity eval for quantized Qwen3 on CPU.
//!
//! Runs the model token-by-token through the actual decode path (the same path we
//! ship), accumulating the negative log-likelihood of each true next token, and
//! reports perplexity = exp(mean NLL). Respects the CANDLE_REQUANT_* env knobs, so
//! `CANDLE_REQUANT_WEIGHTS=q4k` / `CANDLE_REQUANT_LMHEAD=q4k` measure the quality
//! cost of requantization directly.
use anyhow::Result;
use candle::quantized::gguf_file;
use candle::{DType, Device, Tensor};
use candle_transformers::models::quantized_qwen3::ModelWeights as Qwen3;
use clap::Parser;
use tokenizers::Tokenizer;

#[derive(Parser)]
struct Args {
    /// Path to the GGUF model.
    #[arg(long)]
    model: String,
    /// tokenizer.json path; if omitted, downloaded from HF (Qwen/Qwen3-0.6B).
    #[arg(long)]
    tokenizer: Option<String>,
    /// UTF-8 text corpus to evaluate perplexity on.
    #[arg(long)]
    text: String,
    /// Cap the number of tokens evaluated.
    #[arg(long, default_value_t = 2048)]
    max_tokens: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::Cpu;

    let mut file = std::fs::File::open(&args.model)?;
    let content = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(&args.model))?;
    let mut model = Qwen3::from_gguf(content, &mut file, &device)?;

    let tokenizer = match &args.tokenizer {
        Some(p) => Tokenizer::from_file(p).map_err(anyhow::Error::msg)?,
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let path = api
                .model("Qwen/Qwen3-0.6B".to_string())
                .get("tokenizer.json")?;
            Tokenizer::from_file(path).map_err(anyhow::Error::msg)?
        }
    };

    let text = std::fs::read_to_string(&args.text)?;
    let tokens = tokenizer
        .encode(text, false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    let n = tokens.len().min(args.max_tokens);
    anyhow::ensure!(n >= 2, "need at least 2 tokens, got {n}");
    println!("evaluating perplexity over {} tokens", n - 1);

    model.clear_kv_cache();
    let mut nll = 0f64;
    let count = n - 1;
    let t0 = std::time::Instant::now();
    for i in 0..count {
        let input = Tensor::new(&[tokens[i]], &device)?.unsqueeze(0)?; // [1, 1]
        let logits = model.forward(&input, i)?.squeeze(0)?.to_dtype(DType::F32)?; // [vocab]
        let logp = candle_nn::ops::log_softmax(&logits, 0)?;
        let tgt = tokens[i + 1] as usize;
        let lp: f32 = logp.get(tgt)?.to_scalar()?;
        nll += -(lp as f64);
        let done = i + 1;
        if done.is_multiple_of(256) {
            print!(
                "\r  {done}/{count}  running PPL={:.3}",
                (nll / done as f64).exp()
            );
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
    }
    let mean_nll = nll / count as f64;
    println!(
        "\ntokens={count}  mean_NLL={mean_nll:.4}  PPL={:.4}  ({:.1}s)",
        mean_nll.exp(),
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}
