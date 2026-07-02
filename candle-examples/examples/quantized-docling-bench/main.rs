//! Throughput benchmark for the quantized Granite-Docling model, matched to
//! llama.cpp methodology so candle and llama.cpp numbers are comparable:
//!   - prefill (pp): one text forward over a dummy prompt of `--pp` tokens
//!     (comparable to `llama-bench -p`).
//!   - decode  (tg): `--tg` greedy single-token forwards
//!     (comparable to `llama-bench -n`).
//!   - vision: encode `--tiles` synthetic tiles through SigLIP + connector
//!     (comparable to llama-mtmd-cli's "image encoded in N ms"), plus the
//!     end-to-end vision prefill (encode + merge + prefill via `setup`),
//!     which is the shape of the real Docling workload.
//!
//! No tokenizer, no sampling - raw model throughput only. Reports medians
//! over `--reps` measured runs (after `--warmup`).
//!
//! Pair with thread pinning to simulate a Lambda tier, e.g.:
//!   RAYON_NUM_THREADS=2 CANDLE_NUM_THREADS=2 taskset -c 0-1 \
//!     target/release/examples/quantized-docling-bench \
//!     --model granite-docling-258M-Q8_0.gguf \
//!     --mmproj mmproj-granite-docling-258M-Q8_0.gguf --json

use anyhow::Result;
use candle::quantized::gguf_file;
use candle::{Device, IndexOp, Tensor, D};
use candle_transformers::models::granite_docling::{
    config::{QuantizedVisionConfig, TextConfig},
    quantized::Model,
};
use candle_transformers::quantized_var_builder::VarBuilder;
use clap::Parser;
use std::time::Instant;

// llama.cpp <|image|> id for granite-docling; overridable if the GGUF changes.
const IMAGE_TOKEN_ID: u32 = 100270;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Text decoder GGUF file.
    #[arg(long)]
    model: String,

    /// Vision mmproj GGUF file.
    #[arg(long)]
    mmproj: String,

    /// Prompt length in tokens for the text prefill (pp) measurement.
    #[arg(long, default_value_t = 512)]
    pp: usize,

    /// Number of tokens to generate for the decode (tg) measurement.
    #[arg(long, default_value_t = 128)]
    tg: usize,

    /// Number of synthetic image tiles for the vision measurements. A real
    /// Docling page is a grid of tiles plus one global tile.
    #[arg(long, default_value_t = 1)]
    tiles: usize,

    /// Measured repetitions (median is reported).
    #[arg(long, default_value_t = 5)]
    reps: usize,

    /// Warmup repetitions discarded before measuring.
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Dummy token id used to fill the synthetic prompt (must be < vocab size).
    #[arg(long, default_value_t = 100)]
    token_id: u32,

    /// Image placeholder token id (must match the GGUF's <|image|> id).
    #[arg(long, default_value_t = IMAGE_TOKEN_ID)]
    image_token_id: u32,

    /// Skip the vision measurements (text pp/tg only).
    #[arg(long)]
    no_vision: bool,

    /// Emit a single JSON line on stdout instead of a human table.
    #[arg(long)]
    json: bool,
}

/// Greedy next token from the last position of (1, seq, vocab) logits.
fn argmax_last(logits: &Tensor) -> Result<u32> {
    let (_b, seq, _v) = logits.dims3()?;
    Ok(logits
        .i((0, seq - 1))?
        .argmax(D::Minus1)?
        .to_scalar::<u32>()?)
}

/// (median, min, max) of a sample. Returns zeros for an empty slice.
fn stats(xs: &[f64]) -> (f64, f64, f64) {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    let median = if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    };
    (median, s[0], s[n - 1])
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::Cpu;

    // Boot split: gguf reads (header + metadata parse) vs model build
    // (VarBuilder tensor loads + construct + any repack/dequant).
    let t_read = Instant::now();
    let text_cfg = {
        let mut f = std::fs::File::open(&args.model)?;
        let content = gguf_file::Content::read(&mut f).map_err(|e| e.with_path(&args.model))?;
        TextConfig::from_gguf(&content.metadata)?
    };
    let vision_cfg = {
        let mut f = std::fs::File::open(&args.mmproj)?;
        let content = gguf_file::Content::read(&mut f).map_err(|e| e.with_path(&args.mmproj))?;
        QuantizedVisionConfig::from_gguf(&content.metadata)?
    };
    let gguf_read_ms = t_read.elapsed().as_millis();

    let t_build = Instant::now();
    let text_vb = VarBuilder::from_gguf(&args.model, &device)?;
    let vision_vb = VarBuilder::from_gguf(&args.mmproj, &device)?;
    let mut model = Model::new(
        vision_vb,
        &vision_cfg,
        text_vb,
        &text_cfg,
        args.image_token_id,
    )?;
    let model_build_ms = t_build.elapsed().as_millis();

    let tile_size = vision_cfg.image_size;
    let image_seq_len = vision_cfg.image_seq_len();
    if !args.json {
        eprintln!("COLDSTART gguf_read_ms={gguf_read_ms} model_build_ms={model_build_ms}");
        eprintln!(
            "config: vision {tile_size}x{tile_size} -> {image_seq_len} tok/tile, \
             text hidden={} layers={}",
            text_cfg.hidden_size, text_cfg.num_hidden_layers
        );
    }

    // Vision legs: synthetic tiles (constant gray) through the real pipeline.
    let mut vis_ms = Vec::new();
    let mut vp_rates = Vec::new();
    let vp_tokens = args.tiles * image_seq_len + 16;
    if !args.no_vision {
        let pixel_values = Tensor::full(0.5f32, (args.tiles, 3, tile_size, tile_size), &device)?;
        // <image tokens> then a short instruction-sized tail of dummy tokens.
        let mut ids = vec![args.image_token_id; args.tiles * image_seq_len];
        ids.extend(std::iter::repeat_n(args.token_id, 16));
        let input_ids = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;

        for rep in 0..args.warmup + args.reps {
            // Vision encoder + connector alone (llama-mtmd "image encoded in").
            let t = Instant::now();
            let feats = model.encode_image(&pixel_values)?;
            let enc_ms = t.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(&feats);

            // End-to-end vision prefill: encode + merge + text prefill.
            let t = Instant::now();
            let logits = model.setup(&pixel_values, &input_ids)?;
            let vp_dt = t.elapsed().as_secs_f64();
            std::hint::black_box(&logits);
            model.clear_kv_cache();

            let vp_rate = vp_tokens as f64 / vp_dt;
            if rep >= args.warmup {
                vis_ms.push(enc_ms);
                vp_rates.push(vp_rate);
            }
            if !args.json {
                let tag = if rep < args.warmup { "warmup" } else { "run" };
                eprintln!(
                    "{tag} {rep}: vision_encode {enc_ms:.1} ms   \
                     vision_prefill {vp_rate:.2} t/s ({vp_tokens} tok)"
                );
            }
        }
    }

    // Text legs, identical methodology to quantized-qwen3-bench / llama-bench.
    let prompt: Vec<u32> = vec![args.token_id; args.pp];
    let mut pp_rates = Vec::new();
    let mut tg_rates = Vec::new();
    for rep in 0..args.warmup + args.reps {
        model.clear_kv_cache();

        let input = Tensor::new(prompt.as_slice(), &device)?.unsqueeze(0)?;
        let t = Instant::now();
        let logits = model.forward(&input)?;
        let pp_dt = t.elapsed().as_secs_f64();
        let mut next = argmax_last(&logits)?;

        let t = Instant::now();
        for _ in 0..args.tg {
            let input = Tensor::new(&[next], &device)?.unsqueeze(0)?;
            let logits = model.forward(&input)?;
            next = argmax_last(&logits)?;
        }
        let tg_dt = t.elapsed().as_secs_f64();

        let pp_rate = args.pp as f64 / pp_dt;
        let tg_rate = args.tg as f64 / tg_dt;
        if rep >= args.warmup {
            pp_rates.push(pp_rate);
            tg_rates.push(tg_rate);
        }
        if !args.json {
            let tag = if rep < args.warmup { "warmup" } else { "run" };
            eprintln!("{tag} {rep}: pp {pp_rate:.2} t/s   tg {tg_rate:.2} t/s");
        }
    }

    let (vis_med, vis_min, vis_max) = stats(&vis_ms);
    let (vp_med, vp_min, vp_max) = stats(&vp_rates);
    let (pp_med, pp_min, pp_max) = stats(&pp_rates);
    let (tg_med, tg_min, tg_max) = stats(&tg_rates);

    if args.json {
        println!(
            "{{\"engine\":\"candle\",\"pp\":{},\"tg\":{},\"tiles\":{},\"reps\":{},\
\"image_seq_len\":{},\"vision_prefill_tokens\":{},\
\"gguf_read_ms\":{},\"model_build_ms\":{},\
\"vision_encode_ms_median\":{:.1},\"vision_encode_ms_min\":{:.1},\"vision_encode_ms_max\":{:.1},\
\"vision_prefill_tok_s_median\":{:.3},\"vision_prefill_tok_s_min\":{:.3},\"vision_prefill_tok_s_max\":{:.3},\
\"pp_tok_s_median\":{:.3},\"pp_tok_s_min\":{:.3},\"pp_tok_s_max\":{:.3},\
\"tg_tok_s_median\":{:.3},\"tg_tok_s_min\":{:.3},\"tg_tok_s_max\":{:.3}}}",
            args.pp,
            args.tg,
            args.tiles,
            args.reps,
            image_seq_len,
            vp_tokens,
            gguf_read_ms,
            model_build_ms,
            vis_med,
            vis_min,
            vis_max,
            vp_med,
            vp_min,
            vp_max,
            pp_med,
            pp_min,
            pp_max,
            tg_med,
            tg_min,
            tg_max
        );
    } else {
        println!(
            "\ncandle docling  pp{}  tg{}  tiles={}  reps={}",
            args.pp, args.tg, args.tiles, args.reps
        );
        if !args.no_vision {
            println!("  vision encode : {vis_med:.1} ms  [{vis_min:.1}..{vis_max:.1}]");
            println!(
                "  vision prefill: {vp_med:.2} t/s  [{vp_min:.2}..{vp_max:.2}]  ({vp_tokens} tok)"
            );
        }
        println!("  prefill (pp)  : {pp_med:.2} t/s  [{pp_min:.2}..{pp_max:.2}]");
        println!("  decode  (tg)  : {tg_med:.2} t/s  [{tg_min:.2}..{tg_max:.2}]");
    }
    Ok(())
}
