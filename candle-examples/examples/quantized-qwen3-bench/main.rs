//! Throughput benchmark for the quantized Qwen3 model, matched to `llama-bench`
//! methodology so candle and llama.cpp numbers are comparable:
//!   - prefill (pp): one forward over a dummy prompt of `--pp` tokens.
//!   - decode  (tg): `--tg` greedy single-token forwards.
//! No tokenizer, no sampling, no repeat penalty — raw model throughput only.
//! Reports median tokens/s over `--reps` measured runs (after `--warmup`).
//!
//! Pair with thread pinning to simulate a Lambda tier, e.g.:
//!   CANDLE_QMATMUL_DECODE_THREADS=2 CANDLE_QMATMUL_PREFILL_THREADS=2 \
//!     taskset -c 0-1 cargo run --release --example quantized-qwen3-bench -- \
//!     --model model.gguf --json

use anyhow::Result;
use candle::quantized::gguf_file;
use candle::{Device, Tensor};
use candle_transformers::models::quantized_qwen3::ModelWeights as Qwen3;
use clap::Parser;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// GGUF file to load.
    #[arg(long)]
    model: String,

    /// Prompt length in tokens for the prefill (pp) measurement.
    #[arg(long, default_value_t = 512)]
    pp: usize,

    /// Number of tokens to generate for the decode (tg) measurement.
    #[arg(long, default_value_t = 128)]
    tg: usize,

    /// Measured repetitions (median is reported).
    #[arg(long, default_value_t = 5)]
    reps: usize,

    /// Warmup repetitions discarded before measuring.
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Dummy token id used to fill the synthetic prompt (must be < vocab size).
    #[arg(long, default_value_t = 100)]
    token_id: u32,

    /// Emit a single JSON line on stdout instead of a human table.
    #[arg(long)]
    json: bool,
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best as u32
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        xs[n / 2]
    } else {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::Cpu;

    let mut file = std::fs::File::open(&args.model)?;
    let content =
        gguf_file::Content::read(&mut file).map_err(|e| e.with_path(&args.model))?;
    let mut model = Qwen3::from_gguf(content, &mut file, &device)?;

    let prompt: Vec<u32> = vec![args.token_id; args.pp];

    let mut pp_rates = Vec::new();
    let mut tg_rates = Vec::new();

    // Optional matmul-vs-rest profiling of the prefill (CANDLE_MATMUL_PROFILE=1).
    use candle::quantized::k_quants::{
        matmul_profile_reset, MATMUL_CALLS, MATMUL_DOT_NS, MATMUL_QUANT_NS,
    };
    use std::sync::atomic::Ordering::Relaxed;
    let profile = std::env::var("CANDLE_MATMUL_PROFILE").is_ok();
    let mut pp_wall_ns: u128 = 0;
    let mut pp_dot_ns: u64 = 0;
    let mut pp_quant_ns: u64 = 0;
    let mut pp_calls: u64 = 0;
    let mut pp_flash_ns: u64 = 0;
    let mut pp_norm_ns: u64 = 0;
    let mut pp_rope_ns: u64 = 0;
    let mut pp_copy_ns: u64 = 0;
    let mut tg_wall_ns: u128 = 0;
    let mut tg_dot_ns: u64 = 0;
    let mut tg_quant_ns: u64 = 0;
    let mut tg_flash_ns: u64 = 0;
    let mut tg_norm_ns: u64 = 0;
    let mut tg_rope_ns: u64 = 0;
    let mut tg_copy_ns: u64 = 0;

    let total = args.warmup + args.reps;
    for rep in 0..total {
        model.clear_kv_cache();

        // Prefill: one forward over the whole synthetic prompt.
        let input = Tensor::new(prompt.as_slice(), &device)?.unsqueeze(0)?;
        if profile {
            matmul_profile_reset();
            candle_transformers::models::quantized_qwen3::model_profile_reset();
        }
        let t = Instant::now();
        let logits = model.forward(&input, 0)?.squeeze(0)?;
        let pp_elapsed = t.elapsed();
        let pp_dt = pp_elapsed.as_secs_f64();
        if profile && rep >= args.warmup {
            use candle_transformers::models::quantized_qwen3 as m;
            pp_wall_ns += pp_elapsed.as_nanos();
            pp_dot_ns += MATMUL_DOT_NS.load(Relaxed);
            pp_quant_ns += MATMUL_QUANT_NS.load(Relaxed);
            pp_calls += MATMUL_CALLS.load(Relaxed);
            pp_flash_ns += m::FLASH_NS.load(Relaxed);
            pp_norm_ns += m::NORM_NS.load(Relaxed);
            pp_rope_ns += m::ROPE_NS.load(Relaxed);
            pp_copy_ns += m::COPY_NS.load(Relaxed);
        }
        let mut next = argmax(&logits.to_vec1::<f32>()?);

        // Decode: tg greedy single-token forwards.
        if profile {
            matmul_profile_reset();
            candle_transformers::models::quantized_qwen3::model_profile_reset();
        }
        let t = Instant::now();
        for i in 0..args.tg {
            let input = Tensor::new(&[next], &device)?.unsqueeze(0)?;
            let logits = model.forward(&input, args.pp + i)?.squeeze(0)?;
            next = argmax(&logits.to_vec1::<f32>()?);
        }
        let tg_elapsed = t.elapsed();
        let tg_dt = tg_elapsed.as_secs_f64();
        if profile && rep >= args.warmup {
            use candle_transformers::models::quantized_qwen3 as m;
            tg_wall_ns += tg_elapsed.as_nanos();
            tg_dot_ns += MATMUL_DOT_NS.load(Relaxed);
            tg_quant_ns += MATMUL_QUANT_NS.load(Relaxed);
            tg_flash_ns += m::FLASH_NS.load(Relaxed);
            tg_norm_ns += m::NORM_NS.load(Relaxed);
            tg_rope_ns += m::ROPE_NS.load(Relaxed);
            tg_copy_ns += m::COPY_NS.load(Relaxed);
        }

        let pp_rate = args.pp as f64 / pp_dt;
        let tg_rate = args.tg as f64 / tg_dt;
        if rep >= args.warmup {
            pp_rates.push(pp_rate);
            tg_rates.push(tg_rate);
        }
        if !args.json {
            let tag = if rep < args.warmup { "warmup" } else { "run" };
            eprintln!(
                "{tag} {rep}: pp {pp_rate:.2} t/s   tg {tg_rate:.2} t/s",
            );
        }
    }

    let pp_med = median(&mut pp_rates.clone());
    let tg_med = median(&mut tg_rates.clone());
    let pp_min = pp_rates.iter().cloned().fold(f64::INFINITY, f64::min);
    let pp_max = pp_rates.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let tg_min = tg_rates.iter().cloned().fold(f64::INFINITY, f64::min);
    let tg_max = tg_rates.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if args.json {
        println!(
            "{{\"engine\":\"candle\",\"pp\":{},\"tg\":{},\"reps\":{},\
\"pp_tok_s_median\":{:.3},\"pp_tok_s_min\":{:.3},\"pp_tok_s_max\":{:.3},\
\"tg_tok_s_median\":{:.3},\"tg_tok_s_min\":{:.3},\"tg_tok_s_max\":{:.3}}}",
            args.pp, args.tg, args.reps, pp_med, pp_min, pp_max, tg_med, tg_min, tg_max
        );
    } else {
        println!("\ncandle  pp{}  tg{}  reps={}", args.pp, args.tg, args.reps);
        println!("  prefill (pp): {pp_med:.2} t/s  [{pp_min:.2}..{pp_max:.2}]");
        println!("  decode  (tg): {tg_med:.2} t/s  [{tg_min:.2}..{tg_max:.2}]");
    }
    if profile && pp_wall_ns > 0 {
        // MATMUL_DOT_NS/QUANT_NS sum the wall time of each matmul's dot / activation-
        // quantization phase. The remainder of prefill wall time is everything else
        // (attention, norms, rope, softmax, copies, framework overhead).
        let wall = pp_wall_ns as f64;
        let dot = pp_dot_ns as f64;
        let quant = pp_quant_ns as f64;
        let rest = (wall - dot - quant).max(0.0);
        let flash = pp_flash_ns as f64;
        let norm = pp_norm_ns as f64;
        let rope = pp_rope_ns as f64;
        let copy = pp_copy_ns as f64;
        let glue = (rest - flash - norm - rope - copy).max(0.0);
        let pct = |x: f64| 100.0 * x / wall;
        eprintln!(
            "\n[prefill profile over {} reps]  wall={:.1}ms  calls={}\n  \
matmul_dot={:.1}ms ({:.1}%)  matmul_quant={:.1}ms ({:.1}%)\n  \
REST={:.1}ms ({:.1}%) = flash_attn {:.1}ms ({:.1}%) + norm {:.1}ms ({:.1}%) + \
rope {:.1}ms ({:.1}%) + copy {:.1}ms ({:.1}%) + glue {:.1}ms ({:.1}%)",
            args.reps, wall / 1e6, pp_calls,
            dot / 1e6, pct(dot), quant / 1e6, pct(quant),
            rest / 1e6, pct(rest),
            flash / 1e6, pct(flash), norm / 1e6, pct(norm),
            rope / 1e6, pct(rope), copy / 1e6, pct(copy), glue / 1e6, pct(glue),
        );
    }
    if profile && tg_wall_ns > 0 {
        let wall = tg_wall_ns as f64;
        let dot = tg_dot_ns as f64;
        let quant = tg_quant_ns as f64;
        let flash = tg_flash_ns as f64;
        let norm = tg_norm_ns as f64;
        let rope = tg_rope_ns as f64;
        let copy = tg_copy_ns as f64;
        let rest = (wall - dot - quant).max(0.0);
        let glue = (rest - flash - norm - rope - copy).max(0.0);
        let pct = |x: f64| 100.0 * x / wall;
        eprintln!(
            "\n[decode profile over {} reps × {} tok]  wall={:.1}ms\n  \
matmul_dot={:.1}ms ({:.1}%)  matmul_quant={:.1}ms ({:.1}%)\n  \
REST={:.1}ms ({:.1}%) = flash {:.1}ms ({:.1}%) + norm {:.1}ms ({:.1}%) + \
rope {:.1}ms ({:.1}%) + copy {:.1}ms ({:.1}%) + glue {:.1}ms ({:.1}%)",
            args.reps, args.tg, wall / 1e6,
            dot / 1e6, pct(dot), quant / 1e6, pct(quant),
            rest / 1e6, pct(rest),
            flash / 1e6, pct(flash), norm / 1e6, pct(norm),
            rope / 1e6, pct(rope), copy / 1e6, pct(copy), glue / 1e6, pct(glue),
        );
    }
    Ok(())
}
