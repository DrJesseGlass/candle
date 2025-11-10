// Complete standalone test for Q/K interleaving hypothesis
// Run with: cargo run --release --example test_interleave -- --regular /path/to/model.safetensors --quantized /path/to/model.gguf

use anyhow::Result;
use candle::{DType, Device, Tensor};
use candle::quantized::gguf_file;
use clap::Parser;
use std::fs::File;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    regular: String,
    #[arg(long)]
    quantized: String,
}

fn deinterleave_rows(weight: &Tensor) -> Result<Tensor> {
    let (out_features, in_features) = weight.dims2()?;
    let mut rows = Vec::new();
    for i in (0..out_features).step_by(2) {
        rows.push(weight.i(i)?);
    }
    Tensor::stack(&rows, 0)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::Cpu;

    println!("Loading GGUF...");
    let mut file = File::open(&args.quantized)?;
    let gguf = gguf_file::Content::read(&mut file)?;

    println!("Loading safetensors...");
    let regular = candle::safetensors::load(&args.regular, &device)?;

    // Load Q projection weights
    let q_gguf = gguf.tensor(&mut file, "blk.0.attn_q.weight", &device)?;
    let q_gguf_dq = q_gguf.dequantize(&device)?;
    let q_regular = regular.get("model.layers.0.self_attn.q_proj.weight")
        .ok_or_else(|| anyhow::anyhow!("Q weight not found in regular model"))?;

    // Also load K to test interleaving hypothesis
    let k_gguf = gguf.tensor(&mut file, "blk.0.attn_k.weight", &device)?;
    let k_gguf_dq = k_gguf.dequantize(&device)?;
    let k_regular = regular.get("model.layers.0.self_attn.k_proj.weight")
        .ok_or_else(|| anyhow::anyhow!("K weight not found in regular model"))?;

    println!("\n=== Weight Shapes ===");
    println!("Q_GGUF: {:?}", q_gguf_dq.dims());
    println!("Q_regular: {:?}", q_regular.dims());
    println!("K_GGUF: {:?}", k_gguf_dq.dims());
    println!("K_regular: {:?}", k_regular.dims());

    // CRITICAL CHECK: If Q_GGUF is [4096, 2048] instead of [2048, 2048],
    // it contains BOTH Q and K!
    let (q_gguf_out, q_gguf_in) = q_gguf_dq.dims2()?;
    let (q_reg_out, _q_reg_in) = q_regular.dims2()?;

    if q_gguf_out == 2 * q_reg_out {
        println!("\n*** FOUND IT! Q_GGUF has 2x the output features! ***");
        println!("This means Q and K are stored together in the same tensor!");
        println!("Q_GGUF contains both Q ({} dims) and K ({} dims) interleaved", q_reg_out, q_gguf_out - q_reg_out);
    }

    // Create a simple test input
    println!("\n=== Creating test input ===");
    let mut input_vec = vec![0.0f32; 2048];
    input_vec[0] = 1.0;  // One-hot encoding at position 0
    let input = Tensor::from_vec(input_vec, &[1, 1, 2048], &device)?;

    // Run through projections
    println!("\n=== Computing outputs ===");
    let q_out_gguf = input.matmul(&q_gguf_dq.t()?)?;
    let q_out_regular = input.matmul(&q_regular.t()?)?;
    let k_out_regular = input.matmul(&k_regular.t()?)?;

    let q_gguf_vec: Vec<f32> = q_out_gguf.flatten_all()?.to_vec1()?;
    let q_regular_vec: Vec<f32> = q_out_regular.flatten_all()?.to_vec1()?;
    let k_regular_vec: Vec<f32> = k_out_regular.flatten_all()?.to_vec1()?;

    // Test the interleaving hypothesis
    println!("\n=== TESTING INTERLEAVING HYPOTHESIS ===");
    println!("Testing if Q_GGUF[2*i] = Q_regular[i] and Q_GGUF[2*i+1] = K_regular[i]");
    println!();

    let mut q_even_matches = 0;
    let mut k_odd_matches = 0;
    let test_count = 16.min(q_regular_vec.len()).min(k_regular_vec.len());

    for i in 0..test_count {
        // Check if even indices of GGUF match sequential Q_regular
        if 2*i < q_gguf_vec.len() {
            let diff_q = (q_gguf_vec[2*i] - q_regular_vec[i]).abs();
            let matches_q = diff_q < 0.001;
            if matches_q {
                q_even_matches += 1;
            }

            let status_q = if matches_q { "[OK]" } else { "[X]" };
            println!("Q_GGUF[{:2}] = {:9.6} vs Q_regular[{:2}] = {:9.6} | diff = {:.6} {}",
                     2*i, q_gguf_vec[2*i], i, q_regular_vec[i], diff_q, status_q);
        }

        // Check if odd indices of GGUF match sequential K_regular
        if 2*i + 1 < q_gguf_vec.len() {
            let diff_k = (q_gguf_vec[2*i + 1] - k_regular_vec[i]).abs();
            let matches_k = diff_k < 0.001;
            if matches_k {
                k_odd_matches += 1;
            }

            let status_k = if matches_k { "[OK]" } else { "[X]" };
            println!("Q_GGUF[{:2}] = {:9.6} vs K_regular[{:2}] = {:9.6} | diff = {:.6} {}",
                     2*i + 1, q_gguf_vec[2*i + 1], i, k_regular_vec[i], diff_k, status_k);
        }
        println!();
    }

    println!("Results: {}/{} even indices match Q, {}/{} odd indices match K",
             q_even_matches, test_count, k_odd_matches, test_count);

    if q_even_matches >= test_count - 1 && k_odd_matches >= test_count - 1 {
        println!("\n*** CONFIRMED: GGUF has Q and K interleaved! ***");
        println!("Q values are at even indices [0, 2, 4, ...]");
        println!("K values are at odd indices [1, 3, 5, ...]");
    } else if q_even_matches >= test_count - 1 {
        println!("\n*** PARTIAL: Even indices match Q but odd indices don't match K ***");
    } else {
        println!("\nInterleaving hypothesis not confirmed. Different pattern.");
        return Ok(());
    }

    // Apply the fix and verify
    println!("\n=== TESTING FIX ===");
    let q_deinterleaved = deinterleave_rows(&q_gguf_dq)?;
    println!("Deinterleaved Q shape: {:?} (was {:?})", q_deinterleaved.dims(), q_gguf_dq.dims());

    let q_out_fixed = input.matmul(&q_deinterleaved.t()?)?;
    let q_fixed_vec: Vec<f32> = q_out_fixed.flatten_all()?.to_vec1()?;

    println!("\nComparing outputs:");
    println!("Index | Fixed       | Regular     | Difference");
    println!("------|-------------|-------------|-------------");
    for i in 0..12.min(q_fixed_vec.len()).min(q_regular_vec.len()) {
        let diff = (q_fixed_vec[i] - q_regular_vec[i]).abs();
        let status = if diff < 0.01 { "[OK]" } else { "[X]" };
        println!("{:5} | {:11.6} | {:11.6} | {:11.6} {}",
                 i, q_fixed_vec[i], q_regular_vec[i], diff, status);
    }

    // Calculate average difference
    let avg_diff: f32 = q_fixed_vec.iter()
        .zip(q_regular_vec.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / q_fixed_vec.len().min(q_regular_vec.len()) as f32;

    println!("\nAverage difference: {:.8}", avg_diff);

    if avg_diff < 0.01 {
        println!("\n*** SUCCESS! Deinterleaving fixes the issue! ***");
        println!("\nTO FIX YOUR CODE:");
        println!("1. Add the deinterleave_rows function to your code");
        println!("2. Apply it to Q, K, V weights after dequantization:");
        println!("   let q_weight = deinterleave_rows(&q_weight_raw)?;");
    } else {
        println!("\nDeinterleaving didn't fully fix the issue.");
        println!("There may be an additional transformation needed.");
    }

    Ok(())
}