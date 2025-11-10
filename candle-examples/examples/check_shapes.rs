// Quick diagnostic for SmolLM3 GGUF weight layout
// Run: cargo run --release --example quick_test -- --model-dir ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM3-3B/snapshots/*/ --gguf ~/.cache/huggingface/hub/models--ggml-org--SmolLM3-3B-GGUF/snapshots/*/SmolLM3-f16.gguf

use anyhow::Result;
use candle::{Device, IndexOp, Tensor};
use candle::quantized::gguf_file;
use clap::Parser;
use std::fs::File;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model_dir: String,
    #[arg(long)]
    gguf: String,
}

fn find_safetensors_file(dir: &str) -> Result<PathBuf> {
    let path = PathBuf::from(dir);

    if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
        return Ok(path);
    }

    if !path.is_dir() {
        anyhow::bail!("Path is not a directory: {:?}", path);
    }

    // Find any .safetensors file in the directory
    for entry in std::fs::read_dir(&path)? {
        let entry = entry?;
        let entry_path = entry.path();
        if entry_path.is_file() {
            if let Some(ext) = entry_path.extension() {
                if ext == "safetensors" {
                    println!("Found safetensors: {:?}", entry_path);
                    return Ok(entry_path);
                }
            }
        }
    }

    anyhow::bail!("No .safetensors file found in {:?}", path);
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::Cpu;

    println!("=== SmolLM3 GGUF Weight Layout Diagnostic ===\n");

    // Load safetensors
    let safetensors_path = find_safetensors_file(&args.model_dir)?;
    println!("Loading safetensors: {:?}", safetensors_path);
    let regular = candle::safetensors::load(&safetensors_path, &device)?;

    // Load GGUF
    println!("Loading GGUF: {}", args.gguf);
    let mut file = File::open(&args.gguf)?;
    let gguf = gguf_file::Content::read(&mut file)?;

    println!("\n=== STEP 1: Check Shapes ===\n");

    // Get Q weights
    let q_gguf = gguf.tensor(&mut file, "blk.0.attn_q.weight", &device)?;
    let q_gguf_dq = q_gguf.dequantize(&device)?;
    let q_regular = regular.get("model.layers.0.self_attn.q_proj.weight")
        .ok_or_else(|| anyhow::anyhow!("Q weight not found"))?;

    let k_regular = regular.get("model.layers.0.self_attn.k_proj.weight")
        .ok_or_else(|| anyhow::anyhow!("K weight not found"))?;

    println!("Q_GGUF shape:    {:?}", q_gguf_dq.dims());
    println!("Q_regular shape: {:?}", q_regular.dims());
    println!("K_regular shape: {:?}", k_regular.dims());

    let (q_gguf_out, q_gguf_in) = q_gguf_dq.dims2()?;
    let (q_reg_out, _) = q_regular.dims2()?;
    let (k_reg_out, _) = k_regular.dims2()?;

    println!("\nExpected: Q=[{}, {}], K=[{}, {}]", q_reg_out, q_gguf_in, k_reg_out, q_gguf_in);
    println!("Got GGUF: Q=[{}, {}]", q_gguf_out, q_gguf_in);

    if q_gguf_out != q_reg_out {
        println!("\n*** SHAPE MISMATCH DETECTED! ***");
        if q_gguf_out == q_reg_out + k_reg_out {
            println!("GGUF contains Q+K concatenated: {} + {} = {}", q_reg_out, k_reg_out, q_gguf_out);
        } else if q_gguf_out == 2 * q_reg_out {
            println!("GGUF contains Q and K interleaved with padding: 2 * {} = {}", q_reg_out, q_gguf_out);
        } else {
            println!("Unknown layout pattern");
        }
    } else {
        println!("\nShapes match! Testing if values are interleaved within...");
    }

    println!("\n=== STEP 2: Test Output Pattern ===\n");

    // Create simple test input
    let mut input_vec = vec![0.0f32; q_gguf_in];
    input_vec[0] = 1.0;
    let input = Tensor::from_vec(input_vec, &[1, 1, q_gguf_in], &device)?;

    // Compute outputs
    let q_out_gguf = input.matmul(&q_gguf_dq.t()?)?;
    let q_out_regular = input.matmul(&q_regular.t()?)?;
    let k_out_regular = input.matmul(&k_regular.t()?)?;

    let q_gguf_vec: Vec<f32> = q_out_gguf.flatten_all()?.to_vec1()?;
    let q_regular_vec: Vec<f32> = q_out_regular.flatten_all()?.to_vec1()?;
    let k_regular_vec: Vec<f32> = k_out_regular.flatten_all()?.to_vec1()?;

    println!("First 8 outputs:");
    println!("GGUF:    {:?}", &q_gguf_vec[0..8]);
    println!("Regular: {:?}", &q_regular_vec[0..8]);

    println!("\n=== STEP 3: Check Interleaving Pattern ===\n");

    let mut matches_even = 0;
    let mut matches_sequential = 0;

    for i in 0..8.min(q_regular_vec.len()) {
        // Check if sequential matches
        if i < q_gguf_vec.len() {
            let diff_seq = (q_gguf_vec[i] - q_regular_vec[i]).abs();
            if diff_seq < 0.001 {
                matches_sequential += 1;
                print!(".");
            } else {
                print!("X");
            }
        }

        // Check if even indices match sequential
        if 2*i < q_gguf_vec.len() {
            let diff_even = (q_gguf_vec[2*i] - q_regular_vec[i]).abs();
            if diff_even < 0.001 {
                matches_even += 1;
            }
        }
    }
    println!(" (sequential match test)");

    println!("\nResults:");
    println!("  Sequential matches: {}/8", matches_sequential);
    println!("  Even-index matches: {}/8", matches_even);

    if matches_sequential >= 7 {
        println!("\n*** LAYOUT IS CORRECT! ***");
        println!("The issue might be elsewhere (RoPE, attention, etc.)");
    } else if matches_even >= 7 {
        println!("\n*** INTERLEAVING DETECTED! ***");
        println!("Q values are at even indices: 0, 2, 4, 6, ...");
        println!("\nFIX: Extract every other row from q_weight:");
        println!("  let mut q_rows = Vec::new();");
        println!("  for i in (0..out_features).step_by(2) {{");
        println!("      q_rows.push(weight.i(i)?);");
        println!("  }}");
        println!("  let q_weight = Tensor::stack(&q_rows, 0)?;");
    } else {
        println!("\n*** UNKNOWN PATTERN ***");
        println!("Values don't match expected patterns.");
        println!("Checking if odd indices match K...");

        let mut k_matches = 0;
        for i in 0..8.min(k_regular_vec.len()) {
            if 2*i + 1 < q_gguf_vec.len() {
                let diff = (q_gguf_vec[2*i + 1] - k_regular_vec[i]).abs();
                if diff < 0.001 {
                    k_matches += 1;
                }
            }
        }
        println!("Odd indices match K: {}/8", k_matches);
    }

    Ok(())
}