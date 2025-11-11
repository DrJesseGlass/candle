// Comprehensive diagnostic - find EXACTLY where and why outputs diverge
// Place in: candle-examples/examples/compare_smollm3_weights.rs

use anyhow::Result;
use candle::{DType, Device, Tensor};
use candle::quantized::gguf_file;
use clap::Parser;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::File;
use std::path::PathBuf;
use candle::IndexOp;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    regular: String,
    #[arg(long)]
    quantized: String,
}

fn detailed_comparison(name: &str, t1: &Tensor, t2: &Tensor) -> Result<()> {
    let t1 = t1.to_dtype(DType::F32)?;
    let t2 = t2.to_dtype(DType::F32)?;

    if t1.dims() != t2.dims() {
        println!("❌ {} - SHAPE MISMATCH: {:?} vs {:?}", name, t1.dims(), t2.dims());
        return Ok(());
    }

    let diff = (&t1 - &t2)?.abs()?;
    let max_diff: f32 = diff.flatten_all()?.max(0)?.to_scalar()?;
    let min_diff: f32 = diff.flatten_all()?.min(0)?.to_scalar()?;
    let avg_diff: f32 = diff.flatten_all()?.mean(0)?.to_scalar()?;

    let t1_flat: Vec<f32> = t1.flatten_all()?.to_vec1()?;
    let t2_flat: Vec<f32> = t2.flatten_all()?.to_vec1()?;

    let t1_mean: f32 = t1_flat.iter().sum::<f32>() / t1_flat.len() as f32;
    let t2_mean: f32 = t2_flat.iter().sum::<f32>() / t2_flat.len() as f32;

    println!("\n=== {} ===", name);
    println!("Shape: {:?}", t1.dims());
    println!("Regular mean: {:.6}, Quantized mean: {:.6}", t1_mean, t2_mean);
    println!("Difference: min={:.6} avg={:.6} max={:.6}", min_diff, avg_diff, max_diff);

    Ok(())
}

// ===== RECONSTRUCTION FUNCTION - CHANGE THIS TO TEST DIFFERENT STRATEGIES =====
fn reconstruct_qk_weights(gguf_weight: &Tensor, num_heads: usize) -> Result<Tensor> {
    // Strategy: 128-row chunks on first half, then second half
    //
    // For Q (16 heads, 2048 rows):
    //   - First half (rows 0-1023): 8 chunks × 2 (even/odd) = 16 heads total... wait no
    //   - Actually: 1024/128 = 8 chunks, each produces 2 heads = 16 heads in first half? No...
    //   - First half: 1024 rows → 8 heads
    //   - Second half: 1024 rows → 8 heads
    //   - Total: 16 heads
    //
    // For K (4 heads, 512 rows):
    //   - First half (rows 0-255): 2 chunks × 2 (even/odd) = 4 heads in first half? No...
    //   - First half: 256 rows → 2 heads
    //   - Second half: 256 rows → 2 heads
    //   - Total: 4 heads

    let total_rows = gguf_weight.dim(0)?;
    let half_rows = total_rows / 2;
    let chunk_size = 128;
    let chunks_per_half = half_rows / chunk_size;

    let mut heads = Vec::new();

    // First half
    for chunk_idx in 0..chunks_per_half {
        let chunk_start = chunk_idx * chunk_size;

        // Even rows
        let mut head_even = Vec::new();
        for i in (chunk_start..chunk_start + chunk_size).step_by(2) {
            head_even.push(gguf_weight.i(i)?);
        }
        heads.push(Tensor::stack(&head_even, 0)?);

        // Odd rows
        let mut head_odd = Vec::new();
        for i in (chunk_start + 1..chunk_start + chunk_size).step_by(2) {
            head_odd.push(gguf_weight.i(i)?);
        }
        heads.push(Tensor::stack(&head_odd, 0)?);
    }

    // Second half
    for chunk_idx in 0..chunks_per_half {
        let chunk_start = half_rows + chunk_idx * chunk_size;

        // Even rows
        let mut head_even = Vec::new();
        for i in (chunk_start..chunk_start + chunk_size).step_by(2) {
            head_even.push(gguf_weight.i(i)?);
        }
        heads.push(Tensor::stack(&head_even, 0)?);

        // Odd rows
        let mut head_odd = Vec::new();
        for i in (chunk_start + 1..chunk_start + chunk_size).step_by(2) {
            head_odd.push(gguf_weight.i(i)?);
        }
        heads.push(Tensor::stack(&head_odd, 0)?);
    }

    Ok(Tensor::cat(&heads, 0)?)
}

fn load_safetensors(paths: &[PathBuf], device: &Device) -> Result<HashMap<String, Tensor>> {
    use candle::safetensors::load;
    let mut all_tensors = HashMap::new();
    for path in paths {
        let tensors = load(path, device)?;
        all_tensors.extend(tensors);
    }
    Ok(all_tensors)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::Cpu;

    println!("Loading models...\n");

    let safetensors_files = if std::fs::metadata(&args.regular)?.is_dir() {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&args.regular)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext == "safetensors")
                    .unwrap_or(false)
            })
            .collect();
        files.sort();
        files
    } else {
        vec![PathBuf::from(&args.regular)]
    };

    let regular_tensors = load_safetensors(&safetensors_files, &device)?;
    println!("✅ Loaded {} tensors from regular model", regular_tensors.len());

    let mut file = File::open(&args.quantized)?;
    let gguf = gguf_file::Content::read(&mut file)?;
    println!("✅ Loaded GGUF file with {} tensors\n", gguf.tensor_infos.len());

    // Load GGUF tensors ONCE at the beginning and cache them
    println!("=== LOADING GGUF TENSORS ===");
    let mut gguf_tensors = HashMap::new();

    let tensor_names = [
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
    ];

    for name in tensor_names {
        if let Ok(tensor) = gguf.tensor(&mut file, name, &device) {
            let dequantized = tensor.dequantize(&device)?;
            println!("  ✓ Cached {}: {:?}", name, dequantized.dims());
            gguf_tensors.insert(name.to_string(), dequantized);
        }
    }

    println!("\n{}", "=".repeat(80));
    println!("TESTING Q RECONSTRUCTION");
    println!("{}", "=".repeat(80));

    if let Some(reg_q) = regular_tensors.get("model.layers.0.self_attn.q_proj.weight") {
        if let Some(quant_q_dq) = gguf_tensors.get("blk.0.attn_q.weight") {
            let reg_q = reg_q.to_dtype(DType::F32)?;
            let quant_q_dq = quant_q_dq.to_dtype(DType::F32)?;

            println!("\nOriginal GGUF Q shape: {:?}", quant_q_dq.dims());
            println!("Expected (Regular) shape: {:?}", reg_q.dims());

            // Test reconstruction
            let reconstructed_q = reconstruct_qk_weights(&quant_q_dq, 16)?;
            println!("Reconstructed Q shape: {:?}", reconstructed_q.dims());

            // Compare
            detailed_comparison("Q_PROJ Reconstruction", &reg_q, &reconstructed_q)?;

            // Detailed row checks
            println!("\n=== Row-by-row verification (sample) ===");
            let head_dim = 128;
            let test_rows = vec![
                0, 1, 2,              // Start of head 0
                126, 127, 128, 129,   // Boundary head 0/1
                254, 255, 256, 257,   // Boundary head 1/2
                1022, 1023, 1024, 1025, // Boundary head 7/8
                2046, 2047,           // End
            ];

            let mut perfect = 0;
            let mut failures = 0;

            for &i in &test_rows {
                let reg_row: Vec<f32> = reg_q.i(i)?.to_vec1()?;
                let recon_row: Vec<f32> = reconstructed_q.i(i)?.to_vec1()?;

                let diff: f32 = reg_row.iter().zip(recon_row.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f32>() / reg_row.len() as f32;

                let status = if diff < 0.001 {
                    perfect += 1;
                    "✓"
                } else {
                    failures += 1;
                    "✗"
                };

                let head_num = i / head_dim;
                println!("Row {:4} (Head {:2}, offset {:3}): diff = {:.8} {}",
                         i, head_num, i % head_dim, diff, status);
            }

            println!("\n=== SUMMARY ===");
            println!("Perfect matches: {}/{}", perfect, test_rows.len());
            println!("Failures: {}/{}", failures, test_rows.len());
        }
    }

    println!("\n{}", "=".repeat(80));
    println!("TESTING K RECONSTRUCTION");
    println!("{}", "=".repeat(80));

    if let Some(reg_k) = regular_tensors.get("model.layers.0.self_attn.k_proj.weight") {
        if let Some(quant_k_dq) = gguf_tensors.get("blk.0.attn_k.weight") {
            let reg_k = reg_k.to_dtype(DType::F32)?;
            let quant_k_dq = quant_k_dq.to_dtype(DType::F32)?;

            println!("\nOriginal GGUF K shape: {:?}", quant_k_dq.dims());
            println!("Expected (Regular) shape: {:?}", reg_k.dims());

            // For K, we have 4 heads instead of 16
            let reconstructed_k = reconstruct_qk_weights(&quant_k_dq, 4)?;
            println!("Reconstructed K shape: {:?}", reconstructed_k.dims());

            // Compare
            detailed_comparison("K_PROJ Reconstruction", &reg_k, &reconstructed_k)?;

            // Quick check
            let test_rows = vec![0, 127, 128, 255, 256, 383, 384, 511];
            let head_dim = 128;

            println!("\n=== Row-by-row verification (sample) ===");
            for &i in &test_rows {
                let reg_row: Vec<f32> = reg_k.i(i)?.to_vec1()?;
                let recon_row: Vec<f32> = reconstructed_k.i(i)?.to_vec1()?;

                let diff: f32 = reg_row.iter().zip(recon_row.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f32>() / reg_row.len() as f32;

                let status = if diff < 0.001 { "✓" } else { "✗" };
                let head_num = i / head_dim;
                println!("Row {:4} (Head {:2}, offset {:3}): diff = {:.8} {}",
                         i, head_num, i % head_dim, diff, status);
            }
        }
    }

    Ok(())
}