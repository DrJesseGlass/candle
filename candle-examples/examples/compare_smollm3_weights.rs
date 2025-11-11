// Comprehensive diagnostic - find EXACTLY where and why outputs diverge
// Place in: candle-examples/examples/compare_smollm3_weights.rs

use anyhow::Result;
use candle::{DType, Device, Tensor};
use candle::quantized::gguf_file;
use clap::Parser;
use std::collections::HashMap;
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

    // Get actual values
    let t1_flat: Vec<f32> = t1.flatten_all()?.to_vec1()?;
    let t2_flat: Vec<f32> = t2.flatten_all()?.to_vec1()?;

    let t1_mean: f32 = t1_flat.iter().sum::<f32>() / t1_flat.len() as f32;
    let t2_mean: f32 = t2_flat.iter().sum::<f32>() / t2_flat.len() as f32;
    let t1_std: f32 = (t1_flat.iter().map(|&x| (x - t1_mean).powi(2)).sum::<f32>() / t1_flat.len() as f32).sqrt();
    let t2_std: f32 = (t2_flat.iter().map(|&x| (x - t2_mean).powi(2)).sum::<f32>() / t2_flat.len() as f32).sqrt();

    println!("\n=== {} ===", name);
    println!("Shape: {:?}", t1.dims());
    println!("Regular:   mean={:.6} std={:.6} min={:.6} max={:.6}",
             t1_mean, t1_std, t1_flat.iter().copied().fold(f32::INFINITY, f32::min),
             t1_flat.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    println!("Quantized: mean={:.6} std={:.6} min={:.6} max={:.6}",
             t2_mean, t2_std, t2_flat.iter().copied().fold(f32::INFINITY, f32::min),
             t2_flat.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    println!("Difference: min={:.6} avg={:.6} max={:.6}", min_diff, avg_diff, max_diff);

    // Check if means/stds are very different (suggests systematic error)
    let mean_ratio = (t1_mean - t2_mean).abs() / t1_mean.abs().max(0.0001);
    let std_ratio = (t1_std - t2_std).abs() / t1_std.max(0.0001);

    if mean_ratio > 0.01 {
        println!("⚠️  MEAN MISMATCH: {:.2}% difference", mean_ratio * 100.0);
    }
    if std_ratio > 0.1 {
        println!("⚠️  STD MISMATCH: {:.2}% difference", std_ratio * 100.0);
    }

    // Sample values from different regions
    let n = t1_flat.len();
    let indices = [0, n/4, n/2, 3*n/4, n-1];
    print!("Sample values (regular):   ");
    for &i in &indices {
        print!("{:.6} ", t1_flat[i]);
    }
    println!();
    print!("Sample values (quantized): ");
    for &i in &indices {
        print!("{:.6} ", t2_flat[i]);
    }
    println!();

    Ok(())
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

    // ========== KEY CHANGE: Load and cache ALL GGUF tensors ONCE ==========
    println!("=== LOADING AND CACHING GGUF TENSORS ===");
    let mut gguf_tensors = HashMap::new();

    let tensor_names = [
        "token_embd.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "blk.0.ffn_norm.weight",
        "output_norm.weight",
        "output.weight",
    ];

    for name in tensor_names {
        if let Ok(tensor) = gguf.tensor(&mut file, name, &device) {
            let dequantized = tensor.dequantize(&device)?;
            println!("  ✓ Cached {}: {:?}", name, dequantized.dims());
            gguf_tensors.insert(name.to_string(), dequantized);
        } else {
            println!("  ✗ Could not load {}", name);
        }
    }
    // =======================================================================

    println!("\n{}", "=".repeat(80));
    println!("CRITICAL LAYERS ANALYSIS");
    println!("{}", "=".repeat(80));

    // 1. Embeddings - absolutely critical
    if let Some(reg_emb) = regular_tensors.get("model.embed_tokens.weight") {
        if let Some(quant_emb) = gguf_tensors.get("token_embd.weight") {
            detailed_comparison("EMBEDDINGS", reg_emb, quant_emb)?;
        }
    }

    // 2. Layer 0 norms - should be exact or near-exact
    if let Some(reg_norm) = regular_tensors.get("model.layers.0.input_layernorm.weight") {
        if let Some(quant_norm) = gguf_tensors.get("blk.0.attn_norm.weight") {
            detailed_comparison("LAYER 0 INPUT NORM", reg_norm, quant_norm)?;
        }
    }

    // 3. Attention weights - Q, K, V, O
    let attn_weights = [
        ("Q_PROJ", "model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight"),
        ("K_PROJ", "model.layers.0.self_attn.k_proj.weight", "blk.0.attn_k.weight"),
        ("V_PROJ", "model.layers.0.self_attn.v_proj.weight", "blk.0.attn_v.weight"),
        ("O_PROJ", "model.layers.0.self_attn.o_proj.weight", "blk.0.attn_output.weight"),
    ];

    for (name, reg_key, quant_key) in attn_weights {
        if let Some(reg_w) = regular_tensors.get(reg_key) {
            if let Some(quant_w) = gguf_tensors.get(quant_key) {
                detailed_comparison(&format!("LAYER 0 {}", name), reg_w, quant_w)?;
            }
        }
    }

    // ========== Now do row-by-row analysis using CACHED tensor ==========
    if let Some(reg_q) = regular_tensors.get("model.layers.0.self_attn.q_proj.weight") {
        if let Some(quant_q_dq) = gguf_tensors.get("blk.0.attn_q.weight") {
            // Convert both to F32
            let reg_q = reg_q.to_dtype(DType::F32)?;
            let quant_q_dq = quant_q_dq.to_dtype(DType::F32)?;

            println!("\n=== ROW-BY-ROW Q_PROJ COMPARISON ===");

            // Get first 4 rows from each
            for row_idx in 0..4 {
                let reg_row: Vec<f32> = reg_q.i(row_idx)?.to_vec1()?;
                let quant_row: Vec<f32> = quant_q_dq.i(row_idx)?.to_vec1()?;

                println!("\nRow {}:", row_idx);
                println!("  Regular:   {:?}", &reg_row[..8]);
                println!("  Quantized: {:?}", &quant_row[..8]);

                let diff: f32 = reg_row.iter().zip(quant_row.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f32>() / reg_row.len() as f32;
                println!("  Avg diff: {:.8}", diff);
            }

            println!("\n=== INTERLEAVING CHECK ===");
            let reg_row0: Vec<f32> = reg_q.i(0)?.to_vec1()?;
            let reg_row1: Vec<f32> = reg_q.i(1)?.to_vec1()?;
            let quant_row0: Vec<f32> = quant_q_dq.i(0)?.to_vec1()?;
            let quant_row1: Vec<f32> = quant_q_dq.i(1)?.to_vec1()?;
            let quant_row2: Vec<f32> = quant_q_dq.i(2)?.to_vec1()?;

            let match_0_0: f32 = reg_row0.iter().zip(quant_row0.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>() / reg_row0.len() as f32;

            let match_0_1: f32 = reg_row0.iter().zip(quant_row1.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>() / reg_row0.len() as f32;

            let match_1_2: f32 = reg_row1.iter().zip(quant_row2.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>() / reg_row1.len() as f32;

            println!("Regular row 0 vs Quantized row 0: diff = {:.8}", match_0_0);
            println!("Regular row 0 vs Quantized row 1: diff = {:.8}", match_0_1);
            println!("Regular row 1 vs Quantized row 2: diff = {:.8}", match_1_2);

            if match_0_0 < 0.0001 && match_1_2 < 0.0001 {
                println!("\n✓✓✓ Pattern: GGUF has EXTRA row at position 1!");
                println!("GGUF row 0 = Regular row 0");
                println!("GGUF row 1 = ???");
                println!("GGUF row 2 = Regular row 1");
                println!("This explains why deinterleaving works!");
            }

            println!("\n=== TESTING SPLIT-HALF HYPOTHESIS ===");
            let total_rows = reg_q.dim(0)?;
            let half_rows = total_rows / 2;

            println!("Total rows: {}, Half: {}", total_rows, half_rows);

            // Check if GGUF odd rows match Regular second half
            let reg_row_half: Vec<f32> = reg_q.i(half_rows)?.to_vec1()?;

            let match_half: f32 = reg_row_half.iter().zip(quant_row1.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>() / reg_row_half.len() as f32;

            println!("\nRegular row {} vs GGUF row 1: diff = {:.8}", half_rows, match_half);

            if match_half < 0.001 {
                println!("✓✓✓ CONFIRMED: Split-half interleaving!");
                println!("GGUF even rows = Regular first half [0..{}]", half_rows);
                println!("GGUF odd rows = Regular second half [{}..{}]", half_rows, total_rows);
            }

            // Check a few more to confirm pattern
            println!("\n=== VERIFYING SPLIT-HALF PATTERN ===");
            for i in 0..4 {
                let reg_row_first: Vec<f32> = reg_q.i(i)?.to_vec1()?;
                let reg_row_second: Vec<f32> = reg_q.i(half_rows + i)?.to_vec1()?;
                let quant_even: Vec<f32> = quant_q_dq.i(2*i)?.to_vec1()?;
                let quant_odd: Vec<f32> = quant_q_dq.i(2*i + 1)?.to_vec1()?;

                let diff_even: f32 = reg_row_first.iter().zip(quant_even.iter())
                    .map(|(a, b)| (a - b).abs()).sum::<f32>() / reg_row_first.len() as f32;
                let diff_odd: f32 = reg_row_second.iter().zip(quant_odd.iter())
                    .map(|(a, b)| (a - b).abs()).sum::<f32>() / reg_row_second.len() as f32;

                println!("Regular[{:4}] vs GGUF[{:4}]: {:.8} {}",
                         i, 2*i, diff_even, if diff_even < 0.001 { "✓" } else { "✗" });
                println!("Regular[{:4}] vs GGUF[{:4}]: {:.8} {}",
                         half_rows + i, 2*i + 1, diff_odd, if diff_odd < 0.001 { "✓" } else { "✗" });
            }
        }
    }

    // 4. MLP weights
    let mlp_weights = [
        ("GATE_PROJ", "model.layers.0.mlp.gate_proj.weight", "blk.0.ffn_gate.weight"),
        ("UP_PROJ", "model.layers.0.mlp.up_proj.weight", "blk.0.ffn_up.weight"),
        ("DOWN_PROJ", "model.layers.0.mlp.down_proj.weight", "blk.0.ffn_down.weight"),
    ];

    for (name, reg_key, quant_key) in mlp_weights {
        if let Some(reg_w) = regular_tensors.get(reg_key) {
            if let Some(quant_w) = gguf_tensors.get(quant_key) {
                detailed_comparison(&format!("LAYER 0 {}", name), reg_w, quant_w)?;
            }
        }
    }

    // 5. Final layer norm
    if let Some(reg_norm) = regular_tensors.get("model.norm.weight") {
        if let Some(quant_norm) = gguf_tensors.get("output_norm.weight") {
            detailed_comparison("FINAL NORM", reg_norm, quant_norm)?;
        }
    }

    // 6. LM head - critical for output
    if let Some(reg_lm) = regular_tensors.get("lm_head.weight") {
        if let Some(quant_lm) = gguf_tensors.get("output.weight") {
            detailed_comparison("LM HEAD (OUTPUT)", reg_lm, quant_lm)?;
        }
    }

    println!("\n{}", "=".repeat(80));
    println!("DIAGNOSTIC SUMMARY");
    println!("{}", "=".repeat(80));
    println!("\nLook for:");
    println!("  ⚠️  MEAN MISMATCH - suggests weights are scaled differently");
    println!("  ⚠️  STD MISMATCH - suggests different value distributions");
    println!("  Large max_diff but small avg_diff - normal Q8 outliers (OK)");
    println!("  Large avg_diff - serious problem!");
    println!("\nIf split-half hypothesis is confirmed:");
    println!("  GGUF stores Q weights as: [first_half rows interleaved with second_half rows]");
    println!("  Fix: Deinterleave, then concatenate even+odd rows to reconstruct original order");

    Ok(())
}