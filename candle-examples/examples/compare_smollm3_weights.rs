// Comprehensive diagnostic - find EXACTLY where and why outputs diverge
// Place in: candle-examples/examples/compare_smollm3_weights.rs

use anyhow::Result;
use candle::{DType, Device, Tensor};
use candle::quantized::gguf_file;
use clap::Parser;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

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
    println!("✅ Loaded GGUF file with {} tensors", gguf.tensor_infos.len());

    println!("\n{}", "=".repeat(80));
    println!("CRITICAL LAYERS ANALYSIS");
    println!("{}", "=".repeat(80));

    // 1. Embeddings - absolutely critical
    if let Some(reg_emb) = regular_tensors.get("model.embed_tokens.weight") {
        if let Ok(quant_emb) = gguf.tensor(&mut file, "token_embd.weight", &device) {
            let quant_emb_dq = quant_emb.dequantize(&device)?;
            detailed_comparison("EMBEDDINGS", reg_emb, &quant_emb_dq)?;
        }
    }

    // 2. Layer 0 norms - should be exact or near-exact
    if let Some(reg_norm) = regular_tensors.get("model.layers.0.input_layernorm.weight") {
        if let Ok(quant_norm) = gguf.tensor(&mut file, "blk.0.attn_norm.weight", &device) {
            let quant_norm_dq = quant_norm.dequantize(&device)?;
            detailed_comparison("LAYER 0 INPUT NORM", reg_norm, &quant_norm_dq)?;
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
            if let Ok(quant_w) = gguf.tensor(&mut file, quant_key, &device) {
                let quant_w_dq = quant_w.dequantize(&device)?;
                detailed_comparison(&format!("LAYER 0 {}", name), reg_w, &quant_w_dq)?;
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
            if let Ok(quant_w) = gguf.tensor(&mut file, quant_key, &device) {
                let quant_w_dq = quant_w.dequantize(&device)?;
                detailed_comparison(&format!("LAYER 0 {}", name), reg_w, &quant_w_dq)?;
            }
        }
    }

    // 5. Final layer norm
    if let Some(reg_norm) = regular_tensors.get("model.norm.weight") {
        if let Ok(quant_norm) = gguf.tensor(&mut file, "output_norm.weight", &device) {
            let quant_norm_dq = quant_norm.dequantize(&device)?;
            detailed_comparison("FINAL NORM", reg_norm, &quant_norm_dq)?;
        }
    }

    // 6. LM head - critical for output
    if let Some(reg_lm) = regular_tensors.get("lm_head.weight") {
        if let Ok(quant_lm) = gguf.tensor(&mut file, "output.weight", &device) {
            let quant_lm_dq = quant_lm.dequantize(&device)?;
            detailed_comparison("LM HEAD (OUTPUT)", reg_lm, &quant_lm_dq)?;
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

    Ok(())
}