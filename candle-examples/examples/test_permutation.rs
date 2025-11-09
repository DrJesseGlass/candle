// Test if GGUF columns are permuted/reordered
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

fn find_best_column_match(col_idx: usize, regular: &Tensor, quantized: &Tensor) -> Result<(usize, f32)> {
    // Convert both to F32 first
    let regular_f32 = regular.to_dtype(DType::F32)?;
    let quantized_f32 = quantized.to_dtype(DType::F32)?;

    let dims = regular_f32.dims();
    let num_cols = dims[1];

    // Extract the target column from regular model
    let target_col = regular_f32.narrow(1, col_idx, 1)?;

    let mut best_match_idx = col_idx;
    let mut best_diff = f32::MAX;

    // Try a few columns around the target
    let search_range = 50; // Search +/- 50 columns
    let start = if col_idx > search_range { col_idx - search_range } else { 0 };
    let end = (col_idx + search_range).min(num_cols - 1);

    for test_idx in start..=end {
        let test_col = quantized_f32.narrow(1, test_idx, 1)?;
        let diff = (&target_col - &test_col)?.abs()?.flatten_all()?.mean(0)?.to_scalar::<f32>()?;

        if diff < best_diff {
            best_diff = diff;
            best_match_idx = test_idx;
        }
    }

    Ok((best_match_idx, best_diff))
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

    println!("Loading models...");

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

    let mut file = File::open(&args.quantized)?;
    let gguf = gguf_file::Content::read(&mut file)?;

    println!("\n=== Testing Column Permutation Hypothesis ===\n");

    // Test q_proj column 844 (the main problem column)
    if let Some(reg_q) = regular_tensors.get("model.layers.0.self_attn.q_proj.weight") {
        if let Ok(quant_q) = gguf.tensor(&mut file, "blk.0.attn_q.weight", &device) {
            let quant_q_dequant = quant_q.dequantize(&device)?;

            println!("Testing q_proj column 844:");
            let (best_idx, best_diff) = find_best_column_match(844, reg_q, &quant_q_dequant)?;

            // Get original column 844 diff
            let reg_q_f32 = reg_q.to_dtype(DType::F32)?;
            let quant_q_f32 = quant_q_dequant.to_dtype(DType::F32)?;
            let orig_col = reg_q_f32.narrow(1, 844, 1)?;
            let quant_col_844 = quant_q_f32.narrow(1, 844, 1)?;
            let orig_diff = (&orig_col - &quant_col_844)?.abs()?.flatten_all()?.mean(0)?.to_scalar::<f32>()?;

            println!("  Original mapping (col 844 -> col 844): avg_diff = {:.6}", orig_diff);
            println!("  Best match found (col 844 -> col {}): avg_diff = {:.6}", best_idx, best_diff);

            if best_idx != 844 {
                println!("  ⚠️  Column 844 matches better with column {} (improvement: {:.6})",
                         best_idx, orig_diff - best_diff);
            } else {
                println!("  ✅ Column 844 is in the correct position");
            }
        }
    }

    println!();

    // Test k_proj column 844
    if let Some(reg_k) = regular_tensors.get("model.layers.0.self_attn.k_proj.weight") {
        if let Ok(quant_k) = gguf.tensor(&mut file, "blk.0.attn_k.weight", &device) {
            let quant_k_dequant = quant_k.dequantize(&device)?;

            println!("Testing k_proj column 844:");
            let (best_idx, best_diff) = find_best_column_match(844, reg_k, &quant_k_dequant)?;

            let reg_k_f32 = reg_k.to_dtype(DType::F32)?;
            let quant_k_f32 = quant_k_dequant.to_dtype(DType::F32)?;
            let orig_col = reg_k_f32.narrow(1, 844, 1)?;
            let quant_col_844 = quant_k_f32.narrow(1, 844, 1)?;
            let orig_diff = (&orig_col - &quant_col_844)?.abs()?.flatten_all()?.mean(0)?.to_scalar::<f32>()?;

            println!("  Original mapping (col 844 -> col 844): avg_diff = {:.6}", orig_diff);
            println!("  Best match found (col 844 -> col {}): avg_diff = {:.6}", best_idx, best_diff);

            if best_idx != 844 {
                println!("  ⚠️  Column 844 matches better with column {} (improvement: {:.6})",
                         best_idx, orig_diff - best_diff);
            } else {
                println!("  ✅ Column 844 is in the correct position");
            }
        }
    }

    println!();

    // Test column 683
    if let Some(reg_q) = regular_tensors.get("model.layers.0.self_attn.q_proj.weight") {
        if let Ok(quant_q) = gguf.tensor(&mut file, "blk.0.attn_q.weight", &device) {
            let quant_q_dequant = quant_q.dequantize(&device)?;

            println!("Testing q_proj column 683:");
            let (best_idx, best_diff) = find_best_column_match(683, reg_q, &quant_q_dequant)?;

            let reg_q_f32 = reg_q.to_dtype(DType::F32)?;
            let quant_q_f32 = quant_q_dequant.to_dtype(DType::F32)?;
            let orig_col = reg_q_f32.narrow(1, 683, 1)?;
            let quant_col_683 = quant_q_f32.narrow(1, 683, 1)?;
            let orig_diff = (&orig_col - &quant_col_683)?.abs()?.flatten_all()?.mean(0)?.to_scalar::<f32>()?;

            println!("  Original mapping (col 683 -> col 683): avg_diff = {:.6}", orig_diff);
            println!("  Best match found (col 683 -> col {}): avg_diff = {:.6}", best_idx, best_diff);

            if best_idx != 683 {
                println!("  ⚠️  Column 683 matches better with column {} (improvement: {:.6})",
                         best_idx, orig_diff - best_diff);
            } else {
                println!("  ✅ Column 683 is in the correct position");
            }
        }
    }

    println!("\n=== Conclusion ===");
    println!("If columns are permuted, you'll see different 'best match' indices above.");
    println!("If it's just quantization error, the best match will be the same column.");

    Ok(())
}