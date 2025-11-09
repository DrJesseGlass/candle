// Specifically debug LM_HEAD / output layer
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

    let mut file = File::open(&args.quantized)?;
    let gguf = gguf_file::Content::read(&mut file)?;

    println!("=== SEARCHING FOR OUTPUT/LM_HEAD LAYER ===\n");

    // Search for lm_head in regular model
    println!("Looking for lm_head in regular model:");
    let lm_head_candidates = ["lm_head.weight", "lm_head", "output.weight", "model.lm_head.weight"];
    let mut found_regular = None;

    for candidate in lm_head_candidates {
        if let Some(tensor) = regular_tensors.get(candidate) {
            println!("  ✅ Found '{}' - shape: {:?}", candidate, tensor.dims());
            found_regular = Some((candidate, tensor));
            break;
        } else {
            println!("  ❌ Not found: '{}'", candidate);
        }
    }

    // List all tensors with "lm" or "output" or "head" in name
    println!("\nAll tensors in regular model containing 'lm', 'output', or 'head':");
    for (name, tensor) in regular_tensors.iter() {
        if name.to_lowercase().contains("lm") ||
           name.to_lowercase().contains("output") ||
           name.to_lowercase().contains("head") {
            println!("  - {} : {:?}", name, tensor.dims());
        }
    }

    // Search for output in GGUF
    println!("\nLooking for output layer in GGUF:");
    let gguf_candidates = ["output.weight", "lm_head.weight", "output_weight", "token_embd.weight"];
    let mut found_gguf = None;

    for candidate in gguf_candidates {
        match gguf.tensor(&mut file, candidate, &device) {
            Ok(tensor) => {
                let tensor_dq = tensor.dequantize(&device)?;
                println!("  ✅ Found '{}' - shape: {:?}", candidate, tensor_dq.dims());
                found_gguf = Some((candidate, tensor_dq));
                break;
            }
            Err(_) => {
                println!("  ❌ Not found: '{}'", candidate);
            }
        }
    }

    // List all tensors in GGUF with "output" in name
    println!("\nAll tensors in GGUF containing 'output':");
    for (name, info) in gguf.tensor_infos.iter() {
        if name.to_lowercase().contains("output") {
            println!("  - {} : {:?}", name, info.shape);
        }
    }

    println!("\n=== COMPARISON ===\n");

    if let Some((reg_name, reg_tensor)) = found_regular {
        if let Some((gguf_name, gguf_tensor)) = found_gguf {
            let reg_f32 = reg_tensor.to_dtype(DType::F32)?;
            let gguf_f32 = gguf_tensor.to_dtype(DType::F32)?;

            println!("Comparing:");
            println!("  Regular: {} {:?}", reg_name, reg_f32.dims());
            println!("  GGUF:    {} {:?}", gguf_name, gguf_f32.dims());

            if reg_f32.dims() == gguf_f32.dims() {
                let diff = (&reg_f32 - &gguf_f32)?.abs()?;
                let max_diff: f32 = diff.flatten_all()?.max(0)?.to_scalar()?;
                let avg_diff: f32 = diff.flatten_all()?.mean(0)?.to_scalar()?;

                let reg_vec: Vec<f32> = reg_f32.flatten_all()?.to_vec1()?;
                let gguf_vec: Vec<f32> = gguf_f32.flatten_all()?.to_vec1()?;

                let reg_mean: f32 = reg_vec.iter().sum::<f32>() / reg_vec.len() as f32;
                let gguf_mean: f32 = gguf_vec.iter().sum::<f32>() / gguf_vec.len() as f32;

                println!("\n  Regular mean:   {:.6}", reg_mean);
                println!("  GGUF mean:      {:.6}", gguf_mean);
                println!("  Avg difference: {:.6}", avg_diff);
                println!("  Max difference: {:.6}", max_diff);

                let n = reg_vec.len().min(10);
                println!("\n  First 10 values:");
                println!("    Regular: {:?}", &reg_vec[..n]);
                println!("    GGUF:    {:?}", &gguf_vec[..n]);

                if avg_diff > 0.01 {
                    println!("\n  🚨 HIGH AVERAGE DIFFERENCE - This could cause wrong outputs!");
                } else if max_diff > 0.1 {
                    println!("\n  ⚠️  Some outliers but average is good");
                } else {
                    println!("\n  ✅ Excellent match!");
                }

                // Check if they're tied to embeddings
                if let Some(embed) = regular_tensors.get("model.embed_tokens.weight") {
                    let embed_f32 = embed.to_dtype(DType::F32)?;
                    if embed_f32.dims() == reg_f32.dims() {
                        let embed_diff = (&embed_f32 - &reg_f32)?.abs()?;
                        let embed_avg: f32 = embed_diff.flatten_all()?.mean(0)?.to_scalar()?;
                        if embed_avg < 0.0001 {
                            println!("\n  📎 LM_HEAD is TIED to embeddings (weights are identical)");
                        }
                    }
                }
            } else {
                println!("\n  ❌ SHAPE MISMATCH - tensors have different shapes!");
            }
        } else {
            println!("❌ Could not find output layer in GGUF!");
        }
    } else {
        println!("❌ Could not find lm_head in regular model!");
    }

    Ok(())
}