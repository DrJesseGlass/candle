// Enhanced comparison to find outlier locations
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

fn find_outliers(name: &str, t1: &Tensor, t2: &Tensor, threshold: f32) -> Result<()> {
    let t1 = t1.to_dtype(DType::F32)?;
    let t2 = t2.to_dtype(DType::F32)?;

    if t1.dims() != t2.dims() {
        println!("❌ {}: Shape mismatch {:?} vs {:?}", name, t1.dims(), t2.dims());
        return Ok(());
    }

    let diff = (&t1 - &t2)?.abs()?;
    let max_diff: f32 = diff.flatten_all()?.max(0)?.to_scalar()?;
    let avg_diff: f32 = diff.flatten_all()?.mean(0)?.to_scalar()?;

    let t1_flat: Vec<f32> = t1.flatten_all()?.to_vec1()?;
    let t2_flat: Vec<f32> = t2.flatten_all()?.to_vec1()?;
    let diff_flat: Vec<f32> = diff.flatten_all()?.to_vec1()?;

    // Find outliers
    let mut outliers: Vec<(usize, f32, f32, f32)> = vec![];
    for (i, &d) in diff_flat.iter().enumerate() {
        if d > threshold {
            outliers.push((i, t1_flat[i], t2_flat[i], d));
        }
    }

    let status = if max_diff < 0.01 { "✅" } else if max_diff < 0.1 { "⚠️ " } else { "❌" };

    println!("{} {} - {:?} - max_diff={:.6} avg_diff={:.6}",
             status, name, t1.dims(), max_diff, avg_diff);
    println!("   Found {} outliers (diff > {})", outliers.len(), threshold);

    if !outliers.is_empty() {
        // Sort by difference magnitude
        outliers.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

        // Show top 10 worst outliers
        println!("   Top 10 worst outliers:");
        for (i, (idx, v1, v2, d)) in outliers.iter().take(10).enumerate() {
            let dims = t1.dims();
            let coords = if dims.len() == 2 {
                format!("[{}, {}]", idx / dims[1], idx % dims[1])
            } else {
                format!("[{}]", idx)
            };
            println!("     {}. idx={} coords={} regular={:.6} quantized={:.6} diff={:.6}",
                     i+1, idx, coords, v1, v2, d);
        }

        // Statistics
        let outlier_diffs: Vec<f32> = outliers.iter().map(|x| x.3).collect();
        let outlier_avg: f32 = outlier_diffs.iter().sum::<f32>() / outlier_diffs.len() as f32;
        println!("   Outlier stats: count={} avg_diff={:.6} max_diff={:.6}",
                 outliers.len(), outlier_avg, outlier_diffs[0]);

        // Percentage of bad values
        let total = t1_flat.len();
        let outlier_pct = (outliers.len() as f32 / total as f32) * 100.0;
        println!("   Percentage of outliers: {:.4}% ({}/{})",
                 outlier_pct, outliers.len(), total);
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

    println!("Loading regular model...");

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

    println!("Found {} safetensors file(s)", safetensors_files.len());
    let regular_tensors = load_safetensors(&safetensors_files, &device)?;
    println!("Loaded {} tensors from regular model", regular_tensors.len());

    println!("\nLoading quantized model...");
    let mut file = File::open(&args.quantized)?;
    let gguf = gguf_file::Content::read(&mut file)?;
    println!("Loaded GGUF file\n");

    println!("=== Analyzing Problematic Tensors ===\n");

    // Focus on q_proj and k_proj
    let problem_tensors = vec![
        ("model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight", "q_proj", 0.5),
        ("model.layers.0.self_attn.k_proj.weight", "blk.0.attn_k.weight", "k_proj", 0.5),
        ("model.layers.0.self_attn.v_proj.weight", "blk.0.attn_v.weight", "v_proj", 0.001),
        ("model.layers.0.self_attn.o_proj.weight", "blk.0.attn_output.weight", "o_proj", 0.001),
    ];

    for (reg_name, quant_name, short_name, threshold) in problem_tensors {
        if let Some(reg_t) = regular_tensors.get(reg_name) {
            if let Ok(quant_t) = gguf.tensor(&mut file, quant_name, &device) {
                let quant_t_dequant = quant_t.dequantize(&device)?;
                find_outliers(short_name, reg_t, &quant_t_dequant, threshold)?;
            }
        }
    }

    Ok(())
}