//! Offline requantizer: rewrite a GGUF with selected tensors requantized to a
//! lower-bit dtype. Used to bake the decode-speed deploy artifact (e.g. drop the
//! tied embedding / output projection from Q6_K to Q4_K) so there is no load-time
//! requant cost and the file is smaller (faster Lambda cold start).
//!
//! Example: requant just the tied embedding (which is also the lm_head):
//!   gguf-requant --input model.gguf --output model-q4out.gguf \
//!                --tensors token_embd,output --dtype q4k
use anyhow::Result;
use candle::quantized::{gguf_file, GgmlDType, QTensor};
use candle::Device;
use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    input: String,
    #[arg(long)]
    output: String,
    /// Comma-separated substrings; any tensor whose name contains one is requantized.
    #[arg(long, default_value = "token_embd,output")]
    tensors: String,
    /// Target dtype: q4k | q5k | q3k | q4_0.
    #[arg(long, default_value = "q4k")]
    dtype: String,
}

fn parse_dtype(s: &str) -> Result<GgmlDType> {
    Ok(match s {
        "q4k" | "q4_k" => GgmlDType::Q4K,
        "q5k" | "q5_k" => GgmlDType::Q5K,
        "q3k" | "q3_k" => GgmlDType::Q3K,
        "q4_0" => GgmlDType::Q4_0,
        other => anyhow::bail!("unsupported dtype {other}"),
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::Cpu;
    let target = parse_dtype(&args.dtype)?;
    let pats: Vec<&str> = args.tensors.split(',').filter(|s| !s.is_empty()).collect();

    let mut f = std::fs::File::open(&args.input)?;
    let content = gguf_file::Content::read(&mut f).map_err(|e| e.with_path(&args.input))?;

    // Reload + (optionally) requant each tensor; keep owned so the write borrows them.
    let names: Vec<String> = content.tensor_infos.keys().cloned().collect();
    let mut owned: Vec<(String, QTensor)> = Vec::with_capacity(names.len());
    let (mut before, mut after) = (0usize, 0usize);
    for name in &names {
        let qt = content.tensor(&mut f, name, &device)?;
        before += qt.storage_size_in_bytes();
        // Never requant norm weights (tiny F32, precision-critical).
        let is_norm = name.contains("norm");
        let qt = if !is_norm && pats.iter().any(|p| name.contains(p)) && qt.dtype() != target {
            let de = qt.dequantize(&device)?;
            let rq = QTensor::quantize(&de, target)?;
            println!("  requant {name}: {:?} -> {:?}", qt.dtype(), target);
            rq
        } else {
            qt
        };
        after += qt.storage_size_in_bytes();
        owned.push((name.clone(), qt));
    }

    let metadata: Vec<(&str, &gguf_file::Value)> =
        content.metadata.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let tensors: Vec<(&str, &QTensor)> =
        owned.iter().map(|(n, t)| (n.as_str(), t)).collect();

    let mut w = std::fs::File::create(&args.output)?;
    gguf_file::write(&mut w, &metadata, &tensors)?;
    println!(
        "wrote {} ({} tensors): weights {:.1}MB -> {:.1}MB",
        args.output,
        tensors.len(),
        before as f64 / 1e6,
        after as f64 / 1e6,
    );
    Ok(())
}
