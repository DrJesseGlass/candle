//! Offline requantizer: rewrite a GGUF with selected tensors requantized to a
//! lower-bit dtype. Used to bake the decode-speed deploy artifact (e.g. drop the
//! tied embedding / output projection from Q6_K to Q4_K) so there is no load-time
//! requant cost and the file is smaller (faster Lambda cold start).
//!
//! Example: requant just the tied embedding (which is also the lm_head):
//!   gguf-requant --input model.gguf --output model-q4out.gguf \
//!                --tensors token_embd,output --dtype q4k
//!
//! `--pack`: bake the candle-only pre-packed Q4_Kx8 interleaved layout for matmul
//! weights already in Q4_K, so the model loads with a SINGLE copy and no runtime
//! repack (mmap-able). token_embd stays Q4_K (embedding lookup); a tied model with
//! no output.weight additionally gets a packed output.weight emitted.
use anyhow::Result;
use candle::quantized::{
    ggml_file, gguf_file, k_quants::BlockQ4K, repack, GgmlDType, GgmlType, QTensor,
};
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
    /// Bake the pre-packed Q4_Kx8 interleaved layout for Q4_K matmul weights.
    #[arg(long, default_value_t = false)]
    pack: bool,
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

/// Matmul weight name suffixes whose Q4_K tensors get the packed layout.
const PACK_MATMUL: &[&str] = &[
    "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down",
];

fn is_packable_matmul(name: &str) -> bool {
    !name.contains("norm")
        && name.ends_with(".weight")
        && PACK_MATMUL.iter().any(|p| name.contains(p))
}

/// Repack a Q4_K weight QTensor `[n, k]` into a pre-packed Q4_Kx8 QTensor.
/// Reinterprets the raw Q4_K bytes as `[BlockQ4K]`, repacks all n/8 channel
/// groups, and builds a Q4Kx8 QTensor from the resulting bytes (CPU).
fn pack_q4k_to_q4kx8(qt: &QTensor) -> Result<QTensor> {
    let (n, k) = qt.shape().dims2()?;
    anyhow::ensure!(qt.dtype() == GgmlDType::Q4K, "pack expects Q4_K input");
    anyhow::ensure!(n % 8 == 0 && k % 256 == 0, "pack needs n%8==0 && k%256==0");
    let nb = k / 256;
    let data = qt.data()?; // raw Q4_K block bytes, row-major (channel r at r*nb)
    let bs = std::mem::size_of::<BlockQ4K>();
    anyhow::ensure!(
        data.len() == n * nb * bs,
        "unexpected Q4_K byte count {} vs {}",
        data.len(),
        n * nb * bs
    );
    // SAFETY: bytes came straight from a Q4_K QTensor (POD #[repr(C)] BlockQ4K),
    // length is an exact multiple of the block size; copy into an aligned Vec first.
    let mut rows = vec![BlockQ4K::zeros(); n * nb];
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            rows.as_mut_ptr() as *mut u8,
            data.len(),
        );
    }
    let packed = repack::repack_q4k_weight(&rows, n, nb);
    let raw: &[u8] = unsafe {
        std::slice::from_raw_parts(
            packed.as_ptr() as *const u8,
            std::mem::size_of_val(packed.as_slice()),
        )
    };
    let tensor = ggml_file::qtensor_from_ggml(GgmlDType::Q4Kx8, raw, vec![n, k], &Device::Cpu)?;
    Ok(tensor)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::Cpu;
    let target = parse_dtype(&args.dtype)?;
    let pats: Vec<&str> = args.tensors.split(',').filter(|s| !s.is_empty()).collect();

    let mut f = std::fs::File::open(&args.input)?;
    let content = gguf_file::Content::read(&mut f).map_err(|e| e.with_path(&args.input))?;

    // Reload + (optionally) requant/pack each tensor; keep owned so the write borrows them.
    let names: Vec<String> = content.tensor_infos.keys().cloned().collect();
    let mut owned: Vec<(String, QTensor)> = Vec::with_capacity(names.len() + 1);
    let (mut before, mut after) = (0usize, 0usize);
    let mut has_output = false;
    let mut token_embd_q4k: Option<QTensor> = None;

    for name in &names {
        if name == "output.weight" {
            has_output = true;
        }
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

        // Stash a Q4_K token_embd for the tied-lm_head emit (use the post-requant copy).
        if args.pack && name == "token_embd.weight" && qt.dtype() == GgmlDType::Q4K {
            if let Ok((n, _)) = qt.shape().dims2() {
                if n % 8 == 0 {
                    let de = qt.dequantize(&device)?;
                    token_embd_q4k = Some(QTensor::quantize(&de, GgmlDType::Q4K)?);
                }
            }
        }

        // Pack eligible matmul weights (keep token_embd as Q4_K for the lookup).
        let qt = if args.pack
            && qt.dtype() == GgmlDType::Q4K
            && is_packable_matmul(name)
            && qt.shape().dims2().map(|(n, k)| n % 8 == 0 && k % 256 == 0).unwrap_or(false)
        {
            let packed = pack_q4k_to_q4kx8(&qt)?;
            println!("  pack    {name}: {:?} -> {:?}", qt.dtype(), GgmlDType::Q4Kx8);
            packed
        } else {
            qt
        };

        after += qt.storage_size_in_bytes();
        owned.push((name.clone(), qt));
    }

    // Tied embeddings: no output.weight present. Emit a packed output.weight from the
    // Q4_K token_embd so the lm_head gets a fast packed weight (token_embd stays Q4_K).
    if args.pack && !has_output {
        if let Some(te) = token_embd_q4k {
            match pack_q4k_to_q4kx8(&te) {
                Ok(packed) => {
                    println!(
                        "  emit    output.weight (tied): {:?} -> {:?}",
                        GgmlDType::Q4K,
                        GgmlDType::Q4Kx8
                    );
                    after += packed.storage_size_in_bytes();
                    owned.push(("output.weight".to_string(), packed));
                }
                Err(e) => println!("  skip tied output.weight pack: {e}"),
            }
        }
    }

    let metadata: Vec<(&str, &gguf_file::Value)> =
        content.metadata.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let tensors: Vec<(&str, &QTensor)> = owned.iter().map(|(n, t)| (n.as_str(), t)).collect();

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
