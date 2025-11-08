//! SmolLM3 quantized implementation - FIXED VERSION
//!
//! Key fixes:
//! - Removed Q/K norm completely (SmolLM3 doesn't have it)
//! - Better metadata key handling
//! - Added debugging for NoPE layers
//! - Improved error messages

use crate::models::with_tracing::QMatMul;
use crate::quantized_nn::RmsNorm;
use crate::utils::repeat_kv;
use candle::quantized::{gguf_file, QTensor};
use candle::{DType, Device, Result, Tensor};
use candle_nn::{kv_cache::KvCache, Activation, Embedding, Module};
use std::io::{Read, Seek};
use std::sync::Arc;

struct Gguf<R: Read + Seek> {
    ct: gguf_file::Content,
    reader: R,
    device: Device,
}

impl<R: Read + Seek> Gguf<R> {
    fn new(ct: gguf_file::Content, reader: R, device: Device) -> Self {
        Self { ct, reader, device }
    }

    fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        QMatMul::from_weights(ws.into())
    }

    fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        RmsNorm::from_qtensor(ws, eps)
    }

    fn metadata(&self) -> &std::collections::HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }

    fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(&mut self.reader, name, &self.device)
    }
}

#[derive(Debug, Clone)]
struct MlpWeights {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
    act_fn: Activation,
    span: tracing::Span,
}

impl MlpWeights {
    fn new<R: Read + Seek>(gg: &mut Gguf<R>, prefix: &str) -> Result<Self> {
        let gate_proj = gg.qmatmul(&format!("{prefix}.ffn_gate.weight"))?;
        let up_proj = gg.qmatmul(&format!("{prefix}.ffn_up.weight"))?;
        let down_proj = gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?;
        let act_fn = Activation::Silu;
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn,
            span,
        })
    }
}

impl Module for MlpWeights {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let gate = self.gate_proj.forward(x)?.apply(&self.act_fn)?;
        let up = self.up_proj.forward(x)?;
        let gated = (gate * up)?;
        self.down_proj.forward(&gated)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        dev: &Device,
    ) -> Result<Self> {
        let dim = head_dim;
        let max_seq_len = max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        // Always compute in F32 for precision, then convert (like non-quantized)
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            // Compute sin/cos in F32, then convert to target dtype
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct AttentionWeights {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,  // Added: necessary for correct output projection
    rotary_emb: Option<Arc<RotaryEmbedding>>,
    kv_cache: KvCache,
    skip_rope: bool,  // SmolLM3: NoPE support
    span_attn: tracing::Span,
}

impl AttentionWeights {
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rotary_emb: Option<Arc<RotaryEmbedding>>,
        skip_rope: bool,
        prefix: &str,
    ) -> Result<Self> {
        let num_kv_groups = num_heads / num_kv_heads;

        let q_proj = gg.qmatmul(&format!("{prefix}.attn_q.weight"))?;
        let k_proj = gg.qmatmul(&format!("{prefix}.attn_k.weight"))?;
        let v_proj = gg.qmatmul(&format!("{prefix}.attn_v.weight"))?;
        let o_proj = gg.qmatmul(&format!("{prefix}.attn_output.weight"))?;

        let kv_cache = KvCache::new(2, 512);
        let span_attn = tracing::span!(tracing::Level::TRACE, "attn");

        // Calculate hidden_size from attention heads (same as non-quantized version)
        let hidden_size = head_dim * num_heads;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size,
            rotary_emb,
            kv_cache,
            skip_rope,
            span_attn,
        })
    }

    fn forward(&mut self, x: &Tensor, attn_mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b, l, _) = x.dims3()?;

        // 1. Project Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // 2. Reshape to (B, H, L, D)
        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // 3. SmolLM3: Conditionally apply RoPE (NO Q/K NORM)
        let (q, k) = if self.skip_rope {
            // NoPE: Skip rotary embeddings
            (q.contiguous()?, k.contiguous()?)
        } else {
            // Apply RoPE
            if let Some(ref rope) = self.rotary_emb {
                rope.apply(&q, &k, offset)?
            } else {
                (q, k)
            }
        };

        // 4. Update KV cache
        if offset == 0 {
            self.kv_cache.reset();
        }
        let (k, v) = self.kv_cache.append(&k.contiguous()?, &v.contiguous()?)?;

        // 5. Repeat KV for GQA
        let k = repeat_kv(k, self.num_kv_groups)?;
        let v = repeat_kv(v, self.num_kv_groups)?;

        // 6. Attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;

        // 7. Output projection
        // CRITICAL: ensure tensor is contiguous before reshape
        let ctx_transposed = ctx.transpose(1, 2)?;
        let ctx_reshaped = ctx_transposed.contiguous()?.reshape((b, l, self.hidden_size))?;
        ctx_reshaped.apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    self_attn: AttentionWeights,
    mlp: MlpWeights,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl LayerWeights {
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary: Option<Arc<RotaryEmbedding>>,
        layer_idx: usize,
        no_rope_layer_interval: Option<usize>,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");

        let ln1 = gg.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_norm_eps)?;
        let ln2 = gg.rms_norm(&format!("{prefix}.ffn_norm.weight"), rms_norm_eps)?;

        // SmolLM3: Determine if this layer should skip RoPE (NoPE)
        let skip_rope = if let Some(interval) = no_rope_layer_interval {
            (layer_idx + 1) % interval == 0
        } else {
            false
        };

        let self_attn = AttentionWeights::new(
            gg,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rotary,
            skip_rope,
            &prefix,
        )?;
        let mlp = MlpWeights::new(gg, &prefix)?;

        Ok(Self {
            self_attn,
            mlp,
            ln1,
            ln2,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(&h, mask, offset)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = h2.apply(&self.mlp)?;
        x.add(&h2)
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    embed_tokens: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    lm_head: QMatMul,
    device: Device,
    dtype: DType,
    span: tracing::Span,
    span_output: tracing::Span,
}

impl ModelWeights {
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone());

        // Try different metadata key formats
        let md_get = |s: &str| match gg.metadata().get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        // Debug: Print all metadata keys
        println!("Available metadata keys:");
        for key in gg.metadata().keys() {
            println!("  - {}", key);
        }

        // Try to read metadata with fallbacks for different GGUF formats
        let num_attention_heads = md_get("smollm3.attention.head_count")
            .or_else(|_| md_get("llama.attention.head_count"))
            .or_else(|_| md_get("attention.head_count"))?
            .to_u32()? as usize;

        let num_kv_heads = md_get("smollm3.attention.head_count_kv")
            .or_else(|_| md_get("llama.attention.head_count_kv"))
            .or_else(|_| md_get("attention.head_count_kv"))?
            .to_u32()? as usize;

        let rope_freq_base = md_get("smollm3.rope.freq_base")
            .or_else(|_| md_get("llama.rope.freq_base"))
            .or_else(|_| md_get("rope.freq_base"))?
            .to_f32()? as f64;

        let head_dim = md_get("smollm3.rope.dimension_count")
            .or_else(|_| md_get("llama.rope.dimension_count"))
            .or_else(|_| md_get("rope.dimension_count"))?
            .to_u32()? as usize;

        let num_layers = md_get("smollm3.block_count")
            .or_else(|_| md_get("llama.block_count"))
            .or_else(|_| md_get("block_count"))?
            .to_u32()? as usize;

        let hidden_size = md_get("smollm3.embedding_length")
            .or_else(|_| md_get("llama.embedding_length"))
            .or_else(|_| md_get("embedding_length"))?
            .to_u32()? as usize;

        let max_position_embeddings = md_get("smollm3.context_length")
            .or_else(|_| md_get("llama.context_length"))
            .or_else(|_| md_get("context_length"))?
            .to_u32()? as usize;

        let rms_norm_eps = md_get("smollm3.attention.layer_norm_rms_epsilon")
            .or_else(|_| md_get("llama.attention.layer_norm_rms_epsilon"))
            .or_else(|_| md_get("attention.layer_norm_rms_epsilon"))?
            .to_f32()? as f64;

        // SmolLM3: Try to read NoPE interval from metadata, fallback to 4
        let no_rope_layer_interval = md_get("smollm3.no_rope_layer_interval")
            .or_else(|_| md_get("llama.no_rope_layer_interval"))
            .ok()
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .or(Some(4)); // Default to 4 for SmolLM3-3B

        println!("\n=== SmolLM3 Configuration ===");
        println!("Layers: {}", num_layers);
        println!("Attention heads: {} (KV heads: {})", num_attention_heads, num_kv_heads);
        println!("Head dim: {}", head_dim);
        println!("Hidden size: {}", hidden_size);
        println!("RoPE theta: {}", rope_freq_base);
        println!("Max position: {}", max_position_embeddings);

        if let Some(interval) = no_rope_layer_interval {
            let nope_layers: Vec<_> = (0..num_layers)
                .filter(|&i| (i + 1) % interval == 0)
                .collect();
            println!("\n=== NoPE Configuration ===");
            println!("Interval: every {}th layer skips RoPE", interval);
            println!("NoPE layers: {:?}", nope_layers);
            println!("Total: {} RoPE layers, {} NoPE layers",
                num_layers - nope_layers.len(), nope_layers.len());
        }

        let dtype = match gg.metadata().get("general.dtype") {
            Some(v) => match v.to_u32() {
                Ok(0) => DType::F32,
                Ok(1) => DType::F16,
                _ => DType::F16,
            },
            None => DType::F16,
        };

        let embed_tensor = gg.tensor("token_embd.weight")?;
        let embed_shape = embed_tensor.shape();
        println!("GGUF token_embd.weight shape: {:?}", embed_shape);

        let dequantized = embed_tensor.dequantize(device)?;
        println!("Dequantized embedding shape: {:?}", dequantized.shape());

        // Check a few actual values
        let embed_flat = dequantized.flatten_all()?.to_vec1::<f32>()?;
        println!("First 10 embedding values: {:?}", &embed_flat[0..10]);
        println!("Values around token 12366 (Paris): {:?}",
                 &embed_flat[12366*2048..12366*2048+10]);
        let embed_tokens = Embedding::new(embed_tensor.dequantize(device)?, hidden_size);

        // Only create rotary embedding if at least one layer uses RoPE
        let needs_rope = if let Some(interval) = no_rope_layer_interval {
            (0..num_layers).any(|i| (i + 1) % interval != 0)
        } else {
            true
        };

        let rotary = if needs_rope {
            Some(Arc::new(RotaryEmbedding::new(
                dtype,
                head_dim,
                max_position_embeddings,
                rope_freq_base,
                device,
            )?))
        } else {
            None
        };

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(LayerWeights::new(
                &mut gg,
                num_attention_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                rotary.clone(),
                i,
                no_rope_layer_interval,
            )?);
        }

        println!("model built");

        // NOW ADD THESE LINES:
        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;


        // Add debug
        let embed_tensor_debug = gg.tensor("token_embd.weight")?;
        println!("GGUF token_embd.weight shape: {:?}", embed_tensor_debug.shape());

        // SmolLM3 uses tied embeddings
        let lm_head = QMatMul::from_weights(embed_tensor.into())?;

        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype,
            span,
            span_output,
        })
    }

    fn causal_mask(
        &self,
        b: usize,
        tgt: usize,
        offset: usize,
        sw: Option<usize>,
        dtype: DType,  // Use actual tensor dtype instead of self.dtype
    ) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| {
                (0..(tgt + offset)).map(move |j| {
                    let past_ok = j <= i + offset;
                    let sw_ok = match sw {
                        Some(w) => (i + offset) as i64 - j as i64 <= w as i64,
                        None => true,
                    };
                    if past_ok && sw_ok {
                        0.
                    } else {
                        minf
                    }
                })
            })
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(dtype)
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b, l) = input.dims2()?;

        // DEBUG: Print input info
        if offset <= 5 {
            println!("\n=== Forward pass offset={} ===", offset);
            println!("Input shape: {:?}", input.shape());
            if l == 1 {
                let token_id = input.to_vec2::<u32>()?[0][0];
                println!("Processing single token: {}", token_id);
            }
        }

        let mut h = self.embed_tokens.forward(input)?;

        if offset <= 5 {
            let h_stats = h.flatten_all()?.to_vec1::<f32>()?;
            let h_mean: f32 = h_stats.iter().sum::<f32>() / h_stats.len() as f32;
            println!("After embeddings - mean: {:.6}", h_mean);
        }

        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset, None, h.dtype())?)
        };

        for (idx, layer) in self.layers.iter_mut().enumerate() {
            h = layer.forward(&h, causal_mask.as_ref(), offset)?;
            if offset <= 5 && idx % 10 == 0 {
                let h_stats = h.flatten_all()?.to_vec1::<f32>()?;
                let h_mean: f32 = h_stats.iter().sum::<f32>() / h_stats.len() as f32;
                println!("After layer {} - mean: {:.6}", idx, h_mean);
            }
        }

        let h = self.norm.forward(&h)?;

        if offset <= 5 {
            let h_stats = h.flatten_all()?.to_vec1::<f32>()?;
            let h_mean: f32 = h_stats.iter().sum::<f32>() / h_stats.len() as f32;
            println!("After norm - mean: {:.6}", h_mean);
        }

        let _enter = self.span_output.enter();
        let last_hidden = h.narrow(1, l - 1, 1)?;
        let logits = self.lm_head.forward(&last_hidden)?.squeeze(1)?;

        if offset <= 5 {
            let logits_vec = logits.flatten_all()?.to_vec1::<f32>()?;
            let mut indexed: Vec<_> = logits_vec.iter().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
            println!("Top 3 logits:");
            for (idx, val) in indexed.iter().take(3) {
                println!("  Token {}: {:.4}", idx, val);
            }
        }

        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}