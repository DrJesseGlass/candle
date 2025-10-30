//! Qwen3 implementation with quantization support.
//!
//! Based on the Qwen3 architecture and implemented with quantized weights
//! for reduced memory usage and faster inference on compatible hardware.
//!
//! References:
//! - [Qwen3 Models](https://huggingface.co/Qwen/Qwen3-0.6B) (architecture based on official implementations)
//!
use super::with_tracing::QMatMul;
use crate::{quantized_nn::RmsNorm, utils::repeat_kv};
use candle::quantized::{gguf_file, QTensor};
use candle::{DType, Device, Result, Tensor};
use candle_nn::{Activation, Embedding, Module};
use std::io::{Read, Seek};
use std::sync::Arc;

// WASM optimizations:
#[cfg(target_arch = "wasm32")]
use candle_nn::kv_cache::WasmKvCache as KvCache;

// Use standard KV cache on non-WASM
#[cfg(not(target_arch = "wasm32"))]
use candle_nn::kv_cache::KvCache;

// ============================================================================
// Conditional compilation for flash attention backends
// ============================================================================

#[cfg(feature = "flash-attn")]
use candle_flash_attn;

#[cfg(all(not(feature = "flash-attn"), not(feature = "simd-flash-attn")))]
use candle_nn::cpu_flash_attention;

#[cfg(all(not(feature = "flash-attn"), feature = "simd-flash-attn"))]
use candle_nn::simd_flash_attention;

// ============================================================================
// Configuration for optimizations
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComputeMode {
    /// Use the dtype from GGUF metadata (dynamic)
    Dynamic,
    /// Force F32 for all activations (better for SIMD in WASM)
    ForceF32,
    /// Force F16 for all activations
    ForceF16,
}

impl Default for ComputeMode {
    fn default() -> Self {
        ComputeMode::Dynamic
    }
}

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

// ============================================================================
// Rotary embeddings with configurable dtype
// ============================================================================

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    dtype: DType,
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
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
            dtype,
        })
    }

    /// Apply RoPE (q, k shape: B x H x L x D)
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
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    use_flash_attn: bool,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: KvCache,
    span_attn: tracing::Span,
}

impl AttentionWeights {
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        use_flash_attn: bool,
        rotary_emb: Arc<RotaryEmbedding>,
        prefix: &str,
    ) -> Result<Self> {
        let num_kv_groups = num_heads / num_kv_heads;

        let q_proj = gg.qmatmul(&format!("{prefix}.attn_q.weight"))?;
        let k_proj = gg.qmatmul(&format!("{prefix}.attn_k.weight"))?;
        let v_proj = gg.qmatmul(&format!("{prefix}.attn_v.weight"))?;
        let o_proj = gg.qmatmul(&format!("{prefix}.attn_output.weight"))?;

        let q_norm = gg.rms_norm(&format!("{prefix}.attn_q_norm.weight"), rms_norm_eps)?;
        let k_norm = gg.rms_norm(&format!("{prefix}.attn_k_norm.weight"), rms_norm_eps)?;

        // Initialize KV cache with 512 tokens capacity to reduce initial memory allocation.
        // The cache will grow in chunks of 512 tokens when needed.
        //let kv_cache = KvCache::new(2, 512);
        let kv_cache = KvCache::new(2, 256);
        //let kv_cache = KvCache::new(2, 101);

        let span_attn = tracing::span!(tracing::Level::TRACE, "attn");

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            use_flash_attn,
            rotary_emb,
            kv_cache,
            span_attn,
        })
    }

    fn forward(&mut self, x: &Tensor, attn_mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b, l, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;

        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        // Reset KV cache if we're at the first position
        if offset == 0 {
            self.kv_cache.reset();
        }
        let (k, v) = self.kv_cache.append(&k.contiguous()?, &v.contiguous()?)?;

        // Make tensor contiguous to avoid some strided copies
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        if self.use_flash_attn {
            self.forward_flash_attn(&q, &k, &v, attn_mask, b, l)
        } else {
            self.forward_standard_attn(&q, &k, &v, attn_mask, b, l)
        }
    }



    /// GPU flash attention (CUDA)
    #[cfg(feature = "flash-attn")]
    fn forward_flash_attn(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        _attn_mask: Option<&Tensor>,
        b: usize,
        l: usize,
    ) -> Result<Tensor> {
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let ctx = candle_flash_attn::flash_attn(&q, &k, &v, scale, true)?;

        ctx.reshape((b, l, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)
    }

    /// SIMD-optimized CPU flash attention (uses loop bounds, no masking)
    #[cfg(all(not(feature = "flash-attn"), feature = "simd-flash-attn"))]
    fn forward_flash_attn(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        _attn_mask: Option<&Tensor>,
        b: usize,
        l: usize,
    ) -> Result<Tensor> {
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f32).sqrt();

        // SIMD flash attention works with dequantized tensors
        let ctx = match q.dtype() {
            DType::F32 => simd_flash_attention::run_simd_flash_attn_cpu::<f32>(
                &q, &k, &v, scale, true  // is_causal = true
            )?,
            DType::F64 => simd_flash_attention::run_simd_flash_attn_cpu::<f64>(
                &q, &k, &v, scale, true
            )?,
            DType::BF16 => {
                let q_f32 = q.to_dtype(DType::F32)?;
                let k_f32 = k.to_dtype(DType::F32)?;
                let v_f32 = v.to_dtype(DType::F32)?;
                let ctx_f32 = simd_flash_attention::run_simd_flash_attn_cpu::<f32>(
                    &q_f32, &k_f32, &v_f32, scale, true
                )?;
                ctx_f32.to_dtype(DType::BF16)?
            }
            dtype => candle::bail!("Unsupported dtype for SIMD flash attention: {:?}", dtype),
        };

        let ctx = ctx.transpose(1, 2)?;
        ctx.reshape((b, l, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)
    }

    /// Standard CPU flash attention (uses mask tensor)
    #[cfg(all(not(feature = "flash-attn"), not(feature = "simd-flash-attn")))]
    fn forward_flash_attn(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attn_mask: Option<&Tensor>,
        b: usize,
        l: usize,
    ) -> Result<Tensor> {
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f32).sqrt();

        let ctx = match q.dtype() {
            DType::F32 => cpu_flash_attention::run_flash_attn_cpu::<f32>(
                &q, &k, &v, attn_mask, scale, None, None,
            )?,
            DType::F64 => cpu_flash_attention::run_flash_attn_cpu::<f64>(
                &q, &k, &v, attn_mask, scale, None, None,
            )?,
            DType::BF16 => {
                let q_f32 = q.to_dtype(DType::F32)?;
                let k_f32 = k.to_dtype(DType::F32)?;
                let v_f32 = v.to_dtype(DType::F32)?;
                let ctx_f32 = cpu_flash_attention::run_flash_attn_cpu::<f32>(
                    &q_f32, &k_f32, &v_f32, attn_mask, scale, None, None,
                )?;
                ctx_f32.to_dtype(DType::BF16)?
            }
            dtype => candle::bail!("Unsupported dtype for CPU flash attention: {:?}", dtype),
        };

        let ctx = ctx.transpose(1, 2)?;
        ctx.reshape((b, l, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)
    }

    fn forward_standard_attn(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attn_mask: Option<&Tensor>,
        b: usize,
        l: usize,
    ) -> Result<Tensor> {
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        if let Some(m) = attn_mask {
            let m = if m.dtype() != scores.dtype() {
                m.to_dtype(scores.dtype())?
            } else {
                m.clone()
            };
            scores = scores.broadcast_add(&m)?;
        }

        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;
        let reshaped_ctx = ctx
            .transpose(1, 2)?
            .reshape((b, l, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&reshaped_ctx)
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
        use_flash_attn: bool,
        rotary: Arc<RotaryEmbedding>,
        layer_idx: usize,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");

        let ln1 = gg.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_norm_eps)?;
        let ln2 = gg.rms_norm(&format!("{prefix}.ffn_norm.weight"), rms_norm_eps)?;
        let self_attn = AttentionWeights::new(
            gg,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            use_flash_attn,
            rotary,
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
        x + h2
    }
}

// ============================================================================
// Model with configurable optimizations
// ============================================================================

#[derive(Debug, Clone)]
pub struct ModelWeights {
    embed_tokens: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    lm_head: QMatMul,
    device: Device,
    dtype: DType,
    cache_masks: bool,
    cached_masks: Option<std::collections::HashMap<(usize, usize), Tensor>>,
    span: tracing::Span,
    span_output: tracing::Span,
}

impl ModelWeights {
    /// Create model with dynamic dtype from GGUF metadata
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        Self::from_gguf_with_config(ct, reader, device, ComputeMode::Dynamic, false, false)
    }

    /// Create model with configurable compute mode, flash attention, and mask caching
    pub fn from_gguf_with_config<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        compute_mode: ComputeMode,
        use_flash_attn: bool,
        cache_masks: bool,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone());
        let md_get = |s: &str| match gg.metadata().get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        let num_attention_heads = md_get("qwen3.attention.head_count")?.to_u32()? as usize;
        let num_kv_heads = md_get("qwen3.attention.head_count_kv")?.to_u32()? as usize;
        let head_dim = md_get("qwen3.attention.key_length")?.to_u32()? as usize;
        let num_layers = md_get("qwen3.block_count")?.to_u32()? as usize;
        let hidden_size = md_get("qwen3.embedding_length")?.to_u32()? as usize;
        let max_position_embeddings = md_get("qwen3.context_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("qwen3.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("qwen3.rope.freq_base")?.to_f32()? as f64;

        // Determine dtype based on mode
        let dtype = match compute_mode {
            ComputeMode::Dynamic => {
                match gg.metadata().get("general.dtype") {
                    Some(v) => match v.to_u32() {
                        Ok(0) => DType::F32,
                        Ok(1) => DType::F16,
                        _ => DType::F16,
                    },
                    None => DType::F16,
                }
            }
            ComputeMode::ForceF32 => DType::F32,
            ComputeMode::ForceF16 => DType::F16,
        };
        eprintln!("🔍 Using dtype: {:?}", dtype);

        // Dequantize embeddings to the configured dtype
        let embed_tensor = gg.tensor("token_embd.weight")?;
        let embed_weights = embed_tensor.dequantize(device)?;
        let embed_weights = if embed_weights.dtype() != dtype {
            embed_weights.to_dtype(dtype)?
        } else {
            embed_weights
        };
        let embed_tokens = Embedding::new(embed_weights, hidden_size);

        // Create rotary embeddings with configured dtype
        let rotary = Arc::new(RotaryEmbedding::new(
            dtype,
            head_dim,
            max_position_embeddings,
            rope_freq_base,
            device,
        )?);

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(LayerWeights::new(
                &mut gg,
                num_attention_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                use_flash_attn,
                rotary.clone(),
                i,
            )?);
        }

        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;

        let lm_head_tensor = match gg.tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => gg.tensor("token_embd.weight")?,
        };
        let lm_head = QMatMul::from_weights(lm_head_tensor.into())?;

        let cached_masks = if cache_masks {
            Some(std::collections::HashMap::new())
        } else {
            None
        };

        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype,
            cache_masks,
            cached_masks,
            span,
            span_output,
        })
    }

    fn causal_mask(&mut self, b: usize, tgt: usize, offset: usize) -> Result<Tensor> {
        // Check cache if enabled
        if self.cache_masks {
            let key = (tgt, offset);
            if let Some(cached_masks) = &self.cached_masks {
                if let Some(mask) = cached_masks.get(&key) {
                    return if b == 1 {
                        Ok(mask.clone())
                    } else {
                        mask.broadcast_left(b)
                    };
                }
            }
        }

        // Create mask
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| {
                (0..(tgt + offset)).map(move |j| {
                    if j <= i + offset {
                        0.0f32
                    } else {
                        minf
                    }
                })
            })
            .collect();

        let mask_tensor = Tensor::from_slice(
            &mask,
            (1, 1, tgt, tgt + offset),
            &self.device,
        )?
        .to_dtype(self.dtype)?;

        // Cache if enabled
        if self.cache_masks {
            if let Some(cached_masks) = &mut self.cached_masks {
                cached_masks.insert((tgt, offset), mask_tensor.clone());
            }
        }

        if b == 1 {
            Ok(mask_tensor)
        } else {
            mask_tensor.broadcast_left(b)
        }
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {

        let (b, l) = input.dims2()?;
        let mut h = {
            let _enter = self.span.enter();
            self.embed_tokens.forward(input)?
        };

        let causal_mask = if l == 1 {
            None
        } else {
            // Only create mask if NOT using SIMD flash attention
            #[cfg(all(not(feature = "flash-attn"), feature = "simd-flash-attn"))]
            {
                None  // SIMD flash attention uses loop bounds, no mask needed
            }
            #[cfg(not(all(not(feature = "flash-attn"), feature = "simd-flash-attn")))]
            {
                Some(self.causal_mask(b, l, offset)?)
            }
        };

        for layer in &mut self.layers {
            h = layer.forward(&h, causal_mask.as_ref(), offset)?;
        }
        let h = self.norm.forward(&h)?;
        let _enter = self.span_output.enter();
        let last_hidden = h.narrow(1, l - 1, 1)?;
        self.lm_head.forward(&last_hidden)?.squeeze(1)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.self_attn.kv_cache.reset();
        }
    }
}
