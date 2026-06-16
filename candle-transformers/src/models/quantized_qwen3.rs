//! Qwen3 implementation with quantization support.
//!
//! Based on the Qwen3 architecture and implemented with quantized weights
//! for reduced memory usage and faster inference on compatible hardware.
//!
//! References:
//! - [Qwen3 Models](https://huggingface.co/Qwen/Qwen3-0.6B) (architecture based on official implementations)
//!
use super::with_tracing::QMatMul;
use crate::quantized_nn::{EmbedTokens, QuantizedEmbedding, RmsNorm};
use crate::utils::repeat_kv;
use candle::quantized::{gguf_file, QTensor};
use candle::{DType, Device, Result, Storage, Tensor, D};
use candle_nn::attention::cpu_flash::causal::{
    causal_decode_f16kv_interleaved, causal_prefill_f16kv_headmajor,
};
use candle_nn::kv_cache::{ConcatKvCache, RawInterleavedKvCache};
use candle_nn::{Activation, Embedding, Module};
use std::io::{Read, Seek};
use std::sync::Arc;

// --- coarse prefill "rest" profiling (set CANDLE_MATMUL_PROFILE=1) ---------
// Non-matmul forward-pass buckets, to see what dominates (and scales) outside the
// quantized matmuls. Whatever's left of prefill wall after matmul + these is "glue"
// (residuals, reshapes, allocations).
pub static FLASH_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static NORM_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static ROPE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static COPY_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MODEL_PROFILE: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var("CANDLE_MATMUL_PROFILE").is_ok());

/// Time `f`, adding the elapsed ns to `c` when profiling is on.
#[inline]
fn timed<T>(c: &std::sync::atomic::AtomicU64, f: impl FnOnce() -> Result<T>) -> Result<T> {
    if *MODEL_PROFILE {
        let t = std::time::Instant::now();
        let r = f();
        c.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        r
    } else {
        f()
    }
}

/// Reset the model profiling buckets (call before a measured prefill).
pub fn model_profile_reset() {
    use std::sync::atomic::Ordering::Relaxed;
    FLASH_NS.store(0, Relaxed);
    NORM_NS.store(0, Relaxed);
    ROPE_NS.store(0, Relaxed);
    COPY_NS.store(0, Relaxed);
}

pub struct Gguf<R: Read + Seek> {
    ct: gguf_file::Content,
    reader: R,
    device: Device,
}

impl<R: Read + Seek> Gguf<R> {
    pub fn new(ct: gguf_file::Content, reader: R, device: Device) -> Self {
        Self { ct, reader, device }
    }

    pub fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        QMatMul::from_weights(ws.into())
    }

    pub fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        RmsNorm::from_qtensor(ws, eps)
    }

    pub fn metadata(&self) -> &std::collections::HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }

    pub fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(&mut self.reader, name, &self.device)
    }
}

/// Two projections of the same input, fused into one quantized matmul when their
/// weights share a dtype. Rows are quantized independently in every ggml format, so
/// the fusion is a bit-exact row concatenation; it halves the dispatch / fork-join
/// cost and shares one activation quantization. CPU only; other devices keep the
/// split projections.
#[derive(Debug, Clone)]
enum FusedPairProj {
    Fused { proj: QMatMul, n1: usize, n2: usize },
    Split(QMatMul, QMatMul),
}

impl FusedPairProj {
    fn load<R: Read + Seek>(gg: &mut Gguf<R>, name1: &str, name2: &str) -> Result<Self> {
        let t1 = gg.tensor(name1)?;
        let t2 = gg.tensor(name2)?;
        let (n1, k1) = t1.shape().dims2()?;
        let (n2, k2) = t2.shape().dims2()?;
        if gg.device.is_cpu() && t1.dtype() == t2.dtype() && k1 == k2 {
            let fused = QTensor::cat_rows(&[&t1, &t2])?;
            Ok(Self::Fused {
                proj: QMatMul::from_weights(fused.into())?,
                n1,
                n2,
            })
        } else {
            Ok(Self::Split(
                QMatMul::from_weights(t1.into())?,
                QMatMul::from_weights(t2.into())?,
            ))
        }
    }

    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Fused { proj, n1, n2 } => {
                let y = proj.forward(x)?;
                Ok((
                    y.narrow(D::Minus1, 0, *n1)?.contiguous()?,
                    y.narrow(D::Minus1, *n1, *n2)?.contiguous()?,
                ))
            }
            Self::Split(p1, p2) => Ok((p1.forward(x)?, p2.forward(x)?)),
        }
    }
}

#[derive(Debug, Clone)]
struct MlpWeights {
    gate_up: FusedPairProj,
    down_proj: QMatMul,
    act_fn: Activation,
    span: tracing::Span,
}

impl MlpWeights {
    fn new<R: Read + Seek>(gg: &mut Gguf<R>, prefix: &str) -> Result<Self> {
        let gate_up = FusedPairProj::load(
            gg,
            &format!("{prefix}.ffn_gate.weight"),
            &format!("{prefix}.ffn_up.weight"),
        )?;
        let down_proj = gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?;
        let act_fn = Activation::Silu;
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        Ok(Self {
            gate_up,
            down_proj,
            act_fn,
            span,
        })
    }
}

impl Module for MlpWeights {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (gate, up) = self.gate_up.forward(x)?;
        let gate = gate.apply(&self.act_fn)?;
        let gated = (gate * up)?;
        self.down_proj.forward(&gated)
    }
}

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    /// Pre-extracted flat f32 cos/sin for fused decode (zero allocation)
    cos_f32: Vec<f32>,
    sin_f32: Vec<f32>,
    half_d: usize,
}

impl RotaryEmbedding {
    pub fn new(
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
        let sin_t = freqs.sin()?;
        let cos_t = freqs.cos()?;
        let cos_f32 = cos_t
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let sin_f32 = sin_t
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok(Self {
            sin: sin_t,
            cos: cos_t,
            cos_f32,
            sin_f32,
            half_d: dim / 2,
        })
    }

    /// Apply RoPE (q, k shape: B x H x L x D)
    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }

    /// Zero-allocation cos/sin slices for a single position.
    #[inline]
    pub fn cos_sin_at(&self, pos: usize) -> (&[f32], &[f32]) {
        let start = pos * self.half_d;
        let end = start + self.half_d;
        (&self.cos_f32[start..end], &self.sin_f32[start..end])
    }
}

#[derive(Debug, Clone)]
struct AttentionWeights {
    qk_proj: FusedPairProj,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<ConcatKvCache>,
    raw_cache: Option<RawInterleavedKvCache>,
    span_attn: tracing::Span,
}

impl AttentionWeights {
    #[allow(clippy::too_many_arguments)]
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary_emb: Arc<RotaryEmbedding>,
        device: &Device,
        prefix: &str,
    ) -> Result<Self> {
        let num_kv_groups = num_heads / num_kv_heads;
        let hidden_size = num_heads * head_dim;

        // attn_v is often promoted to a wider quant (e.g. Q6_K in *_M files), so only
        // q/k — which share a dtype — are fused; v stays separate.
        let qk_proj = FusedPairProj::load(
            gg,
            &format!("{prefix}.attn_q.weight"),
            &format!("{prefix}.attn_k.weight"),
        )?;
        let v_proj = gg.qmatmul(&format!("{prefix}.attn_v.weight"))?;
        let o_proj = gg.qmatmul(&format!("{prefix}.attn_output.weight"))?;

        let q_norm = gg.rms_norm(&format!("{prefix}.attn_q_norm.weight"), rms_norm_eps)?;
        let k_norm = gg.rms_norm(&format!("{prefix}.attn_k_norm.weight"), rms_norm_eps)?;

        // CPU: use interleaved + raw caches for flash attention
        // GPU: use standard concat KV cache (fallback path)
        let on_cpu = device.is_cpu();
        let kv_cache = if on_cpu {
            None
        } else {
            Some(ConcatKvCache::new(2))
        };
        let raw_cache = if on_cpu {
            // Modest preallocation; the cache doubles itself when a longer
            // context needs it, and 36-layer models pay this per layer.
            Some(RawInterleavedKvCache::new(num_kv_heads, head_dim, 1024))
        } else {
            None
        };

        let span_attn = tracing::span!(tracing::Level::TRACE, "attn");

        Ok(Self {
            qk_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size,
            rotary_emb,
            kv_cache,
            raw_cache,
            span_attn,
        })
    }

    fn forward(&mut self, x: &Tensor, attn_mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b, l, _) = x.dims3()?;

        // QKV projections (q/k in one fused matmul when dtypes match)
        let (q, k) = self.qk_proj.forward(x)?;
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

        // Per-head Q/K norms (must stay as tensor ops)
        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q_flat = timed(&NORM_NS, || self.q_norm.forward(&q_flat))?;
        let k_flat = timed(&NORM_NS, || self.k_norm.forward(&k_flat))?;
        let q = q_flat.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        // RoPE
        let (q, k) = timed(&ROPE_NS, || self.rotary_emb.apply(&q, &k, offset))?;

        // TODO: b > 1 needs varlen CPU flash with interleaved cache support.
        if x.device().is_cpu() && b == 1 {
            let scale = 1.0 / (self.head_dim as f32).sqrt();

            if l == 1 && b == 1 && q.dtype() == DType::F32 {
                // Fused decode: raw slices -> raw cache -> kernel.
                let q_cont = q.squeeze(0)?.squeeze(1)?.contiguous()?;
                let (q_g, q_l) = q_cont.storage_and_layout();
                let q_data: &[f32] = match &*q_g {
                    Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[q_l.start_offset()..],
                    _ => candle::bail!("Expected CPU storage"),
                };

                let k_cont = k.squeeze(0)?.squeeze(1)?.contiguous()?;
                let (k_g, k_l) = k_cont.storage_and_layout();
                let k_data: &[f32] = match &*k_g {
                    Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[k_l.start_offset()..],
                    _ => candle::bail!("Expected CPU storage"),
                };

                let v_cont = v.squeeze(0)?.squeeze(1)?.contiguous()?;
                let (v_g, v_l) = v_cont.storage_and_layout();
                let v_data: &[f32] = match &*v_g {
                    Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[v_l.start_offset()..],
                    _ => candle::bail!("Expected CPU storage"),
                };

                // Write K, V into raw cache (no tensor allocation)
                let k_len = self.num_kv_heads * self.head_dim;
                let rc = self.raw_cache.as_mut().unwrap();
                rc.write_kv(&k_data[..k_len], &v_data[..k_len]);

                // Run interleaved decode kernel
                let kv_len = rc.len();
                let q_len = self.num_heads * self.head_dim;
                let ctx = timed(&FLASH_NS, || {
                    let _flash = tracing::span!(tracing::Level::TRACE, "flash").entered();
                    causal_decode_f16kv_interleaved(
                        &q_data[..q_len],
                        rc.data(),
                        rc.head_stride(),
                        self.num_heads,
                        self.num_kv_heads,
                        self.head_dim,
                        kv_len,
                        scale,
                    )
                })?;

                let ctx = ctx.unsqueeze(0)?.transpose(1, 2)?;
                ctx.reshape((b, l, self.hidden_size))?.apply(&self.o_proj)
            } else {
                // Prefill: write the f16 raw cache, then run causal flash directly
                // over it — the same cache decode reads, so there is no separate
                // f32 KV copy and no per-layer cat of the full cache.
                let k_cont = timed(&COPY_NS, || k.squeeze(0)?.transpose(0, 1)?.contiguous())?;
                let v_cont = timed(&COPY_NS, || v.squeeze(0)?.transpose(0, 1)?.contiguous())?;
                let (kg, kl) = k_cont.storage_and_layout();
                let k_d: &[f32] = match &*kg {
                    Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[kl.start_offset()..],
                    _ => candle::bail!("Expected CPU"),
                };
                let (vg, vl) = v_cont.storage_and_layout();
                let v_d: &[f32] = match &*vg {
                    Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[vl.start_offset()..],
                    _ => candle::bail!("Expected CPU"),
                };
                let rc = self.raw_cache.as_mut().unwrap();
                rc.write_kv_batch(k_d, v_d, l);
                let kv_len = rc.len();

                let q_t = timed(&COPY_NS, || q.transpose(1, 2)?.contiguous())?; // (b, l, h_q, d)
                let (qg, ql) = q_t.storage_and_layout();
                let q_d: &[f32] = match &*qg {
                    Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[ql.start_offset()..],
                    _ => candle::bail!("Expected CPU"),
                };

                let ctx = timed(&FLASH_NS, || {
                    let _flash = tracing::span!(tracing::Level::TRACE, "flash").entered();
                    causal_prefill_f16kv_headmajor(
                        &q_d[..l * self.num_heads * self.head_dim],
                        rc.data(),
                        rc.head_stride(),
                        l,
                        self.num_heads,
                        self.num_kv_heads,
                        self.head_dim,
                        kv_len,
                        scale,
                        offset,
                    )
                })?;
                let ctx = ctx.unsqueeze(0)?.transpose(1, 2)?;
                ctx.reshape((b, l, self.hidden_size))?.apply(&self.o_proj)
            }
        } else {
            // Standard matmul attention (no flash)
            let (k, v) = self.kv_cache.as_mut().unwrap().append(&k, &v)?;

            let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
            let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

            let scale = 1.0 / (self.head_dim as f64).sqrt();
            let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
            if let Some(m) = attn_mask {
                let scores_dtype = scores.dtype();
                let mask = if m.dtype() != scores_dtype {
                    m.to_dtype(scores_dtype)?
                } else {
                    m.clone()
                };
                scores = scores.broadcast_add(&mask)?;
            }
            let probs = candle_nn::ops::softmax_last_dim(&scores)?;
            let ctx = probs.matmul(&v)?;
            let reshaped_ctx = ctx.transpose(1, 2)?.reshape((b, l, self.hidden_size))?;
            self.o_proj.forward(&reshaped_ctx)
        }
    }

    fn clear_kv_cache(&mut self) {
        if let Some(c) = &mut self.kv_cache {
            c.reset();
        }
        if let Some(c) = &mut self.raw_cache {
            c.reset();
        }
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
    #[allow(clippy::too_many_arguments)]
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary: Arc<RotaryEmbedding>,
        device: &Device,
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
            rotary,
            device,
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
        let h = timed(&NORM_NS, || self.ln1.forward(x))?;
        let h = self.self_attn.forward(&h, mask, offset)?;
        let x = (x + h)?;
        let h2 = timed(&NORM_NS, || self.ln2.forward(&x))?;
        let h2 = h2.apply(&self.mlp)?;
        x + h2
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    embed_tokens: EmbedTokens,
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

        let dtype = match gg.metadata().get("general.dtype") {
            Some(v) => match v.to_u32() {
                Ok(0) => DType::F32,
                Ok(1) => DType::F16,
                _ => DType::F16,
            },
            None => DType::F16,
        };

        let embed_tensor = Arc::new(gg.tensor("token_embd.weight")?);
        let embed_tokens = if device.is_cpu() {
            EmbedTokens::Quantized(QuantizedEmbedding::from_arc(embed_tensor.clone())?)
        } else {
            EmbedTokens::Full(Embedding::new(
                embed_tensor.dequantize(device)?,
                hidden_size,
            ))
        };

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
                rotary.clone(),
                device,
                i,
            )?);
        }

        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;
        // Load output projection tensor, falling back to tied embeddings like gemma3;
        // the tied case shares the quantized bytes already held by embed_tokens.
        let lm_head = match gg.tensor("output.weight") {
            Ok(tensor) => QMatMul::from_weights(tensor.into())?,
            Err(_) => QMatMul::from_weights(embed_tensor.clone())?,
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
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;
        // Skip mask materialization only when the CPU flash path will actually
        // run (it requires b == 1); batched CPU falls back to standard attention,
        // which needs the causal mask.
        let causal_mask = if l == 1 || (self.device.is_cpu() && b == 1) {
            None
        } else {
            Some(self.causal_mask(b, l, offset, None)?)
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
            layer.clear_kv_cache();
        }
    }
}
