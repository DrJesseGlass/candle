//! Gemma3 with layer-by-layer debug output.
//! Debug output format matches dump_layers.py for easy comparison.
//!
//! Enable debug with: DEBUG_LAYER=1 (or -1 for all layers)

use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear_b as linear, Activation, Linear, VarBuilder};
use std::sync::Arc;

// =============================================================================
// DEBUG CONFIGURATION
// =============================================================================

/// Set to true to enable debug output
const DEBUG_ENABLED: bool = false; // Disable prefill debug

/// Which layer to debug (-1 = all layers, 0 = layer 0 only, etc.)
const DEBUG_LAYER: i32 = 1; // Debug layers 0 and 1

/// Helper to print debug values in matching format
fn debug_print(name: &str, t: &Tensor, token_idx: usize) {
    if !DEBUG_ENABLED {
        return;
    }

    if let Ok(vals) = t
        .narrow(1, token_idx, 1)
        .and_then(|t| t.narrow(2, 0, 5))
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.to_vec1::<f32>())
    {
        let rounded: Vec<f32> = vals.iter().map(|v| (v * 1e6).round() / 1e6).collect();
        eprintln!("{:30}: {:?}", name, rounded);
    }
}

fn debug_print_4d(name: &str, t: &Tensor, head_idx: usize, token_idx: usize) {
    if !DEBUG_ENABLED {
        return;
    }

    // t is [batch, heads, seq, head_dim]
    if let Ok(vals) = t
        .narrow(1, head_idx, 1) // Select head
        .and_then(|t| t.narrow(2, token_idx, 1)) // Select token
        .and_then(|t| t.narrow(3, 0, 5)) // First 5 dims
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.to_vec1::<f32>())
    {
        let rounded: Vec<f32> = vals.iter().map(|v| (v * 1e6).round() / 1e6).collect();
        eprintln!("{:30}: {:?}", name, rounded);
    }
}

fn should_debug_layer(layer_idx: usize) -> bool {
    DEBUG_ENABLED && (DEBUG_LAYER < 0 || layer_idx as i32 <= DEBUG_LAYER)
}

// =============================================================================
// CONFIG
// =============================================================================

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    pub attention_bias: bool,
    pub head_dim: usize,
    pub hidden_activation: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_local_base_freq: f64,
    pub vocab_size: usize,
    pub final_logit_softcapping: Option<f64>,
    pub attn_logit_softcapping: Option<f64>,
    pub query_pre_attn_scalar: usize,
    pub sliding_window: usize,
    pub sliding_window_pattern: usize,
    pub max_position_embeddings: usize,
}

// =============================================================================
// RMS NORM (Gemma adds 1 to weight)
// =============================================================================

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        // Gemma: weight is (1 + weight), not just weight
        x_normed
            .to_dtype(x_dtype)?
            .broadcast_mul(&(&self.weight + 1.0)?)
    }
}

// =============================================================================
// ROTARY EMBEDDING
// =============================================================================

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        cfg: &Config,
        dev: &Device,
        sliding_window: Option<usize>,
    ) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let rope_freq = if sliding_window.is_some() {
            cfg.rope_local_base_freq
        } else {
            cfg.rope_theta
        };
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_freq.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// =============================================================================
// MLP
// =============================================================================

fn gelu_pytorch_tanh(x: &Tensor) -> Result<Tensor> {
    // GELU with tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    const SQRT_2_OVER_PI: f64 = 0.7978845608028654;
    const COEF: f64 = 0.044715;

    let x_cubed = x.powf(3.0)?;
    let inner = ((x + (&x_cubed * COEF)?)? * SQRT_2_OVER_PI)?;
    let tanh_val = inner.tanh()?;
    let one_plus_tanh = (&tanh_val + 1.0)?;
    x.mul(&one_plus_tanh)?.affine(0.5, 0.0)
}

#[derive(Debug, Clone)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear(hidden_sz, intermediate_sz, false, vb.pp("gate_proj"))?;
        let up_proj = linear(hidden_sz, intermediate_sz, false, vb.pp("up_proj"))?;
        let down_proj = linear(intermediate_sz, hidden_sz, false, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward_debug(&self, xs: &Tensor, layer_idx: usize) -> Result<Tensor> {
        let gate = xs.apply(&self.gate_proj)?;
        let up = xs.apply(&self.up_proj)?;

        if should_debug_layer(layer_idx) {
            debug_print(&format!("L{} gate_proj out", layer_idx), &gate, 0);
            debug_print(&format!("L{} up_proj out", layer_idx), &up, 0);
        }

        // Use GELU with tanh approximation
        let activated = gelu_pytorch_tanh(&gate)?;

        if should_debug_layer(layer_idx) {
            debug_print(&format!("L{} after gelu_tanh", layer_idx), &activated, 0);
        }

        let gate_up = (&activated * &up)?;

        if should_debug_layer(layer_idx) {
            debug_print(&format!("L{} gate * up", layer_idx), &gate_up, 0);
        }

        let out = gate_up.apply(&self.down_proj)?;

        if should_debug_layer(layer_idx) {
            debug_print(&format!("L{} down_proj (mlp out)", layer_idx), &out, 0);
        }

        Ok(out)
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = xs.apply(&self.gate_proj)?;
        let up = xs.apply(&self.up_proj)?;
        let activated = gelu_pytorch_tanh(&gate)?;
        (&activated * &up)?.apply(&self.down_proj)
    }
}

// =============================================================================
// KV CACHE
// =============================================================================

#[derive(Debug, Clone)]
enum KvCache {
    Normal(candle_nn::kv_cache::KvCache),
    Rotating(candle_nn::kv_cache::RotatingKvCache),
}

// =============================================================================
// ATTENTION
// =============================================================================

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    attn_logit_softcapping: Option<f64>,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: KvCache,
    use_flash_attn: bool,
}

impl Attention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        use_flash_attn: bool,
        cfg: &Config,
        sliding_window: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = cfg.head_dim;
        let bias = cfg.attention_bias;
        let q_proj = linear(hidden_sz, num_heads * head_dim, bias, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("v_proj"))?;
        let o_proj = linear(num_heads * head_dim, hidden_sz, bias, vb.pp("o_proj"))?;
        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
        let kv_cache = if let Some(sliding_window) = sliding_window {
            KvCache::Rotating(candle_nn::kv_cache::RotatingKvCache::new(2, sliding_window))
        } else {
            KvCache::Normal(candle_nn::kv_cache::KvCache::new(
                2,
                cfg.max_position_embeddings,
            ))
        };
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
            attn_logit_softcapping: cfg.attn_logit_softcapping,
            rotary_emb,
            kv_cache,
            use_flash_attn,
        })
    }

    fn forward_debug(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let debug = seqlen_offset == 0 && should_debug_layer(layer_idx);

        // Debug during generation for layer 0 only
        let gen_debug = seqlen_offset > 0 && seqlen_offset < 55 && layer_idx == 0;

        // Q, K, V projections
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        if debug {
            debug_print(&format!("L{} q_proj out", layer_idx), &q, 0);
            debug_print(&format!("L{} k_proj out", layer_idx), &k, 0);
            debug_print(&format!("L{} v_proj out", layer_idx), &v, 0);
        }

        // Reshape to [batch, heads, seq, head_dim]
        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Q, K norms
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        if debug {
            debug_print_4d(&format!("L{} q after norm", layer_idx), &q, 0, 0);
            debug_print_4d(&format!("L{} k after norm", layer_idx), &k, 0, 0);
        }

        if gen_debug {
            eprintln!(
                "=== GEN DEBUG L{}: seqlen_offset={}, q_len={} ===",
                layer_idx, seqlen_offset, q_len
            );
        }

        // RoPE
        let (q, k) = self
            .rotary_emb
            .apply_rotary_emb_qkv(&q, &k, seqlen_offset)?;

        if debug {
            debug_print_4d(&format!("L{} q after RoPE", layer_idx), &q, 0, 0);
            debug_print_4d(&format!("L{} k after RoPE", layer_idx), &k, 0, 0);
        }

        if gen_debug {
            debug_print_4d(
                &format!("GEN q after RoPE (pos={})", seqlen_offset),
                &q,
                0,
                0,
            );
        }

        let k = k.contiguous()?;
        let v = v.contiguous()?;

        // Get cache state before append
        let cache_len_before = match &self.kv_cache {
            KvCache::Normal(cache) => cache.current_seq_len(),
            KvCache::Rotating(cache) => cache.current_seq_len(),
        };

        // KV cache
        let (k, v) = match &mut self.kv_cache {
            KvCache::Normal(cache) => cache.append(&k, &v)?,
            KvCache::Rotating(cache) => cache.append(&k, &v)?,
        };

        if gen_debug {
            eprintln!(
                "GEN: cache {} -> {}, k shape {:?}",
                cache_len_before,
                k.dim(2)?,
                k.shape()
            );
        }

        // Expand K, V for GQA
        let k = crate::utils::repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = crate::utils::repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        // Attention computation
        let attn_output = if self.use_flash_attn {
            #[cfg(feature = "flash-attn")]
            {
                let q = q.transpose(1, 2)?;
                let k = k.transpose(1, 2)?;
                let v = v.transpose(1, 2)?;
                let scale = 1f32 / (self.head_dim as f32).sqrt();
                flash_attn(&q, &k, &v, scale, attention_mask.is_some())?.transpose(1, 2)?
            }
            #[cfg(not(feature = "flash-attn"))]
            candle::bail!("flash-attn feature not enabled")
        } else {
            let scale = 1f64 / (self.head_dim as f64).sqrt();
            let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

            // Softcapping
            let attn_weights = match self.attn_logit_softcapping {
                None => attn_weights,
                Some(sc) => ((attn_weights / sc)?.tanh()? * sc)?,
            };

            // Apply mask
            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask)?,
            };

            if gen_debug {
                eprintln!(
                    "GEN: attn_weights shape {:?}, mask={}",
                    attn_weights.shape(),
                    attention_mask.is_some()
                );
            }

            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&v)?
        };

        let out = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)?;

        if debug {
            debug_print(&format!("L{} self_attn out", layer_idx), &out, 0);
        }

        Ok(out)
    }

    fn clear_kv_cache(&mut self) {
        match &mut self.kv_cache {
            KvCache::Normal(c) => c.reset(),
            KvCache::Rotating(c) => c.reset(),
        }
    }
}

// =============================================================================
// DECODER LAYER
// =============================================================================

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    sliding_window: Option<usize>,
}

impl DecoderLayer {
    fn new(
        use_flash_attn: bool,
        cfg: &Config,
        vb: VarBuilder,
        sliding_window: Option<usize>,
    ) -> Result<Self> {
        let rotary_emb = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            cfg,
            vb.device(),
            sliding_window,
        )?);
        let self_attn = Attention::new(
            rotary_emb,
            use_flash_attn,
            cfg,
            sliding_window,
            vb.pp("self_attn"),
        )?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let pre_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            sliding_window,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let debug = seqlen_offset == 0 && should_debug_layer(layer_idx);

        // Input layernorm
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;

        if debug {
            debug_print(&format!("L{} input_layernorm", layer_idx), &xs, 0);
        }

        // Self attention
        let xs = self
            .self_attn
            .forward_debug(&xs, attention_mask, seqlen_offset, layer_idx)?;

        // Post attention layernorm
        let xs = self.post_attention_layernorm.forward(&xs)?;

        if debug {
            debug_print(&format!("L{} post_attn_layernorm", layer_idx), &xs, 0);
        }

        // Add residual
        let xs = (xs + residual)?;

        if debug {
            debug_print(&format!("L{} after attn residual", layer_idx), &xs, 0);
        }

        // Pre-feedforward layernorm
        let residual = &xs;
        let xs_mlp = self.pre_feedforward_layernorm.forward(&xs)?;

        if debug {
            debug_print(&format!("L{} pre_feedforward_ln", layer_idx), &xs_mlp, 0);
        }

        // MLP
        let xs_mlp = self.mlp.forward_debug(&xs_mlp, layer_idx)?;

        // Post feedforward layernorm
        let xs_mlp = self.post_feedforward_layernorm.forward(&xs_mlp)?;

        if debug {
            debug_print(&format!("L{} post_feedforward_ln", layer_idx), &xs_mlp, 0);
        }

        // Add residual
        let xs = (residual + xs_mlp)?;

        if debug {
            debug_print(&format!("L{} layer output", layer_idx), &xs, 0);
        }

        Ok(xs)
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

// =============================================================================
// PREPARE ATTENTION MASK
// =============================================================================

fn prepare_decoder_attention_mask(
    b_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
    sliding_window: Option<usize>,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mask: Vec<_> = if let Some(sliding_window) = sliding_window {
        (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect()
    } else {
        (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0f32 }))
            .collect()
    };
    let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), device)?;
    let mask = if seqlen_offset > 0 {
        let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, device)?;
        Tensor::cat(&[&mask0, &mask], D::Minus1)?
    } else {
        mask
    };
    mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
        .to_dtype(dtype)
}

// =============================================================================
// MODEL
// =============================================================================

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    final_logit_softcapping: Option<f64>,
    device: Device,
    dtype: DType,
    hidden_size: usize,
    sliding_window: usize,
}

impl Model {
    pub fn new(use_flash_attn: bool, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let sliding_window = (layer_idx + 1) % cfg.sliding_window_pattern > 0;
            let layer = DecoderLayer::new(
                use_flash_attn,
                cfg,
                vb_l.pp(layer_idx),
                sliding_window.then_some(cfg.sliding_window),
            )?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = Linear::new(embed_tokens.embeddings().clone(), None);

        if DEBUG_ENABLED {
            eprintln!(
                "Model loaded: {} layers, hidden_size={}",
                layers.len(),
                cfg.hidden_size
            );
        }

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            final_logit_softcapping: cfg.final_logit_softcapping,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            hidden_size: cfg.hidden_size,
            sliding_window: cfg.sliding_window,
        })
    }

    fn create_attention_masks(
        &self,
        batch_size: usize,
        seq_len: usize,
        seqlen_offset: usize,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        // DEBUG: Always create masks to see if this fixes the 50-token bug
        // Original code skipped mask creation when seq_len <= 1

        // Global attention mask (standard causal)
        let mask = prepare_decoder_attention_mask(
            batch_size,
            seq_len,
            seqlen_offset,
            None,
            self.dtype,
            &self.device,
        )?;

        // Sliding window attention mask
        let sliding_mask = prepare_decoder_attention_mask(
            batch_size,
            seq_len,
            seqlen_offset,
            Some(self.sliding_window),
            self.dtype,
            &self.device,
        )?;

        Ok((Some(mask), Some(sliding_mask)))
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;

        // Debug for prefill
        let prefill_debug = seqlen_offset == 0 && seq_len > 40;

        if prefill_debug {
            eprintln!("\n=== PREFILL DEBUG: {} tokens ===", seq_len);
        }

        // Embedding with scaling
        let xs = self.embed_tokens.forward(input_ids)?;
        let mut xs = (xs * (self.hidden_size as f64).sqrt())?;

        // Prepare attention masks
        let (attention_mask, sliding_attention_mask) =
            self.create_attention_masks(b_size, seq_len, seqlen_offset)?;

        // Process layers
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let mask = if layer.sliding_window.is_some() {
                &sliding_attention_mask
            } else {
                &attention_mask
            };
            xs = layer.forward(&xs, mask.as_ref(), seqlen_offset, layer_idx)?;
        }

        // Final norm (only on last token)
        let xs = xs.narrow(1, seq_len - 1, 1)?;
        let xs = self.norm.forward(&xs)?;

        if prefill_debug {
            // Show final hidden state before lm_head
            if let Ok(vals) = xs
                .narrow(2, 0, 5)
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_dtype(DType::F32))
                .and_then(|t| t.to_vec1::<f32>())
            {
                eprintln!("PREFILL final hidden[:5]: {:?}", vals);
            }
        }

        // LM head
        let logits = xs.apply(&self.lm_head)?;

        // Final softcapping
        let logits = match self.final_logit_softcapping {
            None => logits,
            Some(sc) => ((logits / sc)?.tanh()? * sc)?,
        };

        if prefill_debug {
            // Show top predictions after prefill
            if let Ok(logits_vec) = logits
                .squeeze(0)
                .and_then(|t| t.squeeze(0))
                .and_then(|t| t.to_vec1::<f32>())
            {
                let mut indexed: Vec<(usize, f32)> =
                    logits_vec.iter().cloned().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                eprintln!("PREFILL top 5 logits:");
                for i in 0..5.min(indexed.len()) {
                    eprintln!("  {} -> {:.2}", indexed[i].0, indexed[i].1);
                }
            }
            eprintln!("=== END PREFILL DEBUG ===\n");
        }

        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}
