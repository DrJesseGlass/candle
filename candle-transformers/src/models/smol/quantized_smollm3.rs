use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::Activation;
use candle::quantized::{gguf_file, QMatMul};
use std::sync::Arc;
use std::io::{Read, Seek};

const MAX_SEQ_LEN: usize = 4096;

#[derive(Debug, Clone)]
pub struct QuantizedConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub rope_dimension_count: usize,
    pub no_rope_layer_interval: Option<usize>,
}

impl QuantizedConfig {
    /// Load config from GGUF metadata
    pub fn from_gguf(ct: &gguf_file::Content) -> Result<Self> {
        let metadata = &ct.metadata;

        // Helper to get required metadata
        let get_u32 = |key: &str| -> Result<usize> {
            metadata.get(key)
                .and_then(|v| v.to_u32().ok())
                .map(|v| v as usize)
                .ok_or_else(|| candle::Error::Msg(format!("Missing or invalid metadata key: {}", key)))
        };

        let get_f32 = |key: &str| -> Result<f64> {
            metadata.get(key)
                .and_then(|v| v.to_f32().ok())
                .map(|v| v as f64)
                .ok_or_else(|| candle::Error::Msg(format!("Missing or invalid metadata key: {}", key)))
        };

        Ok(Self {
            vocab_size: get_u32("smollm3.vocab_size")?,
            hidden_size: get_u32("smollm3.embedding_length")?,
            intermediate_size: get_u32("smollm3.feed_forward_length")?,
            num_hidden_layers: get_u32("smollm3.block_count")?,
            num_attention_heads: get_u32("smollm3.attention.head_count")?,
            num_key_value_heads: get_u32("smollm3.attention.head_count_kv")?,
            max_position_embeddings: get_u32("smollm3.context_length").unwrap_or(MAX_SEQ_LEN),
            rope_theta: get_f32("smollm3.rope.freq_base")?,
            rms_norm_eps: get_f32("smollm3.attention.layer_norm_rms_epsilon")?,
            rope_dimension_count: get_u32("smollm3.rope.dimension_count")?,
            // SmolLM3-3B uses interval=4 (every 4th layer skips RoPE)
            no_rope_layer_interval: Some(4),
        })
    }

    pub fn should_skip_rope(&self, layer_idx: usize) -> bool {
        if let Some(interval) = self.no_rope_layer_interval {
            return (layer_idx + 1) % interval == 0;
        }
        false
    }

    pub fn head_dim(&self) -> usize {
        self.rope_dimension_count
    }
}

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(candle::D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let norm_x = (x.sqr()?.sum_keepdim(candle::D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        // Multiply by weight (F32) BEFORE converting back to original dtype
        let result = x_normed.broadcast_mul(&self.weight)?;
        result.to_dtype(x_dtype)
    }
}

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dtype: DType, cfg: &QuantizedConfig, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim();
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    pub fn apply_rotary_emb(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(x)
    } else {
        let (b, n_kv_heads, seq_len, head_dim) = x.dims4()?;
        x.unsqueeze(2)?
            .expand(&[b, n_kv_heads, n_rep, seq_len, head_dim])?
            .reshape(&[b, n_kv_heads * n_rep, seq_len, head_dim])
    }
}

#[derive(Debug, Clone)]
struct QuantizedMLP {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
}

impl QuantizedMLP {
    fn new<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        layer_idx: usize,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");
        Ok(Self {
            gate_proj: QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?)?,
            up_proj: QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?)?,
            down_proj: QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.apply(&Activation::Silu)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

#[derive(Debug, Clone)]
struct QuantizedAttention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Option<Arc<RotaryEmbedding>>,
    skip_rope: bool,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl QuantizedAttention {
    fn new<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        cfg: &QuantizedConfig,
        layer_idx: usize,
        rotary_emb: Option<Arc<RotaryEmbedding>>,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");
        let head_dim = cfg.head_dim();
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;

        Ok(Self {
            q_proj: QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?)?,
            k_proj: QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?)?,
            v_proj: QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?)?,
            o_proj: QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?)?,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            rotary_emb,
            skip_rope: cfg.should_skip_rope(layer_idx),
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;

        // Project Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to (B, num_heads, seq_len, head_dim)
        let q = q
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply RoPE if this layer uses it (not NoPE)
        let (q, k) = if self.skip_rope {
            (q.contiguous()?, k.contiguous()?)
        } else {
            if let Some(ref rope) = self.rotary_emb {
                rope.apply_rotary_emb(&q, &k, offset)?
            } else {
                (q, k)
            }
        };

        // Update KV cache
        let (k, v) = if offset == 0 {
            // First token - initialize cache
            self.kv_cache = Some((k.clone(), v.clone()));
            (k, v)
        } else {
            // Subsequent tokens - concatenate with cache
            let (k_cache, v_cache) = self.kv_cache.as_ref().unwrap();
            let k = Tensor::cat(&[k_cache, &k], 2)?;
            let v = Tensor::cat(&[v_cache, &v], 2)?;
            self.kv_cache = Some((k.clone(), v.clone()));
            (k, v)
        };

        // Repeat KV for GQA
        let k = repeat_kv(k, self.num_kv_groups)?;
        let v = repeat_kv(v, self.num_kv_groups)?;

        // Attention computation
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        if let Some(mask) = mask {
            attn_weights = attn_weights.broadcast_add(mask)?;
        }

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Output projection
        attn_output
            .transpose(1, 2)?
            .reshape((b, seq_len, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

#[derive(Debug, Clone)]
struct QuantizedDecoderLayer {
    self_attn: QuantizedAttention,
    mlp: QuantizedMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl QuantizedDecoderLayer {
    fn new<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        cfg: &QuantizedConfig,
        layer_idx: usize,
        rotary_emb: Option<Arc<RotaryEmbedding>>,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");

        Ok(Self {
            self_attn: QuantizedAttention::new(ct, reader, cfg, layer_idx, rotary_emb, device)?,
            mlp: QuantizedMLP::new(ct, reader, layer_idx, device)?,
            input_layernorm: RmsNorm::new(
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?.dequantize(device)?,
                cfg.rms_norm_eps,
            ),
            post_attention_layernorm: RmsNorm::new(
                ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?.dequantize(device)?,
                cfg.rms_norm_eps,
            ),
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x, mask, offset)?;
        let x = (residual + x)?;

        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        residual + x
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct QuantizedModelForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<QuantizedDecoderLayer>,
    norm: RmsNorm,
    lm_head: QMatMul,
    device: Device,
    config: QuantizedConfig,
}

impl QuantizedModelForCausalLM {
    pub fn from_gguf<P: AsRef<std::path::Path>>(path: P, device: &Device) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file)?;

        let config = QuantizedConfig::from_gguf(&content)?;

        // Load embedding
        let embed_tokens = {
            let embed_tensor = content.tensor(&mut file, "token_embd.weight", device)?
                .dequantize(device)?;
            candle_nn::Embedding::new(embed_tensor, config.hidden_size)
        };

        // Create rotary embedding if needed
        let needs_rope = (0..config.num_hidden_layers)
            .any(|i| !config.should_skip_rope(i));
        let rotary_emb = if needs_rope {
            Some(Arc::new(RotaryEmbedding::new(
                DType::F32,
                &config,
                device,
            )?))
        } else {
            None
        };

        // Load decoder layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            layers.push(QuantizedDecoderLayer::new(
                &content,
                &mut file,
                &config,
                layer_idx,
                rotary_emb.clone(),
                device,
            )?);
        }

        // Load output norm
        let norm = RmsNorm::new(
            content.tensor(&mut file, "output_norm.weight", device)?.dequantize(device)?,
            config.rms_norm_eps,
        );

        // Load LM head
        let lm_head = QMatMul::from_qtensor(content.tensor(&mut file, "token_embd.weight", device)?)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            config,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, offset: usize) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;

        // Embed tokens
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        // Create causal mask if needed
        let mask = if seq_len > 1 {
            Some(self.create_causal_mask(batch_size, seq_len, offset)?)
        } else {
            None
        };

        // Forward through decoder layers
        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states, mask.as_ref(), offset)?;
        }

        // Final norm
        hidden_states = self.norm.forward(&hidden_states)?;

        // LM head (only last token for generation)
        let logits = hidden_states
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.lm_head)?;

        Ok(logits)
    }

    fn create_causal_mask(
        &self,
        batch_size: usize,
        tgt_len: usize,
        offset: usize,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len + offset).map(move |j| {
                    if j <= i + offset {
                        0f32
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();

        Tensor::from_slice(
            &mask,
            (batch_size, 1, tgt_len, tgt_len + offset),
            &self.device,
        )
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }

    pub fn config(&self) -> &QuantizedConfig {
        &self.config
    }
}