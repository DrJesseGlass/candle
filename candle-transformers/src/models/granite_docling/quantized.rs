//! Quantized Granite-Docling loader for GGUF text decoder + vision mmproj files.
//!
//! The text decoder mirrors the optimized CPU path of `quantized_qwen3`:
//! f16 interleaved raw KV cache + fused flash decode kernel, flash prefill,
//! fused interleaved RoPE for single-token decode, per-row quantized embedding
//! lookup, and a tied quantized lm_head evaluated only at the last position.
//! Granite differs from Qwen3 in having no per-head q/k norms and using the
//! interleaved (llama.cpp-style) RoPE convention rather than neox.

use super::config::{QuantizedVisionConfig, TextConfig};
use super::{merge_image_tokens, pixel_shuffle};
use crate::models::quantized_siglip;
use crate::models::with_tracing::QMatMul;
use crate::quantized_nn::{self, Linear, RmsNorm};
use crate::quantized_var_builder::VarBuilder;
use crate::utils::repeat_kv;
use candle::quantized::QTensor;
use candle::{DType, Device, Module, Result, Storage, Tensor};
use candle_nn::attention::cpu_flash::causal::causal_decode_f16kv_interleaved;
use candle_nn::attention::{flash_attn, AttnMask};
use candle_nn::kv_cache::{ConcatKvCache, InterleavedKvCache, RawInterleavedKvCacheF16};
use std::sync::Arc;

#[derive(Debug, Clone)]
struct Connector {
    modality_projection: Linear,
    scale_factor: usize,
}

impl Connector {
    fn new(cfg: &QuantizedVisionConfig, vb: VarBuilder) -> Result<Self> {
        let input_dim = cfg.connector_input_dim();
        let output_dim = cfg.projection_dim;
        let modality_projection =
            quantized_nn::linear_no_bias(input_dim, output_dim, vb.pp("model.fc"))?;
        Ok(Self {
            modality_projection,
            scale_factor: cfg.scale_factor,
        })
    }
}

impl Module for Connector {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = pixel_shuffle(xs, self.scale_factor)?;
        xs.apply(&self.modality_projection)
    }
}

#[derive(Clone, Debug)]
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    /// Pre-extracted flat f32 cos/sin for the fused decode path (zero alloc).
    cos_f32: Vec<f32>,
    sin_f32: Vec<f32>,
    half_d: usize,
}

impl RotaryEmbedding {
    fn new(cfg: &TextConfig, dev: &Device) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let max_seq = cfg.max_position_embeddings;
        let theta = cfg.rope_theta;

        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / (theta as f32).powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), dev)?;
        let positions = Tensor::arange(0u32, max_seq as u32, dev)?.to_dtype(DType::F32)?;
        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        let cos_f32 = cos.flatten_all()?.to_vec1::<f32>()?;
        let sin_f32 = sin.flatten_all()?.to_vec1::<f32>()?;
        Ok(Self {
            cos,
            sin,
            cos_f32,
            sin_f32,
            half_d: head_dim / 2,
        })
    }

    /// Apply interleaved RoPE (q, k shape: B x H x L x D).
    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        // CPU f32 decode fast path: fused interleaved rope on raw slices,
        // bit-identical to the op path. Prefill keeps the op path, which
        // parallelizes over t.
        if seq_len == 1 && q.device().is_cpu() && q.dtype() == DType::F32 {
            return Ok((self.rope_i_f32(q, offset)?, self.rope_i_f32(k, offset)?));
        }
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        // GGUF weights use llama.cpp's interleaved RoPE convention.
        let q_embed = candle_nn::rotary_emb::rope_i(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope_i(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }

    // Fused interleaved RoPE on a CPU f32 tensor (B x H x L x D), matching
    // candle_nn::rotary_emb::rope_i (same op order) on raw slices, no apply_op3.
    fn rope_i_f32(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, h, t, d) = x.dims4()?;
        let half = d / 2;
        if half == 0 || 2 * half != d {
            candle::bail!("rope head dim {d} must be a positive even number");
        }
        if half != self.half_d {
            candle::bail!(
                "rope head dim {d} (half {half}) does not match table half_d {}",
                self.half_d
            );
        }
        let max_pos = self.cos_f32.len() / self.half_d;
        if offset + t > max_pos {
            candle::bail!("rope position {} exceeds max {max_pos}", offset + t);
        }
        let xc = x.contiguous()?;
        let (storage, layout) = xc.storage_and_layout();
        let src: &[f32] = match &*storage {
            Storage::Cpu(c) => &c.as_slice::<f32>()?[layout.start_offset()..],
            _ => candle::bail!("rope_i_f32: expected CPU storage"),
        };
        let mut dst = vec![0f32; b * h * t * d];
        for bh in 0..b * h {
            let chunk = bh * t * d;
            for it in 0..t {
                let start = (offset + it) * self.half_d;
                let cos = &self.cos_f32[start..start + self.half_d];
                let sin = &self.sin_f32[start..start + self.half_d];
                let tb = chunk + it * d;
                for j in 0..half {
                    let a = src[tb + 2 * j];
                    let bb = src[tb + 2 * j + 1];
                    dst[tb + 2 * j] = a * cos[j] - bb * sin[j];
                    dst[tb + 2 * j + 1] = a * sin[j] + bb * cos[j];
                }
            }
        }
        Tensor::from_vec(dst, (b, h, t, d), x.device())
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary: Arc<RotaryEmbedding>,
    // CPU: interleaved + raw f16 caches for the flash kernels.
    // Non-CPU: standard concat KV cache fallback.
    kv_cache: Option<ConcatKvCache>,
    interleaved_cache: Option<InterleavedKvCache>,
    raw_cache_f16: Option<RawInterleavedKvCacheF16>,
}

impl Attention {
    fn new(cfg: &TextConfig, rotary: Arc<RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;

        let q_proj = quantized_nn::linear_no_bias(h, num_heads * head_dim, vb.pp("attn_q"))?;
        let k_proj = quantized_nn::linear_no_bias(h, num_kv_heads * head_dim, vb.pp("attn_k"))?;
        let v_proj = quantized_nn::linear_no_bias(h, num_kv_heads * head_dim, vb.pp("attn_v"))?;
        let o_proj = quantized_nn::linear_no_bias(num_heads * head_dim, h, vb.pp("attn_output"))?;

        let on_cpu = vb.device().is_cpu();
        let kv_cache = if on_cpu {
            None
        } else {
            Some(ConcatKvCache::new(2))
        };
        let interleaved_cache = if on_cpu {
            Some(InterleavedKvCache::new(head_dim))
        } else {
            None
        };
        let raw_cache_f16 = if on_cpu {
            Some(RawInterleavedKvCacheF16::new(num_kv_heads, head_dim, 4096))
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            hidden_size: num_heads * head_dim,
            rotary,
            kv_cache,
            interleaved_cache,
            raw_cache_f16,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attn_mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let (b, l, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = self.rotary.apply(&q, &k, offset)?;

        if xs.device().is_cpu() && b == 1 {
            let scale = 1.0 / (self.head_dim as f32).sqrt();

            if l == 1 && q.dtype() == DType::F32 {
                // Fused decode: raw slices -> raw f16 cache -> flash kernel.
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

                let k_len = self.num_kv_heads * self.head_dim;
                let q_len = self.num_heads * self.head_dim;
                let rc = self.raw_cache_f16.as_mut().unwrap();
                rc.write_kv(&k_data[..k_len], &v_data[..k_len]);
                let ctx = causal_decode_f16kv_interleaved(
                    &q_data[..q_len],
                    rc.data(),
                    rc.head_stride(),
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    rc.len(),
                    scale,
                )?;

                ctx.reshape((b, l, self.hidden_size))?.apply(&self.o_proj)
            } else {
                // Prefill: interleaved cache + flash_attn; also populate the raw
                // f16 cache for subsequent decode steps.
                let ic = self.interleaved_cache.as_mut().unwrap();
                let kv = ic.append(&k, &v)?;

                {
                    let k_cont = k.squeeze(0)?.transpose(0, 1)?.contiguous()?;
                    let v_cont = v.squeeze(0)?.transpose(0, 1)?.contiguous()?;
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
                    self.raw_cache_f16
                        .as_mut()
                        .unwrap()
                        .write_kv_batch(k_d, v_d, l);
                }

                let kv_k = kv.narrow(2, 0, self.head_dim)?.unsqueeze(0)?;
                let kv_v = kv.narrow(2, self.head_dim, self.head_dim)?.unsqueeze(0)?;

                let q = q.transpose(1, 2)?.contiguous()?;
                let k = kv_k.contiguous()?;
                let v = kv_v.contiguous()?;

                let ctx = flash_attn::<f32>(
                    &q,
                    &k,
                    &v,
                    scale,
                    AttnMask::causal_with_offset(offset),
                    None,
                    None,
                )?;
                let ctx = ctx.transpose(1, 2)?;
                ctx.reshape((b, l, self.hidden_size))?.apply(&self.o_proj)
            }
        } else {
            // Standard matmul attention (non-CPU or batched fallback).
            let (k, v) = self.kv_cache.as_mut().unwrap().append(&k, &v)?;

            let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
            let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

            let scale = 1.0 / (self.head_dim as f64).sqrt();
            let mut scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?)? * scale)?;
            if let Some(m) = attn_mask {
                scores = scores.broadcast_add(m)?;
            }
            let probs = candle_nn::ops::softmax_last_dim(&scores)?;
            let ctx = probs.matmul(&v)?;
            ctx.transpose(1, 2)?
                .reshape((b, l, self.hidden_size))?
                .apply(&self.o_proj)
        }
    }

    fn clear_kv_cache(&mut self) {
        if let Some(c) = &mut self.kv_cache {
            c.reset();
        }
        if let Some(c) = &mut self.interleaved_cache {
            c.reset();
        }
        if let Some(c) = &mut self.raw_cache_f16 {
            c.reset();
        }
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let gate_proj = quantized_nn::linear_no_bias(h, i, vb.pp("ffn_gate"))?;
        let up_proj = quantized_nn::linear_no_bias(h, i, vb.pp("ffn_up"))?;
        let down_proj = quantized_nn::linear_no_bias(i, h, vb.pp("ffn_down"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self
            .gate_proj
            .forward(xs)?
            .apply(&candle_nn::Activation::Silu)?;
        let up = self.up_proj.forward(xs)?;
        (gate * up)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &TextConfig, rotary: Arc<RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(cfg, rotary, vb.clone())?;
        let mlp = Mlp::new(cfg, vb.clone())?;
        let input_layernorm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("attn_norm"))?;
        let post_attention_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("ffn_norm"))?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, mask, offset)?;
        let xs = (residual + xs)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        residual + xs
    }
}

#[derive(Debug, Clone)]
struct TextModel {
    // Kept quantized; only the input-token rows are dequantized per forward,
    // instead of materializing the full f32 vocab table at load.
    embed_tokens: Arc<QTensor>,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    // Tied lm_head sharing the quantized embedding tensor.
    lm_head: QMatMul,
    // Number of positions already in the KV caches.
    pos: usize,
    device: Device,
}

impl TextModel {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = vb.get((cfg.vocab_size, cfg.hidden_size), "token_embd.weight")?;
        let lm_head = QMatMul::from_weights(embed_tokens.clone())?;

        let rotary = Arc::new(RotaryEmbedding::new(cfg, vb.device())?);
        let vb_layers = vb.pp("blk");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, rotary.clone(), vb_layers.pp(i))?);
        }

        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("output_norm"))?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            pos: 0,
            device: vb.device().clone(),
        })
    }

    fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.embedding(input_ids)
    }

    fn causal_mask(&self, seq_len: usize, past_kv_len: usize) -> Result<Tensor> {
        let total_len = past_kv_len + seq_len;
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..total_len).map(move |j| {
                    if j > past_kv_len + i {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        Tensor::from_vec(mask, (1, 1, seq_len, total_len), &self.device)
    }

    fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let xs = self.embed(input_ids)?;
        self.forward_embeds(&xs)
    }

    /// Runs the decoder over pre-computed input embeddings and returns the
    /// logits for the LAST position only, shape (b, 1, vocab).
    fn forward_embeds(&mut self, xs: &Tensor) -> Result<Tensor> {
        let (_, seq_len, _) = xs.dims3()?;
        let offset = self.pos;

        // CPU flash paths handle causality internally; the mask is only needed
        // by the fallback matmul attention.
        let mask = if seq_len == 1 || self.device.is_cpu() {
            None
        } else {
            Some(self.causal_mask(seq_len, offset)?)
        };

        let mut hidden = xs.clone();
        for layer in self.layers.iter_mut() {
            hidden = layer.forward(&hidden, mask.as_ref(), offset)?;
        }
        self.pos += seq_len;

        let hidden = self.norm.forward(&hidden)?;
        let last_hidden = hidden.narrow(1, seq_len - 1, 1)?;
        self.lm_head.forward(&last_hidden)
    }

    fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.self_attn.clear_kv_cache();
        }
        self.pos = 0;
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    vision_model: quantized_siglip::VisionModel,
    connector: Connector,
    text_model: TextModel,
    image_token_id: u32,
}

impl Model {
    pub fn new(
        vision_vb: VarBuilder,
        vision_cfg: &QuantizedVisionConfig,
        text_vb: VarBuilder,
        text_cfg: &TextConfig,
        image_token_id: u32,
    ) -> Result<Self> {
        let vision_model = quantized_siglip::VisionModel::new(vision_cfg, vision_vb.pp("v"))?;

        let connector = Connector::new(vision_cfg, vision_vb.pp("mm"))?;
        let text_model = TextModel::new(text_cfg, text_vb)?;

        Ok(Self {
            vision_model,
            connector,
            text_model,
            image_token_id,
        })
    }

    pub fn encode_image(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let vision_out = self.vision_model.forward(pixel_values)?;
        let connected = self.connector.forward(&vision_out)?;
        let (n, seq, hidden) = connected.dims3()?;
        connected.reshape((1, n * seq, hidden))
    }

    /// Image + prompt prefill. Returns last-position logits, shape (1, 1, vocab).
    pub fn setup(&mut self, pixel_values: &Tensor, input_ids: &Tensor) -> Result<Tensor> {
        self.text_model.clear_kv_cache();
        let image_features = self.encode_image(pixel_values)?;
        self.prefill_with_image_features(&image_features, input_ids)
    }

    /// The prefill half of [`Model::setup`], for callers that time the vision
    /// encode separately: merge pre-computed image features into the prompt
    /// embeddings and run the text prefill. Does NOT clear the KV cache.
    /// Returns last-position logits, shape (1, 1, vocab).
    pub fn prefill_with_image_features(
        &mut self,
        image_features: &Tensor,
        input_ids: &Tensor,
    ) -> Result<Tensor> {
        let text_embeds = self.text_model.embed(input_ids)?;
        let input_embeds = merge_image_tokens(
            &text_embeds,
            image_features,
            input_ids,
            self.image_token_id,
        )?;
        self.text_model.forward_embeds(&input_embeds)
    }

    /// Text-only forward continuing from the KV cache. Returns last-position
    /// logits, shape (1, 1, vocab).
    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        self.text_model.forward(input_ids)
    }

    pub fn clear_kv_cache(&mut self) {
        self.text_model.clear_kv_cache();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The fused decode rope must match the rope_i op path exactly (same op
    // order), since decode takes the fused path and prefill the op path.
    #[test]
    fn fused_rope_i_matches_op_path() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = TextConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 8,
            hidden_act: candle_nn::Activation::Silu,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            attention_bias: false,
            mlp_bias: false,
            tie_word_embeddings: true,
        };
        let rot = RotaryEmbedding::new(&cfg, &dev)?;
        for offset in [0usize, 1, 7, 63] {
            let x = Tensor::rand(-1.0f32, 1.0, (1, 2, 1, 8), &dev)?;
            let fused = rot.rope_i_f32(&x, offset)?;
            let cos = rot.cos.narrow(0, offset, 1)?;
            let sin = rot.sin.narrow(0, offset, 1)?;
            let op = candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)?;
            let a = fused.flatten_all()?.to_vec1::<f32>()?;
            let b = op.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(b.iter()) {
                assert_eq!(x.to_bits(), y.to_bits(), "offset {offset}");
            }
        }
        Ok(())
    }
}
