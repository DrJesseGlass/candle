//! Utilities for quanitized network layers
//!
//! This module contains various implementations of standard neural network layers, modules and
//! utilities including embedding, linear layers, and various normalization techniques.
//! Most implementations provide quantized weights support.

use crate::models::with_tracing::QMatMul;
use crate::quantized_var_builder::VarBuilder;
use candle::quantized::QTensor;
use candle::{Module, Result, Tensor};

#[derive(Debug, Clone)]
pub struct Embedding {
    inner: candle_nn::Embedding,
    span: tracing::Span,
}

impl Embedding {
    pub fn new(d1: usize, d2: usize, vb: VarBuilder) -> Result<Self> {
        let embeddings = vb.get((d1, d2), "weight")?.dequantize(vb.device())?;
        let inner = candle_nn::Embedding::new(embeddings, d2);
        let span = tracing::span!(tracing::Level::TRACE, "embedding");
        Ok(Self { inner, span })
    }

    pub fn embeddings(&self) -> &Tensor {
        self.inner.embeddings()
    }
}

impl Module for Embedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

/// Token embedding that is either a dense f32 table (GPU) or a row-gather over
/// the quantized weight (CPU; see [`QuantizedEmbedding`]).
#[derive(Debug, Clone)]
pub enum EmbedTokens {
    Full(candle_nn::Embedding),
    Quantized(QuantizedEmbedding),
}

impl Module for EmbedTokens {
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Full(e) => e.forward(ids),
            Self::Quantized(e) => e.forward(ids),
        }
    }
}

/// Embedding lookup that keeps the weight quantized, dequantizing only the rows
/// each forward needs. A 128k-vocab f32 embedding table costs ~1 GB of RSS while
/// decode reads a single row per token; this keeps the quantized bytes (often
/// shared with a tied lm_head) as the only copy. CPU only.
#[derive(Debug, Clone)]
pub struct QuantizedEmbedding {
    weight: std::sync::Arc<QTensor>,
    row_bytes: usize,
    hidden: usize,
    span: tracing::Span,
}

impl QuantizedEmbedding {
    pub fn from_arc(weight: std::sync::Arc<QTensor>) -> Result<Self> {
        let (vocab, hidden) = weight.shape().dims2()?;
        let dtype = weight.dtype();
        if hidden % dtype.block_size() != 0 {
            candle::bail!("embedding dim {hidden} not divisible by block size")
        }
        let row_bytes = hidden / dtype.block_size() * dtype.type_size();
        debug_assert_eq!(weight.storage_size_in_bytes(), vocab * row_bytes);
        let span = tracing::span!(tracing::Level::TRACE, "q-embedding");
        Ok(Self {
            weight,
            row_bytes,
            hidden,
            span,
        })
    }
}

impl Module for QuantizedEmbedding {
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let dims = ids.dims().to_vec();
        // Normalize to U32 first: token-id tensors are often I64 (and U8 is also a valid
        // index dtype), all of which `index_select` accepts. Reading straight to_vec1::<u32>
        // would reject those and fail before lookup, unlike the dense embedding path.
        let ids = ids
            .flatten_all()?
            .to_dtype(candle::DType::U32)?
            .to_vec1::<u32>()?;
        // Gather the quantized rows, dequantize them as one small QTensor.
        let data = self.weight.data()?; // zero-copy borrow on CPU
        let mut rows = Vec::with_capacity(ids.len() * self.row_bytes);
        for &id in &ids {
            let off = id as usize * self.row_bytes;
            rows.extend_from_slice(&data[off..off + self.row_bytes]);
        }
        let storage = candle::quantized::QStorage::from_data(
            std::borrow::Cow::Owned(rows),
            &candle::Device::Cpu,
            self.weight.dtype(),
        )?;
        let gathered = QTensor::new(storage, (ids.len(), self.hidden))?;
        let emb = gathered.dequantize(&candle::Device::Cpu)?;
        let mut out_dims = dims;
        out_dims.push(self.hidden);
        emb.reshape(out_dims)
    }
}

#[derive(Debug, Clone)]
pub struct Linear {
    weight: QMatMul,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn from_arc(weight: std::sync::Arc<QTensor>, bias: Option<Tensor>) -> Result<Self> {
        let weight = QMatMul::from_weights(weight)?;
        Ok(Self { weight, bias })
    }

    pub fn from_weights(weight: QMatMul, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        let x = x.apply(&self.weight)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

pub fn linear_b(in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder) -> Result<Linear> {
    let bias = if bias {
        Some(vb.get(out_dim, "bias")?.dequantize(vb.device())?)
    } else {
        None
    };
    let weight = QMatMul::new(in_dim, out_dim, vb)?;
    Ok(Linear { weight, bias })
}

pub fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let bias = vb.get(out_dim, "bias")?.dequantize(vb.device())?;
    let weight = QMatMul::new(in_dim, out_dim, vb)?;
    Ok(Linear {
        weight,
        bias: Some(bias),
    })
}

pub fn layer_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<candle_nn::LayerNorm> {
    let weight = vb.get(size, "weight")?.dequantize(vb.device())?;
    let bias = vb.get(size, "bias")?.dequantize(vb.device())?;
    Ok(candle_nn::LayerNorm::new(weight, bias, eps))
}

pub fn layer_norm_no_bias(size: usize, eps: f64, vb: VarBuilder) -> Result<candle_nn::LayerNorm> {
    let weight = vb.get(size, "weight")?.dequantize(vb.device())?;
    Ok(candle_nn::LayerNorm::new_no_bias(weight, eps))
}

pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = QMatMul::new(in_dim, out_dim, vb)?;
    Ok(Linear { weight, bias: None })
}

#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
    span: tracing::Span,
}

impl RmsNorm {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let weight = vb.get(size, "weight")?.dequantize(vb.device())?;
        Ok(Self { weight, eps, span })
    }

    pub fn from_qtensor(weight: QTensor, eps: f64) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let weight = weight.dequantize(&weight.device())?;
        Ok(Self { weight, eps, span })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        candle_nn::ops::rms_norm(x, &self.weight, self.eps as f32)
    }
}
