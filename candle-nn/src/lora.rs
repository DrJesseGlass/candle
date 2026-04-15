//! Low-Rank Adaptation (LoRA) for linear layers.
//!
//! LoRA freezes a pretrained linear layer and learns a low-rank update
//! `B @ A` alongside it, where `A: (rank, in)` and `B: (out, rank)`. At
//! forward time the output is
//!
//! ```text
//! y = base(x) + (alpha / rank) * (B @ A @ x)
//! ```
//!
//! Only `A` and `B` are trainable; the base weights stay frozen. This lets
//! you fine-tune a large pretrained model by adding a small number of
//! parameters per layer (typically `rank ∈ {4, 8, 16}`).
//!
//! The canonical reference is
//! [Hu et al., 2021 — LoRA: Low-Rank Adaptation of Large Language Models][lora-paper].
//!
//! [lora-paper]: https://arxiv.org/abs/2106.09685
//!
//! # Quick start
//!
//! ```no_run
//! use candle::{DType, Device, Tensor};
//! use candle_nn::{linear, lora::{LoraConfig, LoraLinear}, Module, VarBuilder, VarMap};
//!
//! # fn main() -> candle::Result<()> {
//! // 1. Load a frozen base layer (e.g. from pretrained weights).
//! let dev = Device::Cpu;
//! let base_vs = VarMap::new();
//! let base_vb = VarBuilder::from_varmap(&base_vs, DType::F32, &dev);
//! let base = linear(512, 512, base_vb)?;
//!
//! // 2. Create a separate VarMap for the trainable LoRA adapters.
//! let lora_vs = VarMap::new();
//! let lora_vb = VarBuilder::from_varmap(&lora_vs, DType::F32, &dev);
//!
//! // 3. Wrap the base layer with a rank-8 LoRA adapter.
//! let cfg = LoraConfig::new(8, 16.0);
//! let lora_layer = LoraLinear::from_linear(base, &cfg, lora_vb.pp("lora"))?;
//!
//! // 4. Only `lora_vs.all_vars()` goes into the optimizer — the base
//! //    layer's weights are not `Var`s and receive no gradients.
//! let x = Tensor::zeros((1, 512), DType::F32, &dev)?;
//! let y = lora_layer.forward(&x)?;
//! # Ok(()) }
//! ```
//!
//! # Merging adapters
//!
//! Once training is done, you can collapse the LoRA adapter back into the
//! base weight matrix using [`LoraLinear::merge`]. The merged layer has the
//! same forward behavior as the unmerged one but no longer carries the
//! low-rank adapters, so inference is free of the extra matmul.
//!
//! ```no_run
//! # use candle::{DType, Device, Tensor};
//! # use candle_nn::{linear, lora::{LoraConfig, LoraLinear}, Module, VarBuilder, VarMap};
//! # fn main() -> candle::Result<()> {
//! # let dev = Device::Cpu;
//! # let base_vs = VarMap::new();
//! # let base_vb = VarBuilder::from_varmap(&base_vs, DType::F32, &dev);
//! # let base = linear(512, 512, base_vb)?;
//! # let lora_vs = VarMap::new();
//! # let lora_vb = VarBuilder::from_varmap(&lora_vs, DType::F32, &dev);
//! # let cfg = LoraConfig::new(8, 16.0);
//! # let lora_layer = LoraLinear::from_linear(base, &cfg, lora_vb.pp("lora"))?;
//! let merged = lora_layer.merge()?;
//! // `merged` is a regular `Linear` with `A` and `B` folded into its weight.
//! # Ok(()) }
//! ```

use candle::{DType, Module, Result, Tensor};

use crate::{Init, Linear, VarBuilder};

/// Configuration for a LoRA adapter.
#[derive(Debug, Clone, Copy)]
pub struct LoraConfig {
    /// Rank of the low-rank decomposition. Must be ≥ 1. Typical values are
    /// 4, 8, 16, 32.
    pub rank: usize,
    /// Scaling factor applied to the LoRA delta. The effective forward is
    /// `base(x) + (alpha / rank) * B(A(x))`. Common convention is to set
    /// `alpha = 2 * rank`.
    pub alpha: f64,
}

impl LoraConfig {
    /// Create a new LoRA configuration.
    ///
    /// # Panics
    /// Panics if `rank == 0`.
    pub fn new(rank: usize, alpha: f64) -> Self {
        assert!(rank > 0, "LoraConfig::new: rank must be ≥ 1");
        Self { rank, alpha }
    }

    /// The effective per-forward scaling factor (`alpha / rank`).
    #[inline]
    pub fn scale(&self) -> f64 {
        self.alpha / self.rank as f64
    }
}

/// A frozen [`Linear`] layer wrapped with a trainable low-rank `B @ A`
/// adapter.
///
/// The base layer's weights are *not* [`candle::Var`]s, so they receive no
/// gradients during backpropagation. The `A` and `B` matrices are loaded
/// from the provided [`VarBuilder`] and are the only trainable parameters
/// of a `LoraLinear`.
#[derive(Clone, Debug)]
pub struct LoraLinear {
    base: Linear,
    lora_a: Tensor, // shape (rank, in_features)
    lora_b: Tensor, // shape (out_features, rank)
    config: LoraConfig,
}

impl LoraLinear {
    /// Wrap an existing [`Linear`] layer with a LoRA adapter.
    ///
    /// `A` is initialized with Kaiming-normal (the standard LoRA init) and
    /// `B` is initialized to zero, so at step 0 the adapter is a perfect
    /// identity — the wrapped layer produces the exact same output as the
    /// base layer until gradient updates begin to move `B` away from zero.
    ///
    /// The adapter variables live under `vb` at paths `"lora_a"` and
    /// `"lora_b"`. Use `vb.pp("some_prefix")` to namespace them.
    pub fn from_linear(base: Linear, config: &LoraConfig, vb: VarBuilder) -> Result<Self> {
        let w = base.weight();
        let dims = w.dims();
        if dims.len() != 2 {
            candle::bail!(
                "LoraLinear::from_linear: base weight must be 2D, got rank {}",
                dims.len()
            );
        }
        let out_features = dims[0];
        let in_features = dims[1];

        let lora_a = vb.get_with_hints(
            (config.rank, in_features),
            "lora_a",
            crate::init::DEFAULT_KAIMING_NORMAL,
        )?;
        let lora_b = vb.get_with_hints((out_features, config.rank), "lora_b", Init::Const(0.0))?;

        Ok(Self {
            base,
            lora_a,
            lora_b,
            config: *config,
        })
    }

    /// Borrow the underlying frozen base layer.
    pub fn base(&self) -> &Linear {
        &self.base
    }

    /// Borrow the LoRA `A` matrix (shape `(rank, in_features)`).
    pub fn lora_a(&self) -> &Tensor {
        &self.lora_a
    }

    /// Borrow the LoRA `B` matrix (shape `(out_features, rank)`).
    pub fn lora_b(&self) -> &Tensor {
        &self.lora_b
    }

    /// The configuration this layer was built with.
    pub fn config(&self) -> &LoraConfig {
        &self.config
    }

    /// Compute the LoRA delta weight `(alpha / rank) * B @ A`.
    ///
    /// Shape: `(out_features, in_features)`. Useful for merging the adapter
    /// into the base weights at deployment time.
    pub fn delta_weight(&self) -> Result<Tensor> {
        let delta = self.lora_b.matmul(&self.lora_a)?;
        delta.affine(self.config.scale(), 0.0)
    }

    /// Collapse the LoRA adapter into a single [`Linear`] layer whose
    /// weight is `base_w + delta_weight`. Bias, if present, is preserved
    /// unchanged.
    ///
    /// The returned layer has no trainable LoRA parameters and runs at the
    /// same cost as a plain [`Linear`] — useful for deploying fine-tuned
    /// weights after training.
    pub fn merge(&self) -> Result<Linear> {
        let base_w = self.base.weight();
        let delta = self
            .delta_weight()?
            .to_dtype(base_w.dtype())?
            .to_device(base_w.device())?;
        let merged_w = (base_w + delta)?;
        Ok(Linear::new(merged_w, self.base.bias().cloned()))
    }
}

impl Module for LoraLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let base_out = self.base.forward(x)?;
        // LoRA delta path: x @ A^T @ B^T, scaled by (alpha/rank).
        //
        // We deliberately do not route this through `Linear::forward` on a
        // bias-less `Linear`, because the extra reshape logic there is
        // tuned for the common path; LoRA's rank dimension is small enough
        // that a pair of matmuls is fine and keeps the code obvious.
        let x_dtype = x.dtype();
        let compute_dtype = promote_to_f32(x_dtype);
        let x_c = if x.dtype() != compute_dtype {
            x.to_dtype(compute_dtype)?
        } else {
            x.clone()
        };
        let a = if self.lora_a.dtype() != compute_dtype {
            self.lora_a.to_dtype(compute_dtype)?
        } else {
            self.lora_a.clone()
        };
        let b = if self.lora_b.dtype() != compute_dtype {
            self.lora_b.to_dtype(compute_dtype)?
        } else {
            self.lora_b.clone()
        };

        // x @ A^T: last dim (in_features) → rank.
        let xa = match *x_c.dims() {
            [b1, b2, m, _] => x_c.reshape((b1 * b2 * m, ()))?.matmul(&a.t()?)?.reshape((
                b1,
                b2,
                m,
                self.config.rank,
            ))?,
            [bsize, m, _] => x_c.reshape((bsize * m, ()))?.matmul(&a.t()?)?.reshape((
                bsize,
                m,
                self.config.rank,
            ))?,
            _ => x_c.matmul(&a.t()?)?,
        };

        // (x @ A^T) @ B^T: rank → out_features.
        let delta = match *xa.dims() {
            [b1, b2, m, _] => {
                xa.reshape((b1 * b2 * m, ()))?
                    .matmul(&b.t()?)?
                    .reshape((b1, b2, m, ()))?
            }
            [bsize, m, _] => {
                xa.reshape((bsize * m, ()))?
                    .matmul(&b.t()?)?
                    .reshape((bsize, m, ()))?
            }
            _ => xa.matmul(&b.t()?)?,
        };

        let delta = delta.affine(self.config.scale(), 0.0)?;
        let delta = if delta.dtype() != base_out.dtype() {
            delta.to_dtype(base_out.dtype())?
        } else {
            delta
        };
        base_out + delta
    }
}

/// Pick a compute dtype for the LoRA matmul. For integer inputs or lower
/// precision inputs we compute in `F32`; floating-point inputs use their
/// native dtype. This keeps LoRA numerically stable while still honoring
/// user dtype choices.
fn promote_to_f32(dtype: DType) -> DType {
    match dtype {
        DType::BF16 | DType::F16 | DType::U8 | DType::U32 | DType::I64 => DType::F32,
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{linear, VarMap};
    use candle::{Device, Tensor};

    fn make_base(in_dim: usize, out_dim: usize, dev: &Device) -> Result<Linear> {
        let vs = VarMap::new();
        let vb = VarBuilder::from_varmap(&vs, DType::F32, dev);
        linear(in_dim, out_dim, vb)
    }

    #[test]
    fn lora_config_scale() {
        let c = LoraConfig::new(8, 16.0);
        assert_eq!(c.scale(), 2.0);
    }

    #[test]
    #[should_panic(expected = "rank must be ≥ 1")]
    fn lora_config_zero_rank_panics() {
        let _ = LoraConfig::new(0, 8.0);
    }

    #[test]
    fn zero_init_b_is_identity_at_step_zero() {
        let dev = Device::Cpu;
        let base = make_base(16, 32, &dev).unwrap();
        let vs = VarMap::new();
        let vb = VarBuilder::from_varmap(&vs, DType::F32, &dev);
        let layer = LoraLinear::from_linear(base.clone(), &LoraConfig::new(4, 8.0), vb).unwrap();

        // Because B is zero-initialized, the LoRA path contributes nothing
        // at step 0 and the wrapped forward must exactly match the base.
        let x = Tensor::randn(0f32, 1.0, (2, 16), &dev).unwrap();
        let base_y = base.forward(&x).unwrap();
        let lora_y = layer.forward(&x).unwrap();
        let diff = (base_y - lora_y).unwrap().abs().unwrap().max_all().unwrap();
        let d = diff.to_scalar::<f32>().unwrap();
        assert!(d < 1e-5, "expected identity at init, got max |diff| = {d}");
    }

    #[test]
    fn delta_weight_matches_manual_matmul() {
        let dev = Device::Cpu;
        let base = make_base(8, 8, &dev).unwrap();
        let vs = VarMap::new();
        let vb = VarBuilder::from_varmap(&vs, DType::F32, &dev);
        let mut layer = LoraLinear::from_linear(base, &LoraConfig::new(4, 8.0), vb).unwrap();

        // Overwrite B with a non-trivial tensor so delta_weight isn't zero.
        let b_new = Tensor::randn(0f32, 1.0, (8, 4), &dev).unwrap();
        for (name, var) in vs.data().lock().unwrap().iter() {
            if name.contains("lora_b") {
                var.set(&b_new).unwrap();
            }
        }
        // Re-create the layer handle so lora_b picks up the update.
        let layer_b = {
            let data = vs.data().lock().unwrap();
            data.iter()
                .find(|(n, _)| n.contains("lora_b"))
                .unwrap()
                .1
                .as_tensor()
                .clone()
        };
        layer.lora_b = layer_b;

        let delta = layer.delta_weight().unwrap();
        assert_eq!(delta.dims(), &[8, 8]);

        // Compare to a manual computation: scale * B @ A
        let a = layer.lora_a().clone();
        let b = layer.lora_b().clone();
        let manual = b
            .matmul(&a)
            .unwrap()
            .affine(layer.config().scale(), 0.0)
            .unwrap();
        let diff = (delta - manual).unwrap().abs().unwrap().max_all().unwrap();
        let d = diff.to_scalar::<f32>().unwrap();
        assert!(d < 1e-5, "delta_weight disagreed with manual B @ A: {d}");
    }

    #[test]
    fn merge_preserves_forward_behavior() {
        let dev = Device::Cpu;
        let base = make_base(16, 32, &dev).unwrap();
        let vs = VarMap::new();
        let vb = VarBuilder::from_varmap(&vs, DType::F32, &dev);
        let mut layer = LoraLinear::from_linear(base, &LoraConfig::new(4, 8.0), vb).unwrap();

        // Perturb B so there's something non-trivial to merge.
        let b_new = Tensor::randn(0f32, 0.1, (32, 4), &dev).unwrap();
        for (name, var) in vs.data().lock().unwrap().iter() {
            if name.contains("lora_b") {
                var.set(&b_new).unwrap();
            }
        }
        let layer_b = {
            let data = vs.data().lock().unwrap();
            data.iter()
                .find(|(n, _)| n.contains("lora_b"))
                .unwrap()
                .1
                .as_tensor()
                .clone()
        };
        layer.lora_b = layer_b;

        let merged = layer.merge().unwrap();

        let x = Tensor::randn(0f32, 1.0, (4, 16), &dev).unwrap();
        let y_lora = layer.forward(&x).unwrap();
        let y_merged = merged.forward(&x).unwrap();
        let diff = (y_lora - y_merged)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap();
        let d = diff.to_scalar::<f32>().unwrap();
        assert!(d < 1e-4, "merge changed forward output: max |diff| = {d}");
    }

    #[test]
    fn forward_handles_3d_input() {
        let dev = Device::Cpu;
        let base = make_base(12, 24, &dev).unwrap();
        let vs = VarMap::new();
        let vb = VarBuilder::from_varmap(&vs, DType::F32, &dev);
        let layer = LoraLinear::from_linear(base, &LoraConfig::new(4, 8.0), vb).unwrap();

        let x = Tensor::randn(0f32, 1.0, (2, 5, 12), &dev).unwrap();
        let y = layer.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 5, 24]);
    }

    #[test]
    fn wrong_rank_base_weight_errors() {
        let dev = Device::Cpu;
        let t = Tensor::zeros((3, 4, 5), DType::F32, &dev).unwrap();
        let bad_base = Linear::new(t, None);
        let vs = VarMap::new();
        let vb = VarBuilder::from_varmap(&vs, DType::F32, &dev);
        let err = LoraLinear::from_linear(bad_base, &LoraConfig::new(4, 8.0), vb).unwrap_err();
        assert!(format!("{err}").contains("2D"));
    }
}
