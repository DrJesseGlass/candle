//! Low-Rank Adaptation (LoRA) for linear layers.

use candle::{DType, Module, Result, Tensor};

use crate::{Init, Linear, VarBuilder};

#[derive(Debug, Clone, Copy)]
pub struct LoraConfig {
    pub rank: usize,
    pub alpha: f64,
}

impl LoraConfig {
    pub fn new(rank: usize, alpha: f64) -> Self {
        assert!(rank > 0, "LoraConfig::new: rank must be >= 1");
        Self { rank, alpha }
    }

    #[inline]
    pub fn scale(&self) -> f64 {
        self.alpha / self.rank as f64
    }
}

/// A frozen [`Linear`] layer wrapped with a trainable low-rank `B @ A` adapter.
#[derive(Clone, Debug)]
pub struct LoraLinear {
    base: Linear,
    lora_a: Tensor,
    lora_b: Tensor,
    config: LoraConfig,
}

impl LoraLinear {
    /// Wrap an existing [`Linear`] with a LoRA adapter; base weight/bias are detached.
    pub fn from_linear(base: Linear, config: &LoraConfig, vb: VarBuilder) -> Result<Self> {
        let base = Linear::new(base.weight().detach(), base.bias().map(|b| b.detach()));
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

    pub fn base(&self) -> &Linear {
        &self.base
    }

    pub fn lora_a(&self) -> &Tensor {
        &self.lora_a
    }

    pub fn lora_b(&self) -> &Tensor {
        &self.lora_b
    }

    pub fn config(&self) -> &LoraConfig {
        &self.config
    }

    /// `(alpha / rank) * B @ A`, shape `(out_features, in_features)`.
    pub fn delta_weight(&self) -> Result<Tensor> {
        let delta = self.lora_b.matmul(&self.lora_a)?;
        delta.affine(self.config.scale(), 0.0)
    }

    /// Fold the adapter into a plain [`Linear`] with `weight = base + delta`.
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
        let compute_dtype = promote_to_f32(x.dtype());
        let cast = |t: &Tensor| -> Result<Tensor> {
            if t.dtype() != compute_dtype {
                t.to_dtype(compute_dtype)
            } else {
                Ok(t.clone())
            }
        };
        let x_c = cast(x)?;
        let a = Linear::new(cast(&self.lora_a)?, None);
        let b = Linear::new(cast(&self.lora_b)?, None);

        let delta = b.forward(&a.forward(&x_c)?)?;
        let delta = delta.affine(self.config.scale(), 0.0)?;
        let delta = if delta.dtype() != base_out.dtype() {
            delta.to_dtype(base_out.dtype())?
        } else {
            delta
        };
        base_out + delta
    }
}

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

    fn overwrite_var(vs: &VarMap, name_contains: &str, value: &Tensor) -> Tensor {
        {
            let data = vs.data().lock().unwrap();
            for (name, var) in data.iter() {
                if name.contains(name_contains) {
                    var.set(value).unwrap();
                }
            }
        }
        let data = vs.data().lock().unwrap();
        data.iter()
            .find(|(n, _)| n.contains(name_contains))
            .unwrap()
            .1
            .as_tensor()
            .clone()
    }

    #[test]
    fn lora_config_scale() {
        let c = LoraConfig::new(8, 16.0);
        assert_eq!(c.scale(), 2.0);
    }

    #[test]
    #[should_panic(expected = "rank must be >= 1")]
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

        let b_new = Tensor::randn(0f32, 1.0, (8, 4), &dev).unwrap();
        layer.lora_b = overwrite_var(&vs, "lora_b", &b_new);

        let delta = layer.delta_weight().unwrap();
        assert_eq!(delta.dims(), &[8, 8]);

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

        let b_new = Tensor::randn(0f32, 0.1, (32, 4), &dev).unwrap();
        layer.lora_b = overwrite_var(&vs, "lora_b", &b_new);

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
    fn base_weights_detached_from_autograd() {
        let dev = Device::Cpu;
        let base_vs = VarMap::new();
        let base_vb = VarBuilder::from_varmap(&base_vs, DType::F32, &dev);
        let base = linear(8, 4, base_vb).unwrap();

        let lora_vs = VarMap::new();
        let lora_vb = VarBuilder::from_varmap(&lora_vs, DType::F32, &dev);
        let mut layer = LoraLinear::from_linear(base, &LoraConfig::new(2, 4.0), lora_vb).unwrap();

        let b_new = Tensor::randn(0f32, 0.1, (4, 2), &dev).unwrap();
        layer.lora_b = overwrite_var(&lora_vs, "lora_b", &b_new);

        let x = Tensor::randn(0f32, 1.0, (3, 8), &dev).unwrap();
        let y = layer.forward(&x).unwrap();
        let loss = y.sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();

        for var in base_vs.all_vars() {
            assert!(
                grads.get(var.as_tensor()).is_none(),
                "base var unexpectedly received a gradient"
            );
        }
        let lora_vars = lora_vs.all_vars();
        assert!(
            lora_vars.iter().any(|v| grads.get(v.as_tensor()).is_some()),
            "no LoRA var received a gradient"
        );
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
