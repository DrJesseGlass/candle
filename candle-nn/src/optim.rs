//! Various optimization algorithms.
use candle::{Result, Tensor, Var};

/// The interface optimizers should implement.
pub trait Optimizer: Sized {
    type Config: Sized;

    fn new(vars: Vec<Var>, config: Self::Config) -> Result<Self>;

    fn step(&mut self, grads: &candle::backprop::GradStore) -> Result<()>;

    fn learning_rate(&self) -> f64;

    fn set_learning_rate(&mut self, lr: f64);

    fn empty(config: Self::Config) -> Result<Self> {
        Self::new(vec![], config)
    }

    fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        self.step(&grads)
    }

    fn from_slice(vars: &[&Var], config: Self::Config) -> Result<Self> {
        let vars: Vec<_> = vars.iter().map(|&v| v.clone()).collect();
        Self::new(vars, config)
    }
}

/// Optimizer for Stochastic Gradient Descent.
///
/// Contrary to the PyTorch implementation of SGD, this version does not support momentum.
#[derive(Debug)]
pub struct SGD {
    vars: Vec<Var>,
    learning_rate: f64,
}

impl Optimizer for SGD {
    type Config = f64;

    fn new(vars: Vec<Var>, learning_rate: f64) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .collect();
        Ok(Self {
            vars,
            learning_rate,
        })
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn step(&mut self, grads: &candle::backprop::GradStore) -> Result<()> {
        for var in self.vars.iter() {
            if let Some(grad) = grads.get(var) {
                var.set(&var.sub(&(grad * self.learning_rate)?)?)?;
            }
        }
        Ok(())
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr
    }
}

impl SGD {
    pub fn into_inner(self) -> Vec<Var> {
        self.vars
    }

    pub fn push(&mut self, var: &Var) {
        self.vars.push(var.clone())
    }
}

#[derive(Clone, Debug)]
pub struct ParamsAdamW {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl Default for ParamsAdamW {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

#[derive(Debug)]
struct VarAdamW {
    var: Var,
    first_moment: Var,
    second_moment: Var,
}

#[derive(Debug)]
pub struct AdamW {
    vars: Vec<VarAdamW>,
    step_t: usize,
    params: ParamsAdamW,
    /// Accumulated gradients for gradient accumulation. `None` when no
    /// gradients have been accumulated yet. Each entry corresponds to
    /// `vars[i].var` in the same order.
    accum_grads: Vec<Option<Tensor>>,
    accum_count: usize,
}

/// Per-step precomputed AdamW coefficients. Shared between the direct
/// `step` path and the gradient-accumulation `step_accumulated` path so
/// both apply the exact same update rule; changing the formula in one
/// place automatically changes it in the other.
struct AdamStep<'a> {
    params: &'a ParamsAdamW,
    lr_lambda: f64,
    scale_m: f64,
    scale_v: f64,
}

impl<'a> AdamStep<'a> {
    fn new(params: &'a ParamsAdamW, step_t: usize) -> Self {
        Self {
            params,
            lr_lambda: params.lr * params.weight_decay,
            scale_m: 1f64 / (1f64 - params.beta1.powi(step_t as i32)),
            scale_v: 1f64 / (1f64 - params.beta2.powi(step_t as i32)),
        }
    }

    // This involves locking 3 RWLocks per params, if the parameters are large this
    // should not be an issue but this may be problematic with models with lots of
    // small parameters.
    fn apply(&self, var: &VarAdamW, g: &Tensor) -> Result<()> {
        let theta = &var.var;
        let m = &var.first_moment;
        let v = &var.second_moment;
        let next_m = ((m.as_tensor() * self.params.beta1)? + (g * (1.0 - self.params.beta1))?)?;
        let next_v =
            ((v.as_tensor() * self.params.beta2)? + (g.sqr()? * (1.0 - self.params.beta2))?)?;
        let m_hat = (&next_m * self.scale_m)?;
        let v_hat = (&next_v * self.scale_v)?;
        let next_theta = (theta.as_tensor() * (1f64 - self.lr_lambda))?;
        let adjusted_grad = (m_hat / (v_hat.sqrt()? + self.params.eps)?)?;
        let next_theta = (next_theta - (adjusted_grad * self.params.lr)?)?;
        m.set(&next_m)?;
        v.set(&next_v)?;
        theta.set(&next_theta)?;
        Ok(())
    }
}

impl Optimizer for AdamW {
    type Config = ParamsAdamW;

    fn new(vars: Vec<Var>, params: ParamsAdamW) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let first_moment = Var::zeros(shape, dtype, device)?;
                let second_moment = Var::zeros(shape, dtype, device)?;
                Ok(VarAdamW {
                    var,
                    first_moment,
                    second_moment,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let n = vars.len();
        Ok(Self {
            vars,
            params,
            step_t: 0,
            accum_grads: vec![None; n],
            accum_count: 0,
        })
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr
    }

    fn step(&mut self, grads: &candle::backprop::GradStore) -> Result<()> {
        // Catch silent state-mixing: calling step/backward_step while
        // gradients are pending in the accumulator means those accumulated
        // micro-batches will never be applied (they'll be overwritten or
        // double-counted). Callers should flush via step_accumulated first.
        debug_assert_eq!(
            self.accum_count, 0,
            "AdamW::step called with {} pending accumulated gradient(s); \
             call step_accumulated() to flush them before step/backward_step",
            self.accum_count
        );
        self.step_t += 1;
        let upd = AdamStep::new(&self.params, self.step_t);
        for var in self.vars.iter() {
            if let Some(g) = grads.get(&var.var) {
                upd.apply(var, g)?;
            }
        }
        Ok(())
    }
}

impl AdamW {
    /// Compute gradients from `loss` and add them to an internal buffer.
    ///
    /// Call this K times (once per micro-batch), then call
    /// [`step_accumulated`] once to apply the averaged gradient update.
    /// This gives an effective batch size of K × micro_batch without
    /// holding K computation graphs in memory simultaneously.
    ///
    /// The accumulated gradients are divided by K in [`step_accumulated`],
    /// so the resulting parameter update equals the one produced by a
    /// single backward pass on the mean of the K per-micro-batch losses.
    /// If each micro-batch loss is itself a mean over its samples, this
    /// matches the gradient of a K× larger mean-reduced batch. If your
    /// losses are sums, divide by the total sample count yourself.
    pub fn accumulate_grad(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        for (i, var_adamw) in self.vars.iter().enumerate() {
            if let Some(g) = grads.get(&var_adamw.var) {
                // Detach unconditionally. Gradients from GradStore don't
                // currently carry op history, but if that ever changes the
                // accumulator would silently chain graphs across K
                // iterations — the exact memory blow-up that gradient
                // accumulation is meant to avoid. Guarding here makes the
                // invariant robust to future candle-core changes.
                self.accum_grads[i] = Some(match &self.accum_grads[i] {
                    Some(existing) => (existing + g)?.detach(),
                    None => g.detach().contiguous()?,
                });
            }
        }
        self.accum_count += 1;
        Ok(())
    }

    /// Apply the accumulated gradients (averaged over the number of
    /// [`accumulate_grad`] calls) and reset the accumulator.
    ///
    /// This performs one AdamW step using the averaged gradient as if it
    /// came from a single backward pass on a batch K× larger.
    ///
    /// Does nothing if no gradients have been accumulated.
    pub fn step_accumulated(&mut self) -> Result<()> {
        if self.accum_count == 0 {
            return Ok(());
        }
        let scale = 1.0 / self.accum_count as f64;
        self.step_t += 1;
        let upd = AdamStep::new(&self.params, self.step_t);
        for (i, var) in self.vars.iter().enumerate() {
            if let Some(acc_g) = &self.accum_grads[i] {
                let g = (acc_g * scale)?; // average over accumulation count
                upd.apply(var, &g)?;
            }
        }
        for g in self.accum_grads.iter_mut() {
            *g = None;
        }
        self.accum_count = 0;
        Ok(())
    }

    pub fn new_lr(vars: Vec<Var>, learning_rate: f64) -> Result<Self> {
        let params = ParamsAdamW {
            lr: learning_rate,
            ..ParamsAdamW::default()
        };
        Self::new(vars, params)
    }

    pub fn params(&self) -> &ParamsAdamW {
        &self.params
    }

    pub fn set_params(&mut self, params: ParamsAdamW) {
        self.params = params;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, Tensor};

    /// Verify that accumulating K micro-batch gradients then stepping once
    /// produces the same parameter update as a single backward_step on
    /// the sum of those K losses.
    #[test]
    fn accumulate_matches_single_step() {
        let dev = Device::Cpu;
        let params = ParamsAdamW {
            lr: 0.01,
            weight_decay: 0.0, // disable WD so the comparison is cleaner
            ..Default::default()
        };

        // Two copies of the same initial variable
        let w1 = Var::from_tensor(&Tensor::new(&[1.0f32, 2.0, 3.0], &dev).unwrap()).unwrap();
        let w2 = Var::from_tensor(&Tensor::new(&[1.0f32, 2.0, 3.0], &dev).unwrap()).unwrap();

        let x1 = Tensor::new(&[0.5f32, -0.5, 1.0], &dev).unwrap();
        let x2 = Tensor::new(&[-1.0f32, 0.3, 0.7], &dev).unwrap();

        // Path A: accumulate 2 micro-batches, step once
        let mut opt_a = AdamW::new(vec![w1.clone()], params.clone()).unwrap();
        let loss1a = (w1.as_tensor() * &x1).unwrap().sum_all().unwrap();
        let loss2a = (w1.as_tensor() * &x2).unwrap().sum_all().unwrap();
        opt_a.accumulate_grad(&loss1a).unwrap();
        opt_a.accumulate_grad(&loss2a).unwrap();
        opt_a.step_accumulated().unwrap();
        let result_a: Vec<f32> = w1.as_tensor().to_vec1().unwrap();

        // Path B: single backward_step on the averaged sum of both losses
        let mut opt_b = AdamW::new(vec![w2.clone()], params).unwrap();
        let loss1b = (w2.as_tensor() * &x1).unwrap().sum_all().unwrap();
        let loss2b = (w2.as_tensor() * &x2).unwrap().sum_all().unwrap();
        let combined = ((&loss1b + &loss2b).unwrap() * 0.5).unwrap(); // average
        opt_b.backward_step(&combined).unwrap();
        let result_b: Vec<f32> = w2.as_tensor().to_vec1().unwrap();

        for (a, b) in result_a.iter().zip(result_b.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "accumulate diverged from single step: {a} vs {b}"
            );
        }
    }

    #[test]
    fn step_accumulated_with_no_accumulation_is_noop() {
        let dev = Device::Cpu;
        let w = Var::from_tensor(&Tensor::new(&[1.0f32, 2.0], &dev).unwrap()).unwrap();
        let mut opt = AdamW::new(vec![w.clone()], ParamsAdamW::default()).unwrap();
        let before: Vec<f32> = w.as_tensor().to_vec1().unwrap();
        opt.step_accumulated().unwrap();
        let after: Vec<f32> = w.as_tensor().to_vec1().unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn accumulate_resets_after_step() {
        let dev = Device::Cpu;
        let w = Var::from_tensor(&Tensor::new(&[1.0f32, 2.0], &dev).unwrap()).unwrap();
        let mut opt = AdamW::new(vec![w.clone()], ParamsAdamW::default()).unwrap();
        let x = Tensor::new(&[1.0f32, 1.0], &dev).unwrap();
        let loss = (w.as_tensor() * &x).unwrap().sum_all().unwrap();
        opt.accumulate_grad(&loss).unwrap();
        opt.step_accumulated().unwrap();
        // After step, accumulator should be empty
        assert_eq!(opt.accum_count, 0);
        assert!(opt.accum_grads.iter().all(|g| g.is_none()));
    }

    /// Multi-variable parity with a non-zero weight decay, so the
    /// decoupled-decay path in AdamStep::apply is actually exercised.
    #[test]
    fn accumulate_multi_var_with_weight_decay() {
        let dev = Device::Cpu;
        let params = ParamsAdamW {
            lr: 0.01,
            weight_decay: 0.1,
            ..Default::default()
        };

        let wa1 = Var::from_tensor(&Tensor::new(&[1.0f32, -2.0, 3.0], &dev).unwrap()).unwrap();
        let wa2 = Var::from_tensor(&Tensor::new(&[0.5f32, 0.5], &dev).unwrap()).unwrap();
        let wb1 = Var::from_tensor(&Tensor::new(&[1.0f32, -2.0, 3.0], &dev).unwrap()).unwrap();
        let wb2 = Var::from_tensor(&Tensor::new(&[0.5f32, 0.5], &dev).unwrap()).unwrap();

        let x1 = Tensor::new(&[0.5f32, -0.5, 1.0], &dev).unwrap();
        let x2 = Tensor::new(&[-1.0f32, 0.3, 0.7], &dev).unwrap();
        let y1 = Tensor::new(&[1.0f32, 0.5], &dev).unwrap();
        let y2 = Tensor::new(&[-0.5f32, 2.0], &dev).unwrap();

        // Path A: accumulate 2 micro-batches over both vars, step once.
        let mut opt_a = AdamW::new(vec![wa1.clone(), wa2.clone()], params.clone()).unwrap();
        let l1a = ((wa1.as_tensor() * &x1).unwrap().sum_all().unwrap()
            + (wa2.as_tensor() * &y1).unwrap().sum_all().unwrap())
        .unwrap();
        let l2a = ((wa1.as_tensor() * &x2).unwrap().sum_all().unwrap()
            + (wa2.as_tensor() * &y2).unwrap().sum_all().unwrap())
        .unwrap();
        opt_a.accumulate_grad(&l1a).unwrap();
        opt_a.accumulate_grad(&l2a).unwrap();
        opt_a.step_accumulated().unwrap();

        // Path B: single backward_step on the averaged sum of both losses.
        let mut opt_b = AdamW::new(vec![wb1.clone(), wb2.clone()], params).unwrap();
        let l1b = ((wb1.as_tensor() * &x1).unwrap().sum_all().unwrap()
            + (wb2.as_tensor() * &y1).unwrap().sum_all().unwrap())
        .unwrap();
        let l2b = ((wb1.as_tensor() * &x2).unwrap().sum_all().unwrap()
            + (wb2.as_tensor() * &y2).unwrap().sum_all().unwrap())
        .unwrap();
        let combined = ((&l1b + &l2b).unwrap() * 0.5).unwrap();
        opt_b.backward_step(&combined).unwrap();

        let a1: Vec<f32> = wa1.as_tensor().to_vec1().unwrap();
        let b1: Vec<f32> = wb1.as_tensor().to_vec1().unwrap();
        let a2: Vec<f32> = wa2.as_tensor().to_vec1().unwrap();
        let b2: Vec<f32> = wb2.as_tensor().to_vec1().unwrap();
        for (a, b) in a1.iter().zip(b1.iter()) {
            assert!((a - b).abs() < 1e-5, "var1 diverged: {a} vs {b}");
        }
        for (a, b) in a2.iter().zip(b2.iter()) {
            assert!((a - b).abs() < 1e-5, "var2 diverged: {a} vs {b}");
        }
    }

    /// A variable that never appears in the loss graph must be left
    /// untouched — the accumulator should skip it and step_accumulated
    /// must not update it.
    #[test]
    fn accumulate_skips_var_without_grad() {
        let dev = Device::Cpu;
        let used = Var::from_tensor(&Tensor::new(&[1.0f32, 2.0], &dev).unwrap()).unwrap();
        let unused = Var::from_tensor(&Tensor::new(&[7.0f32, 8.0], &dev).unwrap()).unwrap();
        let used_before: Vec<f32> = used.as_tensor().to_vec1().unwrap();
        let unused_before: Vec<f32> = unused.as_tensor().to_vec1().unwrap();

        let mut opt =
            AdamW::new(vec![used.clone(), unused.clone()], ParamsAdamW::default()).unwrap();
        let x = Tensor::new(&[0.5f32, 0.5], &dev).unwrap();
        let loss = (used.as_tensor() * &x).unwrap().sum_all().unwrap();
        opt.accumulate_grad(&loss).unwrap();
        opt.accumulate_grad(&loss).unwrap();
        opt.step_accumulated().unwrap();

        let used_after: Vec<f32> = used.as_tensor().to_vec1().unwrap();
        let unused_after: Vec<f32> = unused.as_tensor().to_vec1().unwrap();
        assert_ne!(used_after, used_before, "used var should have moved");
        assert_eq!(unused_after, unused_before, "unused var must not move");
    }
}
