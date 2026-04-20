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
        self.step_t += 1;
        let lr = self.params.lr;
        let lambda = self.params.weight_decay;
        let lr_lambda = lr * lambda;
        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;
        let scale_m = 1f64 / (1f64 - beta1.powi(self.step_t as i32));
        let scale_v = 1f64 / (1f64 - beta2.powi(self.step_t as i32));
        for var in self.vars.iter() {
            let theta = &var.var;
            let m = &var.first_moment;
            let v = &var.second_moment;
            if let Some(g) = grads.get(theta) {
                // This involves locking 3 RWLocks per params, if the parameters are large this
                // should not be an issue but this may be problematic with models with lots of
                // small parameters.
                let next_m = ((m.as_tensor() * beta1)? + (g * (1.0 - beta1))?)?;
                let next_v = ((v.as_tensor() * beta2)? + (g.sqr()? * (1.0 - beta2))?)?;
                let m_hat = (&next_m * scale_m)?;
                let v_hat = (&next_v * scale_v)?;
                let next_theta = (theta.as_tensor() * (1f64 - lr_lambda))?;
                let adjusted_grad = (m_hat / (v_hat.sqrt()? + self.params.eps)?)?;
                let next_theta = (next_theta - (adjusted_grad * lr)?)?;
                m.set(&next_m)?;
                v.set(&next_v)?;
                theta.set(&next_theta)?;
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
    /// The accumulated gradients are **averaged** (divided by K) when
    /// [`step_accumulated`] is called, matching the PyTorch convention.
    pub fn accumulate_grad(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        for (i, var_adamw) in self.vars.iter().enumerate() {
            if let Some(g) = grads.get(&var_adamw.var) {
                match &self.accum_grads[i] {
                    Some(existing) => {
                        self.accum_grads[i] = Some((existing + g)?);
                    }
                    None => {
                        // Detach + contiguous so the tensor owns its storage
                        // and doesn't keep the backward graph alive.
                        self.accum_grads[i] = Some(g.detach().contiguous()?);
                    }
                }
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
        let lr = self.params.lr;
        let lambda = self.params.weight_decay;
        let lr_lambda = lr * lambda;
        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;
        let scale_m = 1f64 / (1f64 - beta1.powi(self.step_t as i32));
        let scale_v = 1f64 / (1f64 - beta2.powi(self.step_t as i32));
        for (i, var) in self.vars.iter().enumerate() {
            if let Some(ref acc_g) = self.accum_grads[i] {
                let g = (acc_g * scale)?; // average over accumulation count
                let theta = &var.var;
                let m = &var.first_moment;
                let v = &var.second_moment;
                let next_m = ((m.as_tensor() * beta1)? + (&g * (1.0 - beta1))?)?;
                let next_v = ((v.as_tensor() * beta2)? + (g.sqr()? * (1.0 - beta2))?)?;
                let m_hat = (&next_m * scale_m)?;
                let v_hat = (&next_v * scale_v)?;
                let next_theta = (theta.as_tensor() * (1f64 - lr_lambda))?;
                let adjusted_grad = (m_hat / (v_hat.sqrt()? + self.params.eps)?)?;
                let next_theta = (next_theta - (adjusted_grad * lr)?)?;
                m.set(&next_m)?;
                v.set(&next_v)?;
                theta.set(&next_theta)?;
            }
        }
        // Reset accumulator
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
}
