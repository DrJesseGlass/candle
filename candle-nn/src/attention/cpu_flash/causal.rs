// Index loops (for t in 0..d) are intentional for SIMD auto-vectorization.
#![allow(clippy::needless_range_loop)]

// Single-batch (B=1) causal attention using loop-bound masking.

use candle::{Device, Result, Storage, Tensor, WithDType};
use rayon::prelude::*;

#[cfg(feature = "f16-attn-dot")]
use super::dot_f16_f16;
#[cfg(not(feature = "f16-attn-dot"))]
use super::dot_f32_f16;
use super::online_softmax::online_softmax_step;
use super::standard::{FLASH_ATTN_POOL, FLASH_DECODE_POOL};
use super::{axpy_f16, dot_f32};

/// Prefetch a cache line for read.
#[inline(always)]
fn prefetch_read<T>(ptr: *const T) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::asm!("prfm pldl1keep, [{ptr}]", ptr = in(reg) ptr, options(nostack, preserves_flags));
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = ptr;
    }
}

// Opt-in (CANDLE_VEC_SOFTMAX_EXP=1) NEON polynomial exp for the flash-attention softmax.
// The N1 instruction breakdown showed `expf` (libm, scalar) as the kernel's `transcendental`
// cost; this replaces it with FMA (~1e-6 accurate, normalized out by softmax). Default OFF so
// the default path stays bit-reproducible; flip on once validated on N1.
#[cfg(target_arch = "aarch64")]
static VEC_SOFTMAX_EXP: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var("CANDLE_VEC_SOFTMAX_EXP").is_ok());

// exp(x) via range-reduction x = n*ln2 + r, exp(x) = 2^n * poly(r). Scalar form so the
// vector body and the <4 tail use the SAME approximation (consistent softmax values).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn poly_exp_scalar(x: f32) -> f32 {
    let x = x.clamp(-87.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E).round();
    let r = x - n * std::f32::consts::LN_2;
    let mut p = 1.0 / 720.0;
    p = p * r + 1.0 / 120.0;
    p = p * r + 1.0 / 24.0;
    p = p * r + 1.0 / 6.0;
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;
    p * f32::from_bits((((n as i32) + 127) << 23) as u32)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn vec_exp_sub_sum_neon(s: &mut [f32], mr: f32) -> f32 {
    use std::arch::aarch64::*;
    #[inline(always)]
    unsafe fn vexpq(x: float32x4_t) -> float32x4_t {
        let x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.0)), vdupq_n_f32(88.0));
        let n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(std::f32::consts::LOG2_E)));
        let r = vfmsq_f32(x, n, vdupq_n_f32(std::f32::consts::LN_2));
        let mut p = vdupq_n_f32(1.0 / 720.0);
        p = vfmaq_f32(vdupq_n_f32(1.0 / 120.0), p, r);
        p = vfmaq_f32(vdupq_n_f32(1.0 / 24.0), p, r);
        p = vfmaq_f32(vdupq_n_f32(1.0 / 6.0), p, r);
        p = vfmaq_f32(vdupq_n_f32(0.5), p, r);
        p = vfmaq_f32(vdupq_n_f32(1.0), p, r);
        p = vfmaq_f32(vdupq_n_f32(1.0), p, r);
        let bits = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
        vmulq_f32(p, vreinterpretq_f32_s32(bits))
    }
    let vmr = vdupq_n_f32(mr);
    let mut vsum = vdupq_n_f32(0.0);
    let n = s.len();
    let p = s.as_mut_ptr();
    let mut i = 0;
    while i + 4 <= n {
        let e = vexpq(vsubq_f32(vld1q_f32(p.add(i)), vmr));
        vst1q_f32(p.add(i), e);
        vsum = vaddq_f32(vsum, e);
        i += 4;
    }
    let mut sum = vaddvq_f32(vsum);
    while i < n {
        let e = poly_exp_scalar(*p.add(i) - mr);
        *p.add(i) = e;
        sum += e;
        i += 1;
    }
    sum
}

/// For each `x` in `s`, set `x = exp(x - mr)` and return the sum. The softmax inner loop.
#[inline(always)]
fn exp_sub_sum(s: &mut [f32], mr: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if *VEC_SOFTMAX_EXP {
        return unsafe { vec_exp_sub_sum_neon(s, mr) };
    }
    let mut sum = 0.0f32;
    for x in s.iter_mut() {
        let e = (*x - mr).exp();
        *x = e;
        sum += e;
    }
    sum
}

/// Causal attention with loop-bound masking, **B=1 only**.
///
/// Squeezes batch dim, extracts contiguous slices, dispatches to
/// f32 or generic kernel. The inner kernels operate on raw slices only.
#[allow(clippy::too_many_arguments)]
pub fn run_causal_attn_cpu<T>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    kv_offset: usize,
    max_bias: Option<f32>,
    softcap: Option<f32>,
) -> Result<Tensor>
where
    T: WithDType,
{
    let b = q.dims()[0];
    if b != 1 {
        candle::bail!(
            "causal::run_causal_attn_cpu is B=1 only (got B={b}). \
             Multi-batch should be routed through the varlen path."
        );
    }

    let q = q.squeeze(0)?.contiguous()?;
    let k = k.squeeze(0)?.contiguous()?;
    let v = v.squeeze(0)?.contiguous()?;

    let (s_q, h_q, d) = q.dims3()?;
    let (s_kv, h_kv, _) = k.dims3()?;
    let (_, h_v, _) = v.dims3()?;

    let max_bias = max_bias.unwrap_or(0.0);
    let softcap = softcap.unwrap_or(0.0);
    // Lean = no ALiBi and no softcap: the common case (Qwen3, SmolLM3, most LLMs).
    // Dispatched as a const generic so the kernel's bias/softcap branches vanish.
    let lean = max_bias == 0.0 && softcap == 0.0;

    let (q_g, q_l) = q.storage_and_layout();
    let q_data: &[T] = match &*q_g {
        Storage::Cpu(cpu) => &cpu.as_slice::<T>()?[q_l.start_offset()..],
        _ => candle::bail!("Expected CPU storage"),
    };
    let (k_g, k_l) = k.storage_and_layout();
    let k_data: &[T] = match &*k_g {
        Storage::Cpu(cpu) => &cpu.as_slice::<T>()?[k_l.start_offset()..],
        _ => candle::bail!("Expected CPU storage"),
    };
    let (v_g, v_l) = v.storage_and_layout();
    let v_data: &[T] = match &*v_g {
        Storage::Cpu(cpu) => &cpu.as_slice::<T>()?[v_l.start_offset()..],
        _ => candle::bail!("Expected CPU storage"),
    };

    let result = match (s_q == 1, lean) {
        (true, true) => causal_decode::<true, T>(
            q_data,
            k_data,
            v_data,
            h_q,
            h_kv,
            h_v,
            d,
            s_kv,
            softmax_scale,
            max_bias,
            softcap,
        ),
        (true, false) => causal_decode::<false, T>(
            q_data,
            k_data,
            v_data,
            h_q,
            h_kv,
            h_v,
            d,
            s_kv,
            softmax_scale,
            max_bias,
            softcap,
        ),
        (false, true) => causal_prefill::<true, T>(
            q_data,
            k_data,
            v_data,
            s_q,
            h_q,
            h_kv,
            h_v,
            d,
            s_kv,
            softmax_scale,
            kv_offset,
            max_bias,
            softcap,
        ),
        (false, false) => causal_prefill::<false, T>(
            q_data,
            k_data,
            v_data,
            s_q,
            h_q,
            h_kv,
            h_v,
            d,
            s_kv,
            softmax_scale,
            kv_offset,
            max_bias,
            softcap,
        ),
    };

    result.and_then(|t| t.unsqueeze(0))
}

// Decode (q_len == 1). Input layout is contiguous (1, S, H, D); we index past the
// batch dim. q[h] starts at h*D; k/v[pos, h] starts at pos*H_kv*D + h*D.
//
// `LEAN` is a const generic: for the common no-ALiBi/no-softcap case the bias and
// softcap branches monomorphize away, giving the same branchless inner loop the
// former hand-written `_lean` kernel had. Generic over `T` so f32 and f16/bf16 share
// one body - for f32, `(x as f64) as f32` is exact, so codegen matches the old
// f32-specialized kernel.
#[allow(clippy::too_many_arguments)]
fn causal_decode<const LEAN: bool, T: WithDType>(
    q_data: &[T],
    k_data: &[T],
    v_data: &[T],
    h_q: usize,
    h_kv: usize,
    h_v: usize,
    d: usize,
    kv_len: usize,
    scale: f32,
    max_bias: f32,
    logit_softcap: f32,
) -> Result<Tensor> {
    let rk = h_q / h_kv;
    let rv = h_q / h_v;
    let n2 = 2_usize.pow((h_q as f32).log2().ceil() as u32);

    let (scale_pre, do_softcap) = if logit_softcap != 0.0 {
        (scale / logit_softcap, true)
    } else {
        (scale, false)
    };

    let k_seq_stride = h_kv * d;
    let v_seq_stride = h_v * d;

    let mut out = vec![0f32; h_q * d];

    FLASH_ATTN_POOL.install(|| {
        out.par_chunks_mut(d).enumerate().for_each_init(
            || vec![0f32; d],
            |acc, (h_i, out_chunk)| {
                let slope = if !LEAN && max_bias > 0.0 {
                    2.0f32.powf(-max_bias * ((h_i + 1) as f32) / n2 as f32)
                } else {
                    0.0
                };
                let k_head_off = (h_i / rk) * d;
                let v_head_off = (h_i / rv) * d;
                let q_row = &q_data[h_i * d..(h_i + 1) * d];

                acc.fill(0.0);
                let mut m = f32::NEG_INFINITY;
                let mut ssum = 0.0f32;

                for kv_pos in 0..kv_len {
                    let k_base = kv_pos * k_seq_stride + k_head_off;
                    let k_row = &k_data[k_base..k_base + d];

                    if kv_pos + 1 < kv_len {
                        prefetch_read(k_data[k_base + k_seq_stride..].as_ptr());
                    }

                    let score = if LEAN {
                        dot_f32(q_row, k_row) * scale
                    } else {
                        let alibi_bias = if max_bias > 0.0 {
                            slope * (kv_pos as f32 - (kv_len - 1) as f32)
                        } else {
                            0.0
                        };
                        let mut s = dot_f32(q_row, k_row) * scale_pre;
                        if do_softcap {
                            s = logit_softcap * s.tanh();
                        }
                        s + alibi_bias
                    };

                    let v_base = kv_pos * v_seq_stride + v_head_off;
                    let v_row = &v_data[v_base..v_base + d];

                    if kv_pos + 1 < kv_len {
                        prefetch_read(v_data[v_base + v_seq_stride..].as_ptr());
                    }

                    online_softmax_step(score, &mut m, &mut ssum, acc, |acc, w| {
                        for t in 0..d {
                            acc[t] += v_row[t].to_f64() as f32 * w;
                        }
                    });
                }

                let inv = if ssum > 0.0 { 1.0 / ssum } else { 0.0 };
                for t in 0..d {
                    out_chunk[t] = acc[t] * inv;
                }
            },
        );
    });

    Tensor::from_vec(out, (h_q, 1usize, d), &Device::Cpu)
}

// Prefill (q_len > 1). `LEAN` and `T` play the same roles as in `causal_decode`;
// the inner loop additionally honours the causal bound `kv_end` and the prefill
// ALiBi offset.
#[allow(clippy::too_many_arguments)]
fn causal_prefill<const LEAN: bool, T: WithDType>(
    q_data: &[T],
    k_data: &[T],
    v_data: &[T],
    s_q: usize,
    h_q: usize,
    h_kv: usize,
    h_v: usize,
    d: usize,
    kv_len: usize,
    scale: f32,
    kv_offset: usize,
    max_bias: f32,
    logit_softcap: f32,
) -> Result<Tensor> {
    let rk = h_q / h_kv;
    let rv = h_q / h_v;
    let n2 = 2_usize.pow((h_q as f32).log2().ceil() as u32);

    let (scale_pre, do_softcap) = if logit_softcap != 0.0 {
        (scale / logit_softcap, true)
    } else {
        (scale, false)
    };

    let q_seq_stride = h_q * d;
    let k_seq_stride = h_kv * d;
    let v_seq_stride = h_v * d;

    let mut out = vec![0f32; h_q * s_q * d];

    FLASH_ATTN_POOL.install(|| {
        out.par_chunks_mut(d)
            .with_min_len(64)
            .enumerate()
            .for_each_init(
                || vec![0f32; d],
                |acc, (row_idx, out_chunk)| {
                    let h_i = row_idx / s_q;
                    let q_pos = row_idx % s_q;

                    let slope = if !LEAN && max_bias > 0.0 {
                        2.0f32.powf(-max_bias * ((h_i + 1) as f32) / n2 as f32)
                    } else {
                        0.0
                    };

                    let k_head_off = (h_i / rk) * d;
                    let v_head_off = (h_i / rv) * d;

                    let q_base = q_pos * q_seq_stride + h_i * d;
                    let q_row = &q_data[q_base..q_base + d];

                    acc.fill(0.0);
                    let mut m = f32::NEG_INFINITY;
                    let mut ssum = 0.0f32;

                    let kv_end = (q_pos + kv_offset + 1).min(kv_len);

                    for kv_pos in 0..kv_end {
                        let k_base = kv_pos * k_seq_stride + k_head_off;
                        let k_row = &k_data[k_base..k_base + d];

                        if kv_pos + 1 < kv_end {
                            prefetch_read(k_data[k_base + k_seq_stride..].as_ptr());
                        }

                        let score = if LEAN {
                            dot_f32(q_row, k_row) * scale
                        } else {
                            let alibi_bias = if max_bias > 0.0 {
                                slope * (kv_pos as i64 - (q_pos + kv_offset) as i64) as f32
                            } else {
                                0.0
                            };
                            let mut s = dot_f32(q_row, k_row) * scale_pre;
                            if do_softcap {
                                s = logit_softcap * s.tanh();
                            }
                            s + alibi_bias
                        };

                        let v_base = kv_pos * v_seq_stride + v_head_off;
                        let v_row = &v_data[v_base..v_base + d];

                        if kv_pos + 1 < kv_end {
                            prefetch_read(v_data[v_base + v_seq_stride..].as_ptr());
                        }

                        online_softmax_step(score, &mut m, &mut ssum, acc, |acc, w| {
                            for t in 0..d {
                                acc[t] += v_row[t].to_f64() as f32 * w;
                            }
                        });
                    }

                    let inv = if ssum > 0.0 { 1.0 / ssum } else { 0.0 };
                    for t in 0..d {
                        out_chunk[t] = acc[t] * inv;
                    }
                },
            );
    });

    Tensor::from_vec(out, (h_q, s_q, d), &Device::Cpu)
}

// Interleaved KV decode.
// KV layout is contiguous (S, H_kv, 2*D) with K=[..,:D] and V=[..,D:2D],
// so one base pointer per position and one prefetch covers both.

/// Decode with interleaved KV cache. No ALiBi, no softcap.
#[allow(clippy::too_many_arguments)]
pub fn causal_decode_f16kv_interleaved(
    q_data: &[f32],
    kv_data: &[half::f16],
    head_stride: usize,
    h_q: usize,
    h_kv: usize,
    d: usize,
    kv_len: usize,
    scale: f32,
) -> Result<Tensor> {
    let rk = h_q / h_kv;

    let mut out = vec![0f32; h_q * d];

    // One task per kv head, computing all `rk` of its GQA query heads in a single
    // pass: the head-major cache makes the stream contiguous, and each K/V row is
    // read from memory once instead of once per query head.
    let process =
        |kv_h: usize, out_chunk: &mut [f32], acc: &mut [f32], m: &mut [f32], ssum: &mut [f32]| {
            let head_base = kv_h * head_stride;

            acc.fill(0.0);
            m.fill(f32::NEG_INFINITY);
            ssum.fill(0.0);

            // `f16-attn-dot`: narrow this kv-head's `rk` query rows to f16 ONCE here
            // (amortized over all `kv_len` positions), so the inner score is a pure
            // f16.f16 dot. Default build skips this entirely.
            #[cfg(feature = "f16-attn-dot")]
            let q_f16: Vec<half::f16> = {
                let base = kv_h * rk * d;
                q_data[base..base + rk * d]
                    .iter()
                    .map(|&x| half::f16::from_f32(x))
                    .collect()
            };

            for kv_pos in 0..kv_len {
                // K and V share a base pointer (adjacent in memory)
                let kv_base = head_base + kv_pos * 2 * d;
                let k_row = &kv_data[kv_base..kv_base + d];
                let v_row = &kv_data[kv_base + d..kv_base + 2 * d];

                // One prefetch loads both next K and V
                if kv_pos + 1 < kv_len {
                    prefetch_read(kv_data[kv_base + 2 * d..].as_ptr());
                }

                for r in 0..rk {
                    let h_i = kv_h * rk + r;
                    #[cfg(not(feature = "f16-attn-dot"))]
                    let score = {
                        let q_row = &q_data[h_i * d..(h_i + 1) * d];
                        dot_f32_f16(q_row, k_row) * scale
                    };
                    #[cfg(feature = "f16-attn-dot")]
                    let score = {
                        let _ = h_i;
                        dot_f16_f16(&q_f16[r * d..(r + 1) * d], k_row) * scale
                    };
                    let acc_r = &mut acc[r * d..(r + 1) * d];
                    online_softmax_step(score, &mut m[r], &mut ssum[r], acc_r, |acc, w| {
                        axpy_f16(acc, v_row, w);
                    });
                }
            }

            for r in 0..rk {
                let inv = if ssum[r] > 0.0 { 1.0 / ssum[r] } else { 0.0 };
                let acc_r = &acc[r * d..(r + 1) * d];
                let out_r = &mut out_chunk[r * d..(r + 1) * d];
                for t in 0..d {
                    out_r[t] = acc_r[t] * inv;
                }
            }
        };

    // Single-thread pool (the common Lambda 1-vCPU tier) has only ~h_kv tiny tasks,
    // so the rayon split/join machinery is pure per-call overhead - run serially.
    // Bit-identical to the parallel path; CANDLE_FLASH_SERIAL=0 forces rayon.
    static FLASH_SERIAL: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("CANDLE_FLASH_SERIAL")
            .map(|s| s != "0")
            .unwrap_or(true)
    });
    if *FLASH_SERIAL && FLASH_DECODE_POOL.current_num_threads() <= 1 {
        let mut acc = vec![0f32; rk * d];
        let mut m = vec![0f32; rk];
        let mut ssum = vec![0f32; rk];
        for (kv_h, out_chunk) in out.chunks_mut(rk * d).enumerate() {
            process(kv_h, out_chunk, &mut acc, &mut m, &mut ssum);
        }
    } else {
        FLASH_DECODE_POOL.install(|| {
            out.par_chunks_mut(rk * d).enumerate().for_each_init(
                || (vec![0f32; rk * d], vec![0f32; rk], vec![0f32; rk]),
                |(acc, m, ssum), (kv_h, out_chunk)| process(kv_h, out_chunk, acc, m, ssum),
            );
        });
    }

    Tensor::from_vec(out, (h_q, 1usize, d), &Device::Cpu)
}

/// Causal prefill over the f16 head-major interleaved KV cache (the same layout
/// [`causal_decode_f16kv_interleaved`] reads, so prefill and decode share one cache).
///
/// Structure matches [`causal_prefill_f32_lean`]: one task per (kv head, query
/// block), the `rk` GQA query heads sharing every K/V row load, KV-blocked softmax.
/// The QK dot widens the f16 K row in-register against the f32 query (`dot_f32_f16`);
/// the PV accumulator stays f32 (`axpy_f16` widens V in-register). Same dot strategy
/// as [`causal_decode_f16kv_interleaved`], including the `f16-attn-dot` variant.
#[allow(clippy::too_many_arguments)]
pub fn causal_prefill_f16kv_headmajor(
    q_data: &[f32],
    kv_data: &[half::f16],
    head_stride: usize,
    s_q: usize,
    h_q: usize,
    h_kv: usize,
    d: usize,
    kv_len: usize,
    scale: f32,
    kv_offset: usize,
) -> Result<Tensor> {
    let rk = h_q / h_kv;
    let q_seq_stride = h_q * d;

    const KB: usize = 128;
    const QB: usize = 32;
    let n_qblocks = s_q.div_ceil(QB);

    let mut out = vec![0f32; h_q * s_q * d];
    struct OutPtr(*mut f32);
    unsafe impl Sync for OutPtr {}
    let out_ptr = OutPtr(out.as_mut_ptr());

    FLASH_ATTN_POOL.install(|| {
        (0..h_kv * n_qblocks).into_par_iter().for_each_init(
            || {
                (
                    vec![0f32; rk * d],          // accumulators
                    vec![0f32; rk * KB],         // weights (scores in pass 1)
                    vec![f32::NEG_INFINITY; rk], // running max
                    vec![0f32; rk],              // running sum
                )
            },
            |(acc, w, m, ssum), task| {
                let kv_h = task / n_qblocks;
                let q0 = (task % n_qblocks) * QB;
                let q1 = (q0 + QB).min(s_q);
                let head_base = kv_h * head_stride;
                let p = &out_ptr;

                // `f16-attn-dot`: narrow this q block's rows to f16 so the QK dot is a
                // pure f16.f16 FMLA. Reused across q positions; the default build keeps
                // Q in f32 and the dot widens K in-register (`dot_f32_f16`).
                #[cfg(feature = "f16-attn-dot")]
                let mut qf16 = vec![half::f16::ZERO; rk * d];

                for q_pos in q0..q1 {
                    let q_pos_base = q_pos * q_seq_stride + kv_h * rk * d;
                    #[cfg(feature = "f16-attn-dot")]
                    for (o, &x) in qf16
                        .iter_mut()
                        .zip(&q_data[q_pos_base..q_pos_base + rk * d])
                    {
                        *o = half::f16::from_f32(x);
                    }

                    acc.fill(0.0);
                    m.fill(f32::NEG_INFINITY);
                    ssum.fill(0.0);
                    let kv_end = (q_pos + kv_offset + 1).min(kv_len);

                    for kv0 in (0..kv_end).step_by(KB) {
                        let kb = KB.min(kv_end - kv0);

                        // Pass 1: scores for all rk heads, each K row loaded once.
                        for j in 0..kb {
                            let kv_base = head_base + (kv0 + j) * 2 * d;
                            let k_row = &kv_data[kv_base..kv_base + d];
                            if j + 1 < kb {
                                prefetch_read(kv_data[kv_base + 2 * d..].as_ptr());
                            }
                            for r in 0..rk {
                                #[cfg(not(feature = "f16-attn-dot"))]
                                let dot = {
                                    let q_row =
                                        &q_data[q_pos_base + r * d..q_pos_base + (r + 1) * d];
                                    dot_f32_f16(q_row, k_row)
                                };
                                #[cfg(feature = "f16-attn-dot")]
                                let dot = dot_f16_f16(&qf16[r * d..(r + 1) * d], k_row);
                                w[r * KB + j] = dot * scale;
                            }
                        }

                        // Per head: one max/rescale per block, then batched exp.
                        for r in 0..rk {
                            let s = &mut w[r * KB..r * KB + kb];
                            let bmax = s.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                            if bmax > m[r] {
                                let f = (m[r] - bmax).exp();
                                if f > 0.0 {
                                    for a in &mut acc[r * d..(r + 1) * d] {
                                        *a *= f;
                                    }
                                    ssum[r] *= f;
                                } else {
                                    acc[r * d..(r + 1) * d].fill(0.0);
                                    ssum[r] = 0.0;
                                }
                                m[r] = bmax;
                            }
                            let mr = m[r];
                            ssum[r] += exp_sub_sum(s, mr);
                        }

                        // Pass 2: PV, each V row loaded once for all rk heads.
                        for j in 0..kb {
                            let kv_base = head_base + (kv0 + j) * 2 * d;
                            let v_row = &kv_data[kv_base + d..kv_base + 2 * d];
                            if j + 1 < kb {
                                prefetch_read(kv_data[kv_base + 2 * d + d..].as_ptr());
                            }
                            for r in 0..rk {
                                let wj = w[r * KB + j];
                                axpy_f16(&mut acc[r * d..(r + 1) * d], v_row, wj);
                            }
                        }
                    }

                    for r in 0..rk {
                        let h_i = kv_h * rk + r;
                        let inv = if ssum[r] > 0.0 { 1.0 / ssum[r] } else { 0.0 };
                        let acc_r = &acc[r * d..(r + 1) * d];
                        // SAFETY: each (h_i, q_pos) output row is written by exactly
                        // one task (kv heads and q blocks partition the rows).
                        let dst = unsafe {
                            std::slice::from_raw_parts_mut(p.0.add(h_i * s_q * d + q_pos * d), d)
                        };
                        for t in 0..d {
                            dst[t] = acc_r[t] * inv;
                        }
                    }
                }
            },
        );
    });

    Tensor::from_vec(out, (h_q, s_q, d), &Device::Cpu)
}
