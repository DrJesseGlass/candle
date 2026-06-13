// Index loops (for t in 0..d) are intentional for SIMD auto-vectorization.
#![allow(clippy::needless_range_loop)]

// Single-batch (B=1) causal attention using loop-bound masking.

use candle::{DType, Device, Result, Storage, Tensor, WithDType};
use rayon::prelude::*;

use super::online_softmax::online_softmax_step;
use super::standard::{FLASH_ATTN_POOL, FLASH_DECODE_POOL};
use super::{axpy_f16, dot_f32, dot_f32_f16};

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

    if q.dtype() == DType::F32 {
        let (q_g, q_l) = q.storage_and_layout();
        let q_data: &[f32] = match &*q_g {
            Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[q_l.start_offset()..],
            _ => candle::bail!("Expected CPU storage"),
        };
        let (k_g, k_l) = k.storage_and_layout();
        let k_data: &[f32] = match &*k_g {
            Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[k_l.start_offset()..],
            _ => candle::bail!("Expected CPU storage"),
        };
        let (v_g, v_l) = v.storage_and_layout();
        let v_data: &[f32] = match &*v_g {
            Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[v_l.start_offset()..],
            _ => candle::bail!("Expected CPU storage"),
        };

        let result = if s_q == 1 {
            causal_decode_f32(
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
            )
        } else {
            causal_prefill_f32(
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
            )
        };

        result.and_then(|t| t.unsqueeze(0))
    } else {
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

        let result = if s_q == 1 {
            causal_decode_generic(
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
            )
        } else {
            causal_prefill_generic(
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
            )
        };

        result.and_then(|t| t.unsqueeze(0))
    }
}

// f32 decode (q_len=1).
// Input layout is contiguous (1, S, H, D); we index past the batch dim.
// q[h] starts at h*D; k/v[pos, h] starts at pos*H_kv*D + h*D.

#[allow(clippy::too_many_arguments)]
fn causal_decode_f32(
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    h_q: usize,
    h_kv: usize,
    h_v: usize,
    d: usize,
    kv_len: usize,
    scale: f32,
    max_bias: f32,
    logit_softcap: f32,
) -> Result<Tensor> {
    // Dispatch to branchless fast path when no ALiBi / softcap
    if max_bias == 0.0 && logit_softcap == 0.0 {
        return causal_decode_f32_lean(q_data, k_data, v_data, h_q, h_kv, h_v, d, kv_len, scale);
    }

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

    FLASH_DECODE_POOL.install(|| {
        out.par_chunks_mut(d).enumerate().for_each_init(
            || vec![0f32; d],
            |acc, (h_i, out_chunk)| {
                let slope = 2.0f32.powf(-max_bias * ((h_i + 1) as f32) / n2 as f32);
                let k_head_off = (h_i / rk) * d;
                let v_head_off = (h_i / rv) * d;
                let q_row = &q_data[h_i * d..(h_i + 1) * d];

                acc.fill(0.0);
                let mut m = f32::NEG_INFINITY;
                let mut ssum = 0.0f32;

                for kv_pos in 0..kv_len {
                    let alibi_bias = slope * (kv_pos as f32 - (kv_len - 1) as f32);
                    let k_base = kv_pos * k_seq_stride + k_head_off;
                    let k_row = &k_data[k_base..k_base + d];

                    if kv_pos + 1 < kv_len {
                        prefetch_read(k_data[k_base + k_seq_stride..].as_ptr());
                    }

                    let mut score = dot_f32(q_row, k_row) * scale_pre;
                    if do_softcap {
                        score = logit_softcap * score.tanh();
                    }
                    score += alibi_bias;

                    let v_base = kv_pos * v_seq_stride + v_head_off;
                    let v_row = &v_data[v_base..v_base + d];

                    if kv_pos + 1 < kv_len {
                        prefetch_read(v_data[v_base + v_seq_stride..].as_ptr());
                    }

                    online_softmax_step(score, &mut m, &mut ssum, acc, |acc, w| {
                        for t in 0..d {
                            acc[t] += v_row[t] * w;
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

/// Lean decode: no ALiBi, no softcap. Zero branches in the inner KV loop.
/// This is the hot path for Qwen3, SmolLM3, and most standard LLMs.
#[allow(clippy::too_many_arguments)]
fn causal_decode_f32_lean(
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    h_q: usize,
    h_kv: usize,
    h_v: usize,
    d: usize,
    kv_len: usize,
    scale: f32,
) -> Result<Tensor> {
    let rk = h_q / h_kv;
    let rv = h_q / h_v;
    let k_seq_stride = h_kv * d;
    let v_seq_stride = h_v * d;

    let mut out = vec![0f32; h_q * d];

    FLASH_DECODE_POOL.install(|| {
        out.par_chunks_mut(d).enumerate().for_each_init(
            || vec![0f32; d],
            |acc, (h_i, out_chunk)| {
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

                    let score = dot_f32(q_row, k_row) * scale;

                    let v_base = kv_pos * v_seq_stride + v_head_off;
                    let v_row = &v_data[v_base..v_base + d];

                    if kv_pos + 1 < kv_len {
                        prefetch_read(v_data[v_base + v_seq_stride..].as_ptr());
                    }

                    online_softmax_step(score, &mut m, &mut ssum, acc, |acc, w| {
                        for t in 0..d {
                            acc[t] += v_row[t] * w;
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
    FLASH_DECODE_POOL.install(|| {
        out.par_chunks_mut(rk * d).enumerate().for_each_init(
            || (vec![0f32; rk * d], vec![0f32; rk], vec![0f32; rk]),
            |(acc, m, ssum), (kv_h, out_chunk)| {
                let head_base = kv_h * head_stride;

                acc.fill(0.0);
                m.fill(f32::NEG_INFINITY);
                ssum.fill(0.0);

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
                        let q_row = &q_data[h_i * d..(h_i + 1) * d];
                        let score = dot_f32_f16(q_row, k_row) * scale;
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
            },
        );
    });

    Tensor::from_vec(out, (h_q, 1usize, d), &Device::Cpu)
}

// f32 prefill (q_len > 1)

#[allow(clippy::too_many_arguments)]
fn causal_prefill_f32(
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
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
    // Dispatch to lean path for common case
    if max_bias == 0.0 && logit_softcap == 0.0 {
        return causal_prefill_f32_lean(
            q_data, k_data, v_data, s_q, h_q, h_kv, h_v, d, kv_len, scale, kv_offset,
        );
    }

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

                    let slope = if max_bias > 0.0 {
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
                        let alibi_bias = if max_bias > 0.0 {
                            slope * (kv_pos as i64 - (q_pos + kv_offset) as i64) as f32
                        } else {
                            0.0
                        };

                        let k_base = kv_pos * k_seq_stride + k_head_off;
                        let k_row = &k_data[k_base..k_base + d];

                        if kv_pos + 1 < kv_end {
                            prefetch_read(k_data[k_base + k_seq_stride..].as_ptr());
                        }

                        let mut score = dot_f32(q_row, k_row) * scale_pre;
                        if do_softcap {
                            score = logit_softcap * score.tanh();
                        }
                        score += alibi_bias;

                        let v_base = kv_pos * v_seq_stride + v_head_off;
                        let v_row = &v_data[v_base..v_base + d];

                        if kv_pos + 1 < kv_end {
                            prefetch_read(v_data[v_base + v_seq_stride..].as_ptr());
                        }

                        online_softmax_step(score, &mut m, &mut ssum, acc, |acc, w| {
                            for t in 0..d {
                                acc[t] += v_row[t] * w;
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

/// Lean prefill: no ALiBi, no softcap.
#[allow(clippy::too_many_arguments)]
fn causal_prefill_f32_lean(
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    s_q: usize,
    h_q: usize,
    h_kv: usize,
    h_v: usize,
    d: usize,
    kv_len: usize,
    scale: f32,
    kv_offset: usize,
) -> Result<Tensor> {
    let rk = h_q / h_kv;
    let rv = h_q / h_v;
    // The GQA sharing below assumes K and V heads coincide (true for all current users).
    if rk != rv || h_kv != h_v {
        return causal_prefill_f32_lean_per_row(
            q_data, k_data, v_data, s_q, h_q, h_kv, h_v, d, kv_len, scale, kv_offset,
        );
    }
    let q_seq_stride = h_q * d;
    let k_seq_stride = h_kv * d;

    // KV block size: K then V block each stream through L1 while the rk score/weight
    // strips and accumulators stay resident.
    const KB: usize = 128;
    // Queries per task: small enough for load balance across the causal triangle,
    // large enough to amortize task setup.
    const QB: usize = 32;
    let n_qblocks = s_q.div_ceil(QB);

    let mut out = vec![0f32; h_q * s_q * d];
    struct OutPtr(*mut f32);
    unsafe impl Sync for OutPtr {}
    let out_ptr = OutPtr(out.as_mut_ptr());

    FLASH_ATTN_POOL.install(|| {
        // One task per (kv head, query block): the rk query heads of a kv head share
        // every K/V row load; scores are computed per KV block so the softmax max
        // check and rescale run once per block per head instead of once per pair.
        (0..h_kv * n_qblocks).into_par_iter().for_each_init(
            || {
                (
                    vec![0f32; rk * d],  // accumulators
                    vec![0f32; rk * KB], // weights (scores in pass 1)
                    vec![f32::NEG_INFINITY; rk],
                    vec![0f32; rk],
                )
            },
            |(acc, w, m, ssum), task| {
                let kv_h = task / n_qblocks;
                let q0 = (task % n_qblocks) * QB;
                let q1 = (q0 + QB).min(s_q);
                let kv_head_off = kv_h * d;
                let p = &out_ptr;

                for q_pos in q0..q1 {
                    acc.fill(0.0);
                    m.fill(f32::NEG_INFINITY);
                    ssum.fill(0.0);
                    let kv_end = (q_pos + kv_offset + 1).min(kv_len);

                    for kv0 in (0..kv_end).step_by(KB) {
                        let kb = KB.min(kv_end - kv0);

                        // Pass 1: scores for all rk heads, each K row loaded once.
                        for j in 0..kb {
                            let k_base = (kv0 + j) * k_seq_stride + kv_head_off;
                            let k_row = &k_data[k_base..k_base + d];
                            if j + 1 < kb {
                                prefetch_read(k_data[k_base + k_seq_stride..].as_ptr());
                            }
                            for r in 0..rk {
                                let h_i = kv_h * rk + r;
                                let q_base = q_pos * q_seq_stride + h_i * d;
                                let q_row = &q_data[q_base..q_base + d];
                                w[r * KB + j] = dot_f32(q_row, k_row) * scale;
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
                            let mut bsum = 0.0f32;
                            for x in s.iter_mut() {
                                let e = (*x - mr).exp();
                                *x = e;
                                bsum += e;
                            }
                            ssum[r] += bsum;
                        }

                        // Pass 2: PV, each V row loaded once for all rk heads.
                        for j in 0..kb {
                            let v_base = (kv0 + j) * k_seq_stride + kv_head_off;
                            let v_row = &v_data[v_base..v_base + d];
                            if j + 1 < kb {
                                prefetch_read(v_data[v_base + k_seq_stride..].as_ptr());
                            }
                            for r in 0..rk {
                                let wj = w[r * KB + j];
                                let acc_r = &mut acc[r * d..(r + 1) * d];
                                for t in 0..d {
                                    acc_r[t] += v_row[t] * wj;
                                }
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
                            std::slice::from_raw_parts_mut(
                                p.0.add(h_i * s_q * d + q_pos * d),
                                d,
                            )
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

/// Causal prefill over the f16 head-major interleaved KV cache (the same layout
/// [`causal_decode_f16kv_interleaved`] reads, so prefill and decode share one cache).
///
/// Structure matches [`causal_prefill_f32_lean`]: one task per (kv head, query
/// block), the `rk` GQA query heads sharing every K/V row load, KV-blocked softmax.
/// Q is converted to f16 once per row so the QK dots run on the f16 FMLA path;
/// the PV accumulator stays f32 (`axpy_f16` widens V in-register).
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
                    vec![half::f16::ZERO; rk * d], // f16 copy of this position's q rows
                )
            },
            |(acc, w, m, ssum, qf16), task| {
                let kv_h = task / n_qblocks;
                let q0 = (task % n_qblocks) * QB;
                let q1 = (q0 + QB).min(s_q);
                let head_base = kv_h * head_stride;
                let p = &out_ptr;

                for q_pos in q0..q1 {
                    for r in 0..rk {
                        let h_i = kv_h * rk + r;
                        let q_base = q_pos * q_seq_stride + h_i * d;
                        for (o, &x) in qf16[r * d..(r + 1) * d]
                            .iter_mut()
                            .zip(&q_data[q_base..q_base + d])
                        {
                            *o = half::f16::from_f32(x);
                        }
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
                                w[r * KB + j] =
                                    dot_f32(&qf16[r * d..(r + 1) * d], k_row) * scale;
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
                            let mut bsum = 0.0f32;
                            for x in s.iter_mut() {
                                let e = (*x - mr).exp();
                                *x = e;
                                bsum += e;
                            }
                            ssum[r] += bsum;
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
                            std::slice::from_raw_parts_mut(
                                p.0.add(h_i * s_q * d + q_pos * d),
                                d,
                            )
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

/// Per-row fallback for the uncommon case of distinct K/V head counts.
#[allow(clippy::too_many_arguments)]
fn causal_prefill_f32_lean_per_row(
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    s_q: usize,
    h_q: usize,
    h_kv: usize,
    h_v: usize,
    d: usize,
    kv_len: usize,
    scale: f32,
    kv_offset: usize,
) -> Result<Tensor> {
    let rk = h_q / h_kv;
    let rv = h_q / h_v;
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

                        let score = dot_f32(q_row, k_row) * scale;

                        let v_base = kv_pos * v_seq_stride + v_head_off;
                        let v_row = &v_data[v_base..v_base + d];

                        if kv_pos + 1 < kv_end {
                            prefetch_read(v_data[v_base + v_seq_stride..].as_ptr());
                        }

                        online_softmax_step(score, &mut m, &mut ssum, acc, |acc, w| {
                            for t in 0..d {
                                acc[t] += v_row[t] * w;
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

// Generic fallback (non-f32)

#[allow(clippy::too_many_arguments)]
fn causal_decode_generic<T: WithDType>(
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

    FLASH_DECODE_POOL.install(|| {
        out.par_chunks_mut(d).enumerate().for_each_init(
            || vec![0f32; d],
            |acc, (h_i, out_chunk)| {
                let slope = if max_bias > 0.0 {
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
                    let alibi_bias = if max_bias > 0.0 {
                        slope * (kv_pos as f32 - (kv_len - 1) as f32)
                    } else {
                        0.0
                    };
                    let k_base = kv_pos * k_seq_stride + k_head_off;
                    let k_row = &k_data[k_base..k_base + d];
                    let mut s_val = dot_f32(q_row, k_row);
                    s_val *= scale_pre;
                    if do_softcap {
                        s_val = logit_softcap * s_val.tanh();
                    }
                    s_val += alibi_bias;

                    let v_base = kv_pos * v_seq_stride + v_head_off;
                    let v_row = &v_data[v_base..v_base + d];
                    online_softmax_step(s_val, &mut m, &mut ssum, acc, |acc, w| {
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

#[allow(clippy::too_many_arguments)]
fn causal_prefill_generic<T: WithDType>(
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
                    let slope = if max_bias > 0.0 {
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
                        let alibi_bias = if max_bias > 0.0 {
                            slope * (kv_pos as i64 - (q_pos + kv_offset) as i64) as f32
                        } else {
                            0.0
                        };
                        let k_base = kv_pos * k_seq_stride + k_head_off;
                        let k_row = &k_data[k_base..k_base + d];
                        let mut s_val = dot_f32(q_row, k_row);
                        s_val *= scale_pre;
                        if do_softcap {
                            s_val = logit_softcap * s_val.tanh();
                        }
                        s_val += alibi_bias;

                        let v_base = kv_pos * v_seq_stride + v_head_off;
                        let v_row = &v_data[v_base..v_base + d];
                        online_softmax_step(s_val, &mut m, &mut ssum, acc, |acc, w| {
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
