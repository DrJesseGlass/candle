#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! SIMD-optimized CPU flash attention using loop bounds instead of masking.
//!
//! Key optimizations:
//! - Uses loop bounds to implement causal attention (no mask tensor needed)
//! - SIMD-friendly F32 operations throughout
//! - Tiled computation for better cache locality
//! - Parallelized with Rayon (on native platforms only)
//!
//! This approach is more efficient than masked attention because:
//! 1. No mask allocation/computation needed
//! 2. Fewer memory accesses (no mask loads)
//! 3. Better branch prediction (loop bounds vs conditional on mask values)
//! 4. SIMD vectorization is easier with predictable loop bounds

use candle::{Device, Result, Storage, Tensor, WithDType};
use std::{f32, iter::Sum};

// ============================================================================
// WASM vs Native: Conditional compilation for parallelization
// ============================================================================

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

#[cfg(not(target_arch = "wasm32"))]
use rayon::ThreadPool;

#[cfg(not(target_arch = "wasm32"))]
use std::sync::LazyLock;

// Thread affinity only for macOS on native platforms
#[cfg(all(target_os = "macos", not(target_arch = "wasm32")))]
unsafe fn set_thread_affinity() {
    use libc::{pthread_set_qos_class_self_np, qos_class_t::QOS_CLASS_USER_INTERACTIVE};
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
}

#[cfg(not(all(target_os = "macos", not(target_arch = "wasm32"))))]
#[inline(always)]
unsafe fn set_thread_affinity() {}

// Native: Use Rayon thread pool
#[cfg(not(target_arch = "wasm32"))]
static FLASH_ATTN_POOL: LazyLock<ThreadPool> = LazyLock::new(|| {
    rayon::ThreadPoolBuilder::new()
        .start_handler(|_| unsafe {
            set_thread_affinity();
        })
        .build()
        .expect("Failed to build custom Rayon thread-pool for flash-attention")
});

/// SIMD-friendly chunk size for dot products
const DOT_CHUNK: usize = 8;

/// Size (in KV positions) processed by each inner-tile job
const TILE_KV: usize = 32;

#[inline]
fn vec_dot_simd<T: WithDType + Sum + Copy + std::ops::Mul<Output = T>>(a: &[T], b: &[T]) -> T {
    let mut sum = T::zero();
    let chunks = a.len() / DOT_CHUNK;

    // Process 8 elements at a time for better SIMD utilization
    for i in 0..chunks {
        let i_chunk = i * DOT_CHUNK;
        sum = sum
            + a[i_chunk] * b[i_chunk]
            + a[i_chunk + 1] * b[i_chunk + 1]
            + a[i_chunk + 2] * b[i_chunk + 2]
            + a[i_chunk + 3] * b[i_chunk + 3]
            + a[i_chunk + 4] * b[i_chunk + 4]
            + a[i_chunk + 5] * b[i_chunk + 5]
            + a[i_chunk + 6] * b[i_chunk + 6]
            + a[i_chunk + 7] * b[i_chunk + 7];
    }

    // Handle remainder
    for i in (chunks * DOT_CHUNK)..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// SIMD-optimized flash attention for CPU using loop bounds instead of masking.
///
/// **Key difference from standard implementation:**
/// Instead of computing attention for all KV positions and then masking,
/// we only compute attention for valid positions using loop bounds.
///
/// For causal attention at query position `q_pos`:
/// - Only compute attention for KV positions 0..=(q_pos + offset)
/// - This avoids computing and then masking out invalid positions
///
/// **Inputs:**
/// - `q`: (bs, seq, qhead, hidden)
/// - `k`: (bs, kv_seq, kv_head, hidden)
/// - `v`: (bs, kv_seq, kv_head, hidden)
/// - `softmax_scale`: scaling factor before softmax
/// - `is_causal`: whether to use causal attention (loop bounds)
///
/// **Output:** (bs, qhead, seq, hidden)
pub fn run_simd_flash_attn_cpu<T>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    is_causal: bool,
) -> Result<Tensor>
where
    T: WithDType + Sum + num_traits::real::Real,
{
    //Add debug logging
    eprintln!("🔍 SIMD Flash Attn: dtype={:?}, q_shape={:?}", q.dtype(), q.dims());

    // Extract CPU slices
    let (q_guard, q_layout) = q.storage_and_layout();
    let q_data: &[T] = if let Storage::Cpu(cpu) = &*q_guard {
        let data = cpu.as_slice::<T>()?;
        &data[q_layout.start_offset()..]
    } else {
        return Err(candle::Error::Msg("Expected CPU storage for q".into()));
    };

    let (k_guard, k_layout) = k.storage_and_layout();
    let k_data: &[T] = if let Storage::Cpu(cpu) = &*k_guard {
        let data = cpu.as_slice::<T>()?;
        &data[k_layout.start_offset()..]
    } else {
        return Err(candle::Error::Msg("Expected CPU storage for k".into()));
    };

    let (v_guard, v_layout) = v.storage_and_layout();
    let v_data: &[T] = if let Storage::Cpu(cpu) = &*v_guard {
        let data = cpu.as_slice::<T>()?;
        &data[v_layout.start_offset()..]
    } else {
        return Err(candle::Error::Msg("Expected CPU storage for v".into()));
    };

    let q_stride = q.stride();
    let k_stride = k.stride();
    let v_stride = v.stride();

    // Fast path for decode: q_len == 1
    if q.shape().dims()[1] == 1 {
        return simd_flash_attn_cpu_single_q(
            q_data,
            k_data,
            v_data,
            q.shape().dims(),
            k.shape().dims(),
            v.shape().dims(),
            q_stride,
            k_stride,
            v_stride,
            softmax_scale,
            is_causal,
        );
    }

    simd_flash_attn_cpu(
        q_data,
        k_data,
        v_data,
        q.shape().dims(),
        k.shape().dims(),
        v.shape().dims(),
        q_stride,
        k_stride,
        v_stride,
        softmax_scale,
        is_causal,
    )
}

/// Optimized path for decode: q_len == 1
/// Uses loop bounds to avoid computing attention for future positions
#[allow(clippy::too_many_arguments)]
fn simd_flash_attn_cpu_single_q<T: WithDType + Sum + num_traits::real::Real>(
    q_data: &[T],
    k_data: &[T],
    v_data: &[T],
    qshape: &[usize],
    kshape: &[usize],
    vshape: &[usize],
    qstride: &[usize],
    kstride: &[usize],
    vstride: &[usize],
    scale: f32,
    is_causal: bool,
) -> Result<Tensor> {
    let (b, _q_len, h, d) = (qshape[0], qshape[1], qshape[2], qshape[3]);
    let kv_len = kshape[1];
    let k_h = kshape[2];
    let v_h = vshape[2];
    let rk2 = h / k_h;
    let rv2 = h / v_h;
    let dv = d;

    let mut out = vec![0f32; b * h * dv];

    // For single query (decode), the valid KV range is 0..kv_len
    // (causal attention doesn't restrict since we're at the last position)
    let kv_tiles = kv_len.div_ceil(TILE_KV);

    // WASM: Single-threaded processing
    #[cfg(target_arch = "wasm32")]
    {
        for (row_idx, out_chunk) in out.chunks_mut(dv).enumerate() {
            process_single_q_row(
                row_idx, out_chunk, b, h, d, dv, kv_len, kv_tiles, rk2, rv2,
                q_data, k_data, v_data, qstride, kstride, vstride, scale,
            );
        }
    }

    // Native: Parallel processing with Rayon
    #[cfg(not(target_arch = "wasm32"))]
    {
        FLASH_ATTN_POOL.install(|| {
            out.par_chunks_mut(dv)
                .with_min_len(64)
                .enumerate()
                .for_each(|(row_idx, out_chunk)| {
                    process_single_q_row(
                        row_idx, out_chunk, b, h, d, dv, kv_len, kv_tiles, rk2, rv2,
                        q_data, k_data, v_data, qstride, kstride, vstride, scale,
                    );
                });
        });
    }

    let out_shape = (b, h, 1usize, dv);
    Tensor::from_vec(out, out_shape, &Device::Cpu)
}

#[allow(clippy::too_many_arguments)]
fn process_single_q_row<T: WithDType + Sum + num_traits::real::Real>(
    row_idx: usize,
    out_chunk: &mut [f32],
    b: usize,
    h: usize,
    d: usize,
    dv: usize,
    kv_len: usize,
    kv_tiles: usize,
    rk2: usize,
    rv2: usize,
    q_data: &[T],
    k_data: &[T],
    v_data: &[T],
    qstride: &[usize],
    kstride: &[usize],
    vstride: &[usize],
    scale: f32,
) {
    let b_i = row_idx / h;
    let h_i = row_idx % h;

    let k_head = h_i / rk2;
    let v_head = h_i / rv2;

    // Gather Q row once
    let q_base = b_i * qstride[0] + h_i * qstride[2];
    let mut q_row: Vec<T> = Vec::with_capacity(d);
    for di in 0..d {
        q_row.push(q_data[q_base + di * qstride[3]]);
    }

    // Process in tiles for better cache locality
    let mut vkq = vec![0f32; dv];
    let mut s_tot = 0.0f32;
    let mut m = f32::NEG_INFINITY;

    for tile_idx in 0..kv_tiles {
        let tile_start = tile_idx * TILE_KV;
        let tile_end = (tile_start + TILE_KV).min(kv_len);

        let mut k_row: Vec<T> = Vec::with_capacity(d);

        for kv_pos in tile_start..tile_end {
            // K row
            let k_base = b_i * kstride[0] + kv_pos * kstride[1] + k_head * kstride[2];
            k_row.clear();
            for di in 0..d {
                k_row.push(k_data[k_base + di * kstride[3]]);
            }

            // Compute attention score
            let s_val = vec_dot_simd::<T>(&q_row, &k_row).to_f64() as f32 * scale;

            // Online softmax
            let m_old = m;
            let (ms, vs) = if s_val > m {
                m = s_val;
                let ms = (m_old - m).exp();
                for v in vkq.iter_mut() {
                    *v *= ms;
                }
                (ms, 1.0f32)
            } else {
                (1.0f32, (s_val - m).exp())
            };

            // V row
            let v_base = b_i * vstride[0] + kv_pos * vstride[1] + v_head * vstride[2];
            for d_i in 0..dv {
                vkq[d_i] += v_data[v_base + d_i * vstride[3]].to_f64() as f32 * vs;
            }

            s_tot = s_tot * ms + vs;
        }
    }

    // Final normalization
    let inv_s = 1.0 / s_tot;
    for (out_val, vkq_val) in out_chunk.iter_mut().zip(vkq.iter()) {
        *out_val = *vkq_val * inv_s;
    }
}

/// Main SIMD flash attention using loop bounds for causal masking
#[allow(clippy::too_many_arguments)]
fn simd_flash_attn_cpu<T: WithDType + Sum + num_traits::real::Real>(
    q_data: &[T],
    k_data: &[T],
    v_data: &[T],
    qshape: &[usize],
    kshape: &[usize],
    vshape: &[usize],
    qstride: &[usize],
    kstride: &[usize],
    vstride: &[usize],
    scale: f32,
    is_causal: bool,
) -> Result<Tensor> {
    let (b, q_len, h, d) = (qshape[0], qshape[1], qshape[2], qshape[3]);
    let kv_len = kshape[1];
    let k_h = kshape[2];
    let v_h = vshape[2];
    let rk2 = h / k_h;
    let rv2 = h / v_h;
    let dv = d;

    let mut out = vec![0f32; b * q_len * h * dv];

    // WASM: Single-threaded processing
    #[cfg(target_arch = "wasm32")]
    {
        for (row_idx, out_chunk) in out.chunks_mut(dv).enumerate() {
            process_attn_row(
                row_idx, out_chunk, b, q_len, h, d, dv, kv_len, rk2, rv2,
                q_data, k_data, v_data, qstride, kstride, vstride, scale, is_causal,
            );
        }
    }

    // Native: Parallel processing with Rayon
    #[cfg(not(target_arch = "wasm32"))]
    {
        FLASH_ATTN_POOL.install(|| {
            out.par_chunks_mut(dv)
                .with_min_len(64)
                .enumerate()
                .for_each(|(row_idx, out_chunk)| {
                    process_attn_row(
                        row_idx, out_chunk, b, q_len, h, d, dv, kv_len, rk2, rv2,
                        q_data, k_data, v_data, qstride, kstride, vstride, scale, is_causal,
                    );
                });
        });
    }

    let out_shape = (b, h, q_len, dv);
    Tensor::from_vec(out, out_shape, &Device::Cpu)
}

#[allow(clippy::too_many_arguments)]
fn process_attn_row<T: WithDType + Sum + num_traits::real::Real>(
    row_idx: usize,
    out_chunk: &mut [f32],
    b: usize,
    q_len: usize,
    h: usize,
    d: usize,
    dv: usize,
    kv_len: usize,
    rk2: usize,
    rv2: usize,
    q_data: &[T],
    k_data: &[T],
    v_data: &[T],
    qstride: &[usize],
    kstride: &[usize],
    vstride: &[usize],
    scale: f32,
    is_causal: bool,
) {
    let rows_per_batch = h * q_len;
    let b_i = row_idx / rows_per_batch;
    let rem = row_idx % rows_per_batch;
    let h_i = rem / q_len;
    let q_pos = rem % q_len;

    let k_head = h_i / rk2;
    let v_head = h_i / rv2;

    let mut vkq = vec![0f32; dv];
    let mut s = 0.0f32;
    let mut m = f32::NEG_INFINITY;

    let mut q_row: Vec<T> = Vec::with_capacity(d);
    let mut k_row: Vec<T> = Vec::with_capacity(d);

    // Gather Q row
    let q_base = b_i * qstride[0] + q_pos * qstride[1] + h_i * qstride[2];
    for di in 0..d {
        q_row.push(q_data[q_base + di * qstride[3]]);
    }

    // OPTIMIZATION: Use loop bounds for causal attention instead of mask
    // For causal attention, only compute up to current position
    let kv_end = if is_causal {
        (q_pos + 1).min(kv_len)
    } else {
        kv_len
    };

    // Iterate only over valid KV positions
    for kv_pos in 0..kv_end {
        // K row
        let k_base = b_i * kstride[0] + kv_pos * kstride[1] + k_head * kstride[2];
        k_row.clear();
        for di in 0..d {
            k_row.push(k_data[k_base + di * kstride[3]]);
        }

        // Compute attention score with SIMD-friendly dot product
        let s_val = vec_dot_simd::<T>(&q_row, &k_row);
        let s_val_f32 = s_val.to_f64() as f32 * scale;

        // Online softmax
        let m_old = m;
        let (ms, vs) = if s_val_f32 > m {
            m = s_val_f32;
            let ms = (m_old - m).exp();
            for v in vkq.iter_mut() {
                *v *= ms;
            }
            (ms, 1.0f32)
        } else {
            (1.0f32, (s_val_f32 - m).exp())
        };

        // V row
        let v_base = b_i * vstride[0] + kv_pos * vstride[1] + v_head * vstride[2];
        for d_i in 0..dv {
            vkq[d_i] += v_data[v_base + d_i * vstride[3]].to_f64() as f32 * vs;
        }

        s = s * ms + vs;
    }

    // Final normalization
    let inv_s = 1.0 / s;
    for (out_val, vkq_val) in out_chunk.iter_mut().zip(vkq.iter()) {
        *out_val = *vkq_val * inv_s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::DType;

    #[test]
    fn test_simd_flash_attn_causal() -> Result<()> {
        let dev = Device::Cpu;
        let (b, seq, h, d) = (1, 4, 2, 8);

        // Create test tensors
        let q = Tensor::randn(0f32, 1.0, (b, seq, h, d), &dev)?.to_dtype(DType::F32)?;
        let k = Tensor::randn(0f32, 1.0, (b, seq, h, d), &dev)?.to_dtype(DType::F32)?;
        let v = Tensor::randn(0f32, 1.0, (b, seq, h, d), &dev)?.to_dtype(DType::F32)?;

        let scale = 1.0 / (d as f32).sqrt();

        // Run SIMD flash attention with causal
        let out = run_simd_flash_attn_cpu::<f32>(&q, &k, &v, scale, true)?;

        // Output should have shape (b, h, seq, d)
        assert_eq!(out.dims(), &[b, h, seq, d]);

        Ok(())
    }

    #[test]
    fn test_simd_flash_attn_non_causal() -> Result<()> {
        let dev = Device::Cpu;
        let (b, seq, h, d) = (1, 4, 2, 8);

        let q = Tensor::randn(0f32, 1.0, (b, seq, h, d), &dev)?.to_dtype(DType::F32)?;
        let k = Tensor::randn(0f32, 1.0, (b, seq, h, d), &dev)?.to_dtype(DType::F32)?;
        let v = Tensor::randn(0f32, 1.0, (b, seq, h, d), &dev)?.to_dtype(DType::F32)?;

        let scale = 1.0 / (d as f32).sqrt();

        let out = run_simd_flash_attn_cpu::<f32>(&q, &k, &v, scale, false)?;

        assert_eq!(out.dims(), &[b, h, seq, d]);

        Ok(())
    }
}