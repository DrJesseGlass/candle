//! Microbenchmark for Q4_K GEMV dot kernels (aarch64).
//!
//! Benches the in-tree kernel against experimental variants over a matrix larger
//! than L2 so the numbers reflect DRAM streaming, matching the decode (m=1) GEMV.
//!
//! Run: cargo run --release --example qbench
#![allow(clippy::needless_range_loop)]

use candle_core::quantized::{k_quants, GgmlDType, QTensor};
use candle_core::{Device, Result, Tensor};
use half::f16;
use std::time::Instant;

const QK_K: usize = 256;

// Byte-compatible mirrors of the k_quants blocks (fields there are pub(crate)).
#[repr(C)]
#[derive(Clone, Copy)]
struct Q4K {
    d: f16,
    dmin: f16,
    scales: [u8; 12],
    qs: [u8; QK_K / 2],
}

#[repr(C)]
struct Q8K {
    d: f32,
    qs: [i8; QK_K],
    bsums: [i16; QK_K / 16],
}

const _: () = assert!(std::mem::size_of::<Q4K>() == 144);
const _: () = assert!(std::mem::size_of::<Q8K>() == 292);

#[cfg(target_arch = "aarch64")]
mod kernels {
    use super::{Q4K, Q8K, QK_K};
    use std::arch::aarch64::*;

    #[inline(always)]
    unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let mut acc = acc;
        core::arch::asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack, preserves_flags),
        );
        acc
    }

    #[inline(always)]
    unsafe fn prefetch(ptr: *const u8) {
        core::arch::asm!(
            "prfm pldl1strm, [{p}]",
            p = in(reg) ptr,
            options(nostack, preserves_flags)
        );
    }

    /// Scale/min preamble shared by all variants: returns (d, scales[8] in low
    /// bytes, and the -dmin*Σ(mins*bsums) contribution).
    #[inline(always)]
    unsafe fn preamble(x: &Q4K, y: &Q8K, scales: &mut [u8; 16]) -> (f32, f32) {
        const KMASK1: u32 = 0x3f3f3f3f;
        const KMASK2: u32 = 0x0f0f0f0f;
        const KMASK3: u32 = 0x03030303;
        let d = y.d * x.d.to_f32();
        let dmin = y.d * x.dmin.to_f32();

        let q8sums = vpaddq_s16(vld1q_s16(y.bsums.as_ptr()), vld1q_s16(y.bsums.as_ptr().add(8)));

        let mut utmp = [0u32; 4];
        std::ptr::copy_nonoverlapping(x.scales.as_ptr(), utmp.as_mut_ptr() as *mut u8, 12);

        let mins8 = vld1_u32(
            [
                utmp[1] & KMASK1,
                ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4),
            ]
            .as_ptr(),
        );
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[0] &= KMASK1;

        let mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
        let prod = vaddq_s32(
            vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
            vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)),
        );
        let minus = dmin * vaddvq_s32(prod) as f32;

        std::ptr::copy_nonoverlapping(utmp.as_ptr() as *const u8, scales.as_mut_ptr(), 16);
        (d, minus)
    }

    /// v0: structure of the current in-tree kernel (per-32 horizontal reduce).
    pub unsafe fn v0_current(n: usize, xs: &[Q4K], ys: &[Q8K]) -> f32 {
        let m4b = vdupq_n_u8(0xF);
        let mut sumf = 0f32;
        let mut scales = [0u8; 16];
        for i in 0..n / QK_K {
            let (x, y) = (&xs[i], &ys[i]);
            let (d, minus) = preamble(x, y, &mut scales);
            sumf -= minus;

            let mut q4 = x.qs.as_ptr();
            let mut q8 = y.qs.as_ptr();
            let mut sumi1 = 0i32;
            let mut sumi2 = 0i32;
            for j in 0..QK_K / 64 {
                let q4bits = vld1q_u8_x2(q4);
                q4 = q4.add(32);
                let q8bytes = vld1q_s8_x2(q8);
                q8 = q8.add(32);
                let p0 = sdot(
                    vdupq_n_s32(0),
                    vreinterpretq_s8_u8(vandq_u8(q4bits.0, m4b)),
                    q8bytes.0,
                );
                let p01 = sdot(p0, vreinterpretq_s8_u8(vandq_u8(q4bits.1, m4b)), q8bytes.1);
                sumi1 += vaddvq_s32(p01) * scales[2 * j] as i32;

                let q8bytes = vld1q_s8_x2(q8);
                q8 = q8.add(32);
                let p2 = sdot(
                    vdupq_n_s32(0),
                    vreinterpretq_s8_u8(vshrq_n_u8(q4bits.0, 4)),
                    q8bytes.0,
                );
                let p23 = sdot(p2, vreinterpretq_s8_u8(vshrq_n_u8(q4bits.1, 4)), q8bytes.1);
                sumi2 += vaddvq_s32(p23) * scales[2 * j + 1] as i32;
            }
            sumf += d * (sumi1 + sumi2) as f32;
        }
        sumf
    }

    /// v1: vector scale accumulation — one horizontal reduce per superblock
    /// instead of eight.
    pub unsafe fn v1_vscale(n: usize, xs: &[Q4K], ys: &[Q8K]) -> f32 {
        let m4b = vdupq_n_u8(0xF);
        let mut sumf = 0f32;
        let mut scales = [0u8; 16];
        for i in 0..n / QK_K {
            let (x, y) = (&xs[i], &ys[i]);
            let (d, minus) = preamble(x, y, &mut scales);
            sumf -= minus;

            let mut q4 = x.qs.as_ptr();
            let mut q8 = y.qs.as_ptr();
            let mut acc = vdupq_n_s32(0);
            for j in 0..QK_K / 64 {
                let q4bits = vld1q_u8_x2(q4);
                q4 = q4.add(32);
                let q8bytes = vld1q_s8_x2(q8);
                q8 = q8.add(32);
                let p0 = sdot(
                    vdupq_n_s32(0),
                    vreinterpretq_s8_u8(vandq_u8(q4bits.0, m4b)),
                    q8bytes.0,
                );
                let p01 = sdot(p0, vreinterpretq_s8_u8(vandq_u8(q4bits.1, m4b)), q8bytes.1);
                acc = vmlaq_n_s32(acc, p01, scales[2 * j] as i32);

                let q8bytes = vld1q_s8_x2(q8);
                q8 = q8.add(32);
                let p2 = sdot(
                    vdupq_n_s32(0),
                    vreinterpretq_s8_u8(vshrq_n_u8(q4bits.0, 4)),
                    q8bytes.0,
                );
                let p23 = sdot(p2, vreinterpretq_s8_u8(vshrq_n_u8(q4bits.1, 4)), q8bytes.1);
                acc = vmlaq_n_s32(acc, p23, scales[2 * j + 1] as i32);
            }
            sumf += d * vaddvq_s32(acc) as f32;
        }
        sumf
    }

    /// v2: v1 + two superblocks in flight (independent load/dot chains for MLP).
    pub unsafe fn v2_unroll2(n: usize, xs: &[Q4K], ys: &[Q8K]) -> f32 {
        let nb = n / QK_K;
        let mut sumf = 0f32;
        let mut i = 0;
        while i + 2 <= nb {
            sumf += pair(&xs[i], &ys[i], &xs[i + 1], &ys[i + 1], false);
            i += 2;
        }
        while i < nb {
            sumf += v1_vscale(QK_K, &xs[i..i + 1], &ys[i..i + 1]);
            i += 1;
        }
        sumf
    }

    /// v3: v2 + software prefetch two superblocks ahead.
    pub unsafe fn v3_pf(n: usize, xs: &[Q4K], ys: &[Q8K]) -> f32 {
        let nb = n / QK_K;
        let mut sumf = 0f32;
        let mut i = 0;
        while i + 2 <= nb {
            sumf += pair(&xs[i], &ys[i], &xs[i + 1], &ys[i + 1], true);
            i += 2;
        }
        while i < nb {
            sumf += v1_vscale(QK_K, &xs[i..i + 1], &ys[i..i + 1]);
            i += 1;
        }
        sumf
    }

    #[inline(always)]
    unsafe fn pair(x0: &Q4K, y0: &Q8K, x1: &Q4K, y1: &Q8K, pf: bool) -> f32 {
        let m4b = vdupq_n_u8(0xF);
        let mut scales0 = [0u8; 16];
        let mut scales1 = [0u8; 16];
        let (d0, minus0) = preamble(x0, y0, &mut scales0);
        let (d1, minus1) = preamble(x1, y1, &mut scales1);

        if pf {
            // Two superblocks ahead: x1 + 144 .. (rows are contiguous in the GEMV).
            let ahead = (x1 as *const Q4K as *const u8).add(144);
            prefetch(ahead);
            prefetch(ahead.add(64));
            prefetch(ahead.add(128));
            prefetch(ahead.add(144));
            prefetch(ahead.add(208));
        }

        let mut q4a = x0.qs.as_ptr();
        let mut q8a = y0.qs.as_ptr();
        let mut q4b = x1.qs.as_ptr();
        let mut q8b = y1.qs.as_ptr();
        let mut acca = vdupq_n_s32(0);
        let mut accb = vdupq_n_s32(0);
        for j in 0..QK_K / 64 {
            let qa_bits = vld1q_u8_x2(q4a);
            q4a = q4a.add(32);
            let qb_bits = vld1q_u8_x2(q4b);
            q4b = q4b.add(32);

            let a8_lo = vld1q_s8_x2(q8a);
            q8a = q8a.add(32);
            let b8_lo = vld1q_s8_x2(q8b);
            q8b = q8b.add(32);

            let pa = sdot(
                sdot(
                    vdupq_n_s32(0),
                    vreinterpretq_s8_u8(vandq_u8(qa_bits.0, m4b)),
                    a8_lo.0,
                ),
                vreinterpretq_s8_u8(vandq_u8(qa_bits.1, m4b)),
                a8_lo.1,
            );
            let pb = sdot(
                sdot(
                    vdupq_n_s32(0),
                    vreinterpretq_s8_u8(vandq_u8(qb_bits.0, m4b)),
                    b8_lo.0,
                ),
                vreinterpretq_s8_u8(vandq_u8(qb_bits.1, m4b)),
                b8_lo.1,
            );
            acca = vmlaq_n_s32(acca, pa, scales0[2 * j] as i32);
            accb = vmlaq_n_s32(accb, pb, scales1[2 * j] as i32);

            let a8_hi = vld1q_s8_x2(q8a);
            q8a = q8a.add(32);
            let b8_hi = vld1q_s8_x2(q8b);
            q8b = q8b.add(32);

            let pa = sdot(
                sdot(
                    vdupq_n_s32(0),
                    vreinterpretq_s8_u8(vshrq_n_u8(qa_bits.0, 4)),
                    a8_hi.0,
                ),
                vreinterpretq_s8_u8(vshrq_n_u8(qa_bits.1, 4)),
                a8_hi.1,
            );
            let pb = sdot(
                sdot(
                    vdupq_n_s32(0),
                    vreinterpretq_s8_u8(vshrq_n_u8(qb_bits.0, 4)),
                    b8_hi.0,
                ),
                vreinterpretq_s8_u8(vshrq_n_u8(qb_bits.1, 4)),
                b8_hi.1,
            );
            acca = vmlaq_n_s32(acca, pa, scales0[2 * j + 1] as i32);
            accb = vmlaq_n_s32(accb, pb, scales1[2 * j + 1] as i32);
        }
        d0 * vaddvq_s32(acca) as f32 - minus0 + d1 * vaddvq_s32(accb) as f32 - minus1
    }
}

fn bench_gemv<F: Fn(usize, &[Q4K], &[Q8K]) -> f32 + Sync>(
    name: &str,
    rows: usize,
    k: usize,
    w: &[Q4K],
    y: &[Q8K],
    f: F,
) {
    let bpr = k / QK_K;
    let bytes_per_row = bpr * std::mem::size_of::<Q4K>();
    // warmup + timed
    for pass in 0..2 {
        let iters = if pass == 0 { 2 } else { 10 };
        let t = Instant::now();
        let mut acc = 0f64;
        for _ in 0..iters {
            for r in 0..rows {
                acc += f(k, &w[r * bpr..(r + 1) * bpr], y) as f64;
            }
        }
        let dt = t.elapsed().as_secs_f64();
        std::hint::black_box(acc);
        if pass == 1 {
            let gbs = (rows * bytes_per_row * iters) as f64 / dt / 1e9;
            println!("  {name:12} 1T: {gbs:6.2} GB/s");
        }
    }
    // 4-thread row-parallel (decode-pool shaped)
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let iters = 10;
    let t = Instant::now();
    pool.install(|| {
        use rayon::prelude::*;
        for _ in 0..iters {
            let s: f64 = (0..rows)
                .into_par_iter()
                .with_min_len(128)
                .map(|r| f(k, &w[r * bpr..(r + 1) * bpr], y) as f64)
                .sum();
            std::hint::black_box(s);
        }
    });
    let dt = t.elapsed().as_secs_f64();
    let gbs = (rows * bytes_per_row * iters) as f64 / dt / 1e9;
    println!("  {name:12} 4T: {gbs:6.2} GB/s");
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let (rows, k) = (65536usize, 1024usize); // ~37 MB of Q4_K — larger than L2
    let bpr = k / QK_K;

    println!("generating {rows}x{k} Q4_K matrix...");
    let wt = Tensor::randn(0f32, 1f32, (rows, k), &dev)?;
    let wq = QTensor::quantize(&wt, GgmlDType::Q4K)?;
    let wbytes = wq.data()?.into_owned();
    let w: &[Q4K] = unsafe { std::slice::from_raw_parts(wbytes.as_ptr() as *const Q4K, rows * bpr) };

    let at = Tensor::randn(0f32, 1f32, (1, k), &dev)?;
    let aq = QTensor::quantize(&at, GgmlDType::Q8K)?;
    let abytes = aq.data()?.into_owned();
    let y: &[Q8K] = unsafe { std::slice::from_raw_parts(abytes.as_ptr() as *const Q8K, bpr) };

    // In-tree reference via the public trait, over the same bytes.
    let w_tree: &[k_quants::BlockQ4K] =
        unsafe { std::slice::from_raw_parts(wbytes.as_ptr() as *const _, rows * bpr) };
    let y_tree: &[k_quants::BlockQ8K] =
        unsafe { std::slice::from_raw_parts(abytes.as_ptr() as *const _, bpr) };

    #[cfg(target_arch = "aarch64")]
    {
        use k_quants::GgmlType;
        // correctness: all variants vs in-tree on a sample of rows
        let mut max_err = [0f32; 4];
        for r in (0..rows).step_by(997) {
            let xs = &w[r * bpr..(r + 1) * bpr];
            let reference: f32 =
                k_quants::BlockQ4K::vec_dot(k, &w_tree[r * bpr..(r + 1) * bpr], y_tree);
            let outs = unsafe {
                [
                    kernels::v0_current(k, xs, y),
                    kernels::v1_vscale(k, xs, y),
                    kernels::v2_unroll2(k, xs, y),
                    kernels::v3_pf(k, xs, y),
                ]
            };
            for (e, o) in max_err.iter_mut().zip(outs) {
                *e = e.max((o - reference).abs() / reference.abs().max(1.0));
            }
        }
        println!("max rel err vs in-tree: v0={:.2e} v1={:.2e} v2={:.2e} v3={:.2e}", max_err[0], max_err[1], max_err[2], max_err[3]);

        println!("benchmarks (matrix streamed, decode-GEMV shaped):");
        bench_gemv("in-tree", rows, k, w, y, |n, xs, _| {
            let xs_t: &[k_quants::BlockQ4K] =
                unsafe { std::slice::from_raw_parts(xs.as_ptr() as *const _, xs.len()) };
            k_quants::BlockQ4K::vec_dot(n, xs_t, y_tree)
        });
        bench_gemv("v0_current", rows, k, w, y, |n, xs, ys| unsafe {
            kernels::v0_current(n, xs, ys)
        });
        bench_gemv("v1_vscale", rows, k, w, y, |n, xs, ys| unsafe {
            kernels::v1_vscale(n, xs, ys)
        });
        bench_gemv("v2_unroll2", rows, k, w, y, |n, xs, ys| unsafe {
            kernels::v2_unroll2(n, xs, ys)
        });
        bench_gemv("v3_pf", rows, k, w, y, |n, xs, ys| unsafe {
            kernels::v3_pf(n, xs, ys)
        });
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (w, y, w_tree, y_tree);
        println!("aarch64 only");
    }
    Ok(())
}
