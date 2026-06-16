use super::k_quants::{
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ4_0, BlockQ5K, BlockQ6K, BlockQ8K, BlockQ8_0, QK8_0, QK_K,
};
use super::repack::BlockQ4Kx8;
use byteorder::{ByteOrder, LittleEndian};

#[allow(unused_imports)]
#[cfg(target_arch = "arm")]
use core::arch::arm::*;

#[allow(unused_imports)]
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

// Dot of two int8x16 vectors, grouped into four int32 lane-sums. Callers always reduce
// across lanes afterwards, so the lane grouping is irrelevant — only the total matters.
// Uses the hardware ARMv8.2 DotProd `SDOT` instruction when available (it is on Apple
// Silicon / any `-Ctarget-feature=+dotprod` build), replacing ~5 emulation ops with one;
// otherwise falls back to the widening-multiply emulation.
#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
#[inline(always)]
unsafe fn vdotq_s32(a: int8x16_t, b: int8x16_t) -> int32x4_t {
    // The `vdotq_s32` std intrinsic is still unstable (stdarch_neon_dotprod), so emit the
    // SDOT instruction directly. Safe on stable because `dotprod` is in the target features.
    let mut acc = vdupq_n_s32(0);
    core::arch::asm!(
        "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
        acc = inout(vreg) acc,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack, preserves_flags),
    );
    acc
}

#[cfg(not(all(target_arch = "aarch64", target_feature = "dotprod")))]
#[inline(always)]
unsafe fn vdotq_s32(a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    let p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1))
}

#[inline(always)]
pub(crate) fn vec_dot_q4_0_q8_0(n: usize, xs: &[BlockQ4_0], ys: &[BlockQ8_0]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK8_0),
        "vec_dot_q4_0_q8_0: {n} is not divisible by {QK8_0}"
    );
    let nb = n / QK8_0;
    unsafe {
        let mut sumv0 = vdupq_n_f32(0.0f32);
        for i in 0..nb {
            let x0 = &xs[i];
            let y0 = &ys[i];

            let m4b = vdupq_n_u8(0x0F);
            let s8b = vdupq_n_s8(0x8);

            let v0_0 = vld1q_u8(x0.qs.as_ptr());

            // 4-bit -> 8-bit
            let v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
            let v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));

            // sub 8
            let v0_0ls = vsubq_s8(v0_0l, s8b);
            let v0_0hs = vsubq_s8(v0_0h, s8b);

            // load y
            let v1_0l = vld1q_s8(y0.qs.as_ptr());
            let v1_0h = vld1q_s8(y0.qs.as_ptr().add(16));

            let pl0 = vdotq_s32(v0_0ls, v1_0l);
            let ph0 = vdotq_s32(v0_0hs, v1_0h);
            sumv0 = vmlaq_n_f32(
                sumv0,
                vcvtq_f32_s32(vaddq_s32(pl0, ph0)),
                x0.d.to_f32() * y0.d.to_f32(),
            );
        }
        vaddvq_f32(sumv0)
    }
}

#[inline(always)]
pub(crate) fn vec_dot_q8_0_q8_0(n: usize, xs: &[BlockQ8_0], ys: &[BlockQ8_0]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK8_0),
        "vec_dot_q8_0_q8_0: {n} is not divisible by {QK8_0}"
    );
    let nb = n / QK8_0;
    unsafe {
        let mut sumv0 = vdupq_n_f32(0.0f32);
        for i in 0..nb {
            let x0 = &xs[i];
            let y0 = &ys[i];

            let x0_0 = vld1q_s8(x0.qs.as_ptr());
            let x0_1 = vld1q_s8(x0.qs.as_ptr().add(16));

            // load y
            let y0_0 = vld1q_s8(y0.qs.as_ptr());
            let y0_1 = vld1q_s8(y0.qs.as_ptr().add(16));

            let p0 = vdotq_s32(x0_0, y0_0);
            let p1 = vdotq_s32(x0_1, y0_1);

            sumv0 = vmlaq_n_f32(
                sumv0,
                vcvtq_f32_s32(vaddq_s32(p0, p1)),
                x0.d.to_f32() * y0.d.to_f32(),
            );
        }
        vaddvq_f32(sumv0)
    }
}

#[inline(always)]
pub(crate) fn vec_dot_q8k_q8k(n: usize, xs: &[BlockQ8K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q8k_q8k: {n} is not divisible by {QK_K}"
    );
    let mut sumf = 0f32;
    for (xs, ys) in xs.iter().zip(ys.iter()) {
        unsafe {
            let mut sum_i = vdupq_n_s32(0);
            let scale = xs.d * ys.d;
            let xs = xs.qs.as_ptr();
            let ys = ys.qs.as_ptr();
            for i in (0..QK_K).step_by(16) {
                let xs = vld1q_s8(xs.add(i));
                let ys = vld1q_s8(ys.add(i));
                let xy = vdotq_s32(xs, ys);
                sum_i = vaddq_s32(sum_i, xy)
            }
            sumf += vaddvq_s32(sum_i) as f32 * scale
        }
    }
    sumf
}

#[inline(always)]
pub(crate) fn vec_dot_q6k_q8k(n: usize, xs: &[BlockQ6K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q6k_q8k: {n} is not divisible by {QK_K}"
    );
    let mut sum = 0f32;
    unsafe {
        let m4b = vdupq_n_u8(0xF);

        let mone = vdupq_n_u8(3);

        for (x, y) in xs.iter().zip(ys.iter()) {
            let d_all = x.d.to_f32();

            let mut q6 = x.ql.as_ptr();
            let mut qh = x.qh.as_ptr();
            let mut q8 = y.qs.as_ptr();

            let mut scale = x.scales.as_ptr();

            let q8sums = vld1q_s16_x2(y.bsums.as_ptr());
            let scales = vld1q_s8(scale);
            let q6scales = int16x8x2_t(
                vmovl_s8(vget_low_s8(scales)),
                vmovl_s8(vget_high_s8(scales)),
            );

            let prod = vaddq_s32(
                vaddq_s32(
                    vmull_s16(vget_low_s16(q8sums.0), vget_low_s16(q6scales.0)),
                    vmull_s16(vget_high_s16(q8sums.0), vget_high_s16(q6scales.0)),
                ),
                vaddq_s32(
                    vmull_s16(vget_low_s16(q8sums.1), vget_low_s16(q6scales.1)),
                    vmull_s16(vget_high_s16(q8sums.1), vget_high_s16(q6scales.1)),
                ),
            );
            let isum_mins = vaddvq_s32(prod);

            let mut isum = 0i32;

            for _j in 0..QK_K / 128 {
                let qhbits = vld1q_u8_x2(qh);
                qh = qh.add(32);
                let q6bits = vld1q_u8_x4(q6);
                q6 = q6.add(64);
                let q8bytes = vld1q_s8_x4(q8);
                q8 = q8.add(64);

                let q6h_0 = vshlq_n_u8(vandq_u8(mone, qhbits.0), 4);
                let q6h_1 = vshlq_n_u8(vandq_u8(mone, qhbits.1), 4);
                let shifted = vshrq_n_u8(qhbits.0, 2);
                let q6h_2 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
                let shifted = vshrq_n_u8(qhbits.1, 2);
                let q6h_3 = vshlq_n_u8(vandq_u8(mone, shifted), 4);

                let q6bytes_0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.0, m4b), q6h_0));
                let q6bytes_1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.1, m4b), q6h_1));
                let q6bytes_2 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.2, m4b), q6h_2));
                let q6bytes_3 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.3, m4b), q6h_3));

                let p0 = vdotq_s32(q6bytes_0, q8bytes.0);
                let p1 = vdotq_s32(q6bytes_1, q8bytes.1);
                let (scale0, scale1) = (*scale as i32, *scale.add(1) as i32);
                isum += vaddvq_s32(p0) * scale0 + vaddvq_s32(p1) * scale1;
                scale = scale.add(2);

                let p2 = vdotq_s32(q6bytes_2, q8bytes.2);
                let p3 = vdotq_s32(q6bytes_3, q8bytes.3);
                let (scale0, scale1) = (*scale as i32, *scale.add(1) as i32);
                isum += vaddvq_s32(p2) * scale0 + vaddvq_s32(p3) * scale1;
                scale = scale.add(2);

                let q8bytes = vld1q_s8_x4(q8);
                q8 = q8.add(64);

                let shifted = vshrq_n_u8(qhbits.0, 4);
                let q6h_0 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
                let shifted = vshrq_n_u8(qhbits.1, 4);
                let q6h_1 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
                let shifted = vshrq_n_u8(qhbits.0, 6);
                let q6h_2 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
                let shifted = vshrq_n_u8(qhbits.1, 6);
                let q6h_3 = vshlq_n_u8(vandq_u8(mone, shifted), 4);

                let q6bytes_0 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.0, 4), q6h_0));
                let q6bytes_1 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.1, 4), q6h_1));
                let q6bytes_2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.2, 4), q6h_2));
                let q6bytes_3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.3, 4), q6h_3));

                let p0 = vdotq_s32(q6bytes_0, q8bytes.0);
                let p1 = vdotq_s32(q6bytes_1, q8bytes.1);
                let (scale0, scale1) = (*scale as i32, *scale.add(1) as i32);
                isum += vaddvq_s32(p0) * scale0 + vaddvq_s32(p1) * scale1;
                scale = scale.add(2);

                let p2 = vdotq_s32(q6bytes_2, q8bytes.2);
                let p3 = vdotq_s32(q6bytes_3, q8bytes.3);
                let (scale0, scale1) = (*scale as i32, *scale.add(1) as i32);
                isum += vaddvq_s32(p2) * scale0 + vaddvq_s32(p3) * scale1;
                scale = scale.add(2);
            }
            sum += d_all * y.d * ((isum - 32 * isum_mins) as f32);
        }
    }
    sum
}

#[inline(always)]
pub(crate) fn vec_dot_q5k_q8k(n: usize, xs: &[BlockQ5K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q5k_q8k: {n} is not divisible by {QK_K}"
    );
    let mut sumf = 0f32;
    let mut utmp = [0u32; 4];
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    unsafe {
        let m4b = vdupq_n_u8(0xF);
        let mone = vdupq_n_u8(1);
        let mtwo = vdupq_n_u8(2);

        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = y.d * x.d.to_f32();
            let dmin = y.d * x.dmin.to_f32();

            let q8sums = vpaddq_s16(
                vld1q_s16(y.bsums.as_ptr()),
                vld1q_s16(y.bsums.as_ptr().add(8)),
            );

            LittleEndian::read_u32_into(&x.scales, &mut utmp[0..3]);

            utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
            let uaux = utmp[1] & KMASK1;
            utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
            utmp[2] = uaux;
            utmp[0] &= KMASK1;

            let mins8 = vld1_u8((utmp.as_ptr() as *const u8).add(8));
            let mins = vreinterpretq_s16_u16(vmovl_u8(mins8));
            let prod = vaddq_s32(
                vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
                vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)),
            );
            let sumi_mins = vaddvq_s32(prod);

            let mut scales = utmp.as_ptr() as *const u8;

            let mut q5 = x.qs.as_ptr();
            let mut q8 = y.qs.as_ptr();

            let mut qhbits = vld1q_u8_x2(x.qh.as_ptr());

            let mut sumi = 0i32;

            for _j in 0..QK_K / 64 {
                let q5bits = vld1q_u8_x2(q5);
                q5 = q5.add(32);
                let q8bytes = vld1q_s8_x4(q8);
                q8 = q8.add(64);

                let q5h_0 = vshlq_n_u8(vandq_u8(mone, qhbits.0), 4);
                let q5h_1 = vshlq_n_u8(vandq_u8(mone, qhbits.1), 4);
                let q5h_2 = vshlq_n_u8(vandq_u8(mtwo, qhbits.0), 3);
                let q5h_3 = vshlq_n_u8(vandq_u8(mtwo, qhbits.1), 3);
                qhbits.0 = vshrq_n_u8(qhbits.0, 2);
                qhbits.1 = vshrq_n_u8(qhbits.1, 2);

                let q5bytes_0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits.0, m4b), q5h_0));
                let q5bytes_1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits.1, m4b), q5h_1));
                let q5bytes_2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits.0, 4), q5h_2));
                let q5bytes_3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits.1, 4), q5h_3));

                let p0 = vdotq_s32(q5bytes_0, q8bytes.0);
                let p1 = vdotq_s32(q5bytes_1, q8bytes.1);
                sumi += vaddvq_s32(vaddq_s32(p0, p1)) * *scales as i32;
                scales = scales.add(1);

                let p2 = vdotq_s32(q5bytes_2, q8bytes.2);
                let p3 = vdotq_s32(q5bytes_3, q8bytes.3);
                sumi += vaddvq_s32(vaddq_s32(p2, p3)) * *scales as i32;
                scales = scales.add(1);
            }
            sumf += d * sumi as f32 - dmin * sumi_mins as f32;
        }
    }
    sumf
}

#[inline(always)]
pub(crate) fn vec_dot_q4k_q8k(n: usize, xs: &[BlockQ4K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q4k_q8k: {n} is not divisible by {QK_K}"
    );
    let mut sumf = 0f32;
    let mut utmp = [0u32; 4];
    let mut scales = [0u8; 16];
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    unsafe {
        let m4b = vdupq_n_u8(0xF);

        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = y.d * x.d.to_f32();
            let dmin = y.d * x.dmin.to_f32();

            let q8sums = vpaddq_s16(
                vld1q_s16(y.bsums.as_ptr()),
                vld1q_s16(y.bsums.as_ptr().add(8)),
            );

            LittleEndian::read_u32_into(&x.scales, &mut utmp[0..3]);

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
            sumf -= dmin * vaddvq_s32(prod) as f32;

            LittleEndian::write_u32_into(&utmp, &mut scales);

            let mut q4 = x.qs.as_ptr();
            let mut q8 = y.qs.as_ptr();

            let mut sumi1 = 0i32;
            let mut sumi2 = 0i32;

            for j in 0..QK_K / 64 {
                let q4bits = vld1q_u8_x2(q4);
                q4 = q4.add(32);
                let q8bytes = vld1q_s8_x2(q8);
                q8 = q8.add(32);
                let q4bytes = int8x16x2_t(
                    vreinterpretq_s8_u8(vandq_u8(q4bits.0, m4b)),
                    vreinterpretq_s8_u8(vandq_u8(q4bits.1, m4b)),
                );
                let p0 = vdotq_s32(q4bytes.0, q8bytes.0);
                let p1 = vdotq_s32(q4bytes.1, q8bytes.1);
                sumi1 += vaddvq_s32(vaddq_s32(p0, p1)) * scales[2 * j] as i32;

                let q8bytes = vld1q_s8_x2(q8);
                q8 = q8.add(32);
                let q4bytes = int8x16x2_t(
                    vreinterpretq_s8_u8(vshrq_n_u8(q4bits.0, 4)),
                    vreinterpretq_s8_u8(vshrq_n_u8(q4bits.1, 4)),
                );
                let p2 = vdotq_s32(q4bytes.0, q8bytes.0);
                let p3 = vdotq_s32(q4bytes.1, q8bytes.1);
                sumi2 += vaddvq_s32(vaddq_s32(p2, p3)) * scales[2 * j + 1] as i32;
            }
            sumf += d * (sumi1 + sumi2) as f32;
        }
    }
    sumf
}

/// Multi-row Q4K×Q8K dot (R ≤ 4 rows): unpacks each weight superblock once and dots
/// it against R activation rows. The nibble load/mask/shift work is shared, which is
/// the dominant non-SDOT cost of the single-row kernel; per-row arithmetic order is
/// identical to `vec_dot_q4k_q8k`, so results match it bit-for-bit.
#[inline(always)]
pub(crate) fn vec_dot_q4k_q8k_xr<const R: usize>(
    n: usize,
    xs: &[BlockQ4K],
    ys: &[&[BlockQ8K]; R],
    dst: &mut [f32],
) {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q4k_q8k_xr: {n} is not divisible by {QK_K}"
    );
    let nrows = R;
    // R up to 8: one weight column is unpacked into 4 nibble vectors (lo.0/.1,
    // hi.0/.1) shared across all R rows, plus R int32 accumulators. R=8 → ~12
    // live vectors, within N1's 32-register file (the 2-D NR×MR kernel spilled
    // because it held NR× the nibble vectors; this holds exactly one column's).
    debug_assert!(nrows >= 1 && nrows <= 8 && dst.len() == nrows);
    let mut utmp = [0u32; 4];
    let mut scales = [0u8; 16];
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let mut sumf = [0f32; R];
    unsafe {
        let m4b = vdupq_n_u8(0xF);

        for (i, x) in xs.iter().enumerate() {
            // Shared: unpack the weight superblock's 6-bit scales/mins.
            LittleEndian::read_u32_into(&x.scales, &mut utmp[0..3]);
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
            LittleEndian::write_u32_into(&utmp, &mut scales);
            let xd = x.d.to_f32();
            let xdmin = x.dmin.to_f32();

            // Per row: the -dmin * Σ(mins · bsums) correction.
            for (r, sf) in sumf.iter_mut().enumerate().take(nrows) {
                let y = &ys[r][i];
                let q8sums = vpaddq_s16(
                    vld1q_s16(y.bsums.as_ptr()),
                    vld1q_s16(y.bsums.as_ptr().add(8)),
                );
                let prod = vaddq_s32(
                    vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
                    vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)),
                );
                *sf -= y.d * xdmin * vaddvq_s32(prod) as f32;
            }

            let mut q4 = x.qs.as_ptr();
            let mut q8p = [core::ptr::null::<i8>(); R];
            for r in 0..nrows {
                q8p[r] = ys[r][i].qs.as_ptr();
            }
            // Vector accumulators: one horizontal reduction per superblock per row
            // instead of one per 32 weights (i32 adds are associative, so this is
            // bit-identical to the scalar-sum form).
            let mut acc = [vdupq_n_s32(0); R];

            for j in 0..QK_K / 64 {
                let q4bits = vld1q_u8_x2(q4);
                q4 = q4.add(32);
                let lo = int8x16x2_t(
                    vreinterpretq_s8_u8(vandq_u8(q4bits.0, m4b)),
                    vreinterpretq_s8_u8(vandq_u8(q4bits.1, m4b)),
                );
                let hi = int8x16x2_t(
                    vreinterpretq_s8_u8(vshrq_n_u8(q4bits.0, 4)),
                    vreinterpretq_s8_u8(vshrq_n_u8(q4bits.1, 4)),
                );

                for r in 0..nrows {
                    let q8 = vld1q_s8_x2(q8p[r]);
                    q8p[r] = q8p[r].add(32);
                    let p = vaddq_s32(vdotq_s32(lo.0, q8.0), vdotq_s32(lo.1, q8.1));
                    acc[r] = vmlaq_n_s32(acc[r], p, scales[2 * j] as i32);
                    let q8 = vld1q_s8_x2(q8p[r]);
                    q8p[r] = q8p[r].add(32);
                    let p = vaddq_s32(vdotq_s32(hi.0, q8.0), vdotq_s32(hi.1, q8.1));
                    acc[r] = vmlaq_n_s32(acc[r], p, scales[2 * j + 1] as i32);
                }
            }
            for r in 0..nrows {
                sumf[r] += ys[r][i].d * xd * vaddvq_s32(acc[r]) as f32;
            }
        }
    }
    dst.copy_from_slice(&sumf[..nrows]);
}

/// 2-D register-blocked Q4_K × Q8_K microkernel: `NR` weight columns × `MR`
/// activation rows in one pass. Each weight superblock's nibble unpack is shared
/// across the `MR` rows (as in `_xr`), and each activation's Q8 load is shared
/// across the `NR` weight columns — so *both* operands are reused from registers,
/// cutting weight-decode by `MR×` and activation traffic by `NR×` (the structure
/// llama.cpp's GEMM uses). Per-(col,row) arithmetic order is identical to
/// `vec_dot_q4k_q8k`, so results are bit-for-bit identical. `dst` is `NR×MR`
/// row-major: `dst[w * MR + a]` = column `w` · row `a`.
#[inline(always)]
#[allow(clippy::needless_range_loop)]
pub(crate) fn vec_dot_q4k_q8k_mn<const NR: usize, const MR: usize>(
    n: usize,
    xs: &[&[BlockQ4K]; NR],
    ys: &[&[BlockQ8K]; MR],
    dst: &mut [f32],
) {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q4k_q8k_mn: {n} is not divisible by {QK_K}"
    );
    debug_assert!(NR >= 1 && MR >= 1 && dst.len() == NR * MR);
    let nb = n / QK_K;
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    for d in dst.iter_mut() {
        *d = 0.0;
    }
    unsafe {
        let m4b = vdupq_n_u8(0xF);
        for i in 0..nb {
            // --- per weight column: unpack 6-bit scales/mins, base pointers ---
            let mut scales_w = [[0u8; 16]; NR];
            let mut mins_w = [vdupq_n_s16(0); NR];
            let mut xd = [0f32; NR];
            let mut xdmin = [0f32; NR];
            let mut q4p = [core::ptr::null::<u8>(); NR];
            for w in 0..NR {
                let x = &xs[w][i];
                let mut utmp = [0u32; 4];
                LittleEndian::read_u32_into(&x.scales, &mut utmp[0..3]);
                let mins8 = vld1_u32(
                    [
                        utmp[1] & KMASK1,
                        ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4),
                    ]
                    .as_ptr(),
                );
                utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
                utmp[0] &= KMASK1;
                mins_w[w] = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
                LittleEndian::write_u32_into(&utmp, &mut scales_w[w]);
                xd[w] = x.d.to_f32();
                xdmin[w] = x.dmin.to_f32();
                q4p[w] = x.qs.as_ptr();
            }

            // --- per activation row: base pointer; the -dmin·Σ(mins·bsums) term ---
            let mut q8p = [core::ptr::null::<i8>(); MR];
            for a in 0..MR {
                let y = &ys[a][i];
                q8p[a] = y.qs.as_ptr();
                let q8sums = vpaddq_s16(
                    vld1q_s16(y.bsums.as_ptr()),
                    vld1q_s16(y.bsums.as_ptr().add(8)),
                );
                let yd = y.d;
                for w in 0..NR {
                    let prod = vaddq_s32(
                        vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins_w[w])),
                        vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins_w[w])),
                    );
                    dst[w * MR + a] -= yd * xdmin[w] * vaddvq_s32(prod) as f32;
                }
            }

            // --- main: NR×MR int32 accumulators, both operands reused in-register ---
            let mut acc = [[vdupq_n_s32(0); MR]; NR];
            for j in 0..QK_K / 64 {
                // Register-pressure note: 16 int32 accumulators (NR×MR) already use
                // half the NEON file. Materializing lo+hi nibbles for all NR columns
                // (16 vectors) on top of that spills on N1. So split into two passes,
                // each holding only the 8 nibble vectors it needs; q4bits is reloaded
                // from L1 for the high pass. Per-(w,a) op order is unchanged (low 2j
                // then high 2j+1), so this stays bit-identical to the scalar path.

                // Pass 1: low-nibble sub-block (scale index 2j).
                let mut lo0 = [vdupq_n_s8(0); NR];
                let mut lo1 = [vdupq_n_s8(0); NR];
                for w in 0..NR {
                    let q4bits = vld1q_u8_x2(q4p[w]);
                    lo0[w] = vreinterpretq_s8_u8(vandq_u8(q4bits.0, m4b));
                    lo1[w] = vreinterpretq_s8_u8(vandq_u8(q4bits.1, m4b));
                }
                for a in 0..MR {
                    let q8 = vld1q_s8_x2(q8p[a]);
                    q8p[a] = q8p[a].add(32);
                    for w in 0..NR {
                        let p = vaddq_s32(vdotq_s32(lo0[w], q8.0), vdotq_s32(lo1[w], q8.1));
                        acc[w][a] = vmlaq_n_s32(acc[w][a], p, scales_w[w][2 * j] as i32);
                    }
                }

                // Pass 2: high-nibble sub-block (scale index 2j+1); reload + advance.
                let mut hi0 = [vdupq_n_s8(0); NR];
                let mut hi1 = [vdupq_n_s8(0); NR];
                for w in 0..NR {
                    let q4bits = vld1q_u8_x2(q4p[w]);
                    q4p[w] = q4p[w].add(32);
                    hi0[w] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.0, 4));
                    hi1[w] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.1, 4));
                }
                for a in 0..MR {
                    let q8 = vld1q_s8_x2(q8p[a]);
                    q8p[a] = q8p[a].add(32);
                    for w in 0..NR {
                        let p = vaddq_s32(vdotq_s32(hi0[w], q8.0), vdotq_s32(hi1[w], q8.1));
                        acc[w][a] = vmlaq_n_s32(acc[w][a], p, scales_w[w][2 * j + 1] as i32);
                    }
                }
            }
            for w in 0..NR {
                for a in 0..MR {
                    dst[w * MR + a] += ys[a][i].d * xd[w] * vaddvq_s32(acc[w][a]) as f32;
                }
            }
        }
    }
}

/// SDOT GEMM over candle's interleaved `BlockQ4Kx8`: `NC` output channels starting
/// at `c_off` (NC + c_off ≤ 8) × `MR` activation rows. Output `dst[c * MR + a]` =
/// channel `c_off+c` · row `a`, numerically identical to `vec_dot_q4k_q8k` per pair.
///
/// The repacking payoff is the loop order: scales/mins read pre-unpacked (no 6-bit
/// twiddle), each chunk's channel weights contiguous, and each q8 activation load
/// reused across all NC channels. Nibbles taken a half at a time (lo then hi) to
/// keep `NC × 2` weight vectors live alongside the `NC × MR` accumulators — so the
/// NC/MR product sets register pressure (8×4 spills on N1; 4×4 and 8×2 fit).
#[inline(always)]
#[allow(clippy::needless_range_loop)]
pub(crate) fn gemm_q4kx_q8k<const NC: usize, const MR: usize>(
    packed: &[BlockQ4Kx8],
    c_off: usize,
    rows: &[&[BlockQ8K]; MR],
    dst: &mut [f32],
) {
    const PACK_ROWS: usize = 8; // channels stored per packed block (chunk stride)
    debug_assert!(dst.len() == NC * MR && c_off + NC <= PACK_ROWS);
    let nb = packed.len();
    // Running accumulator per (channel, row), updated as `-= mins` then `+= main`
    // per super-block — the SAME two-op float sequence as `vec_dot`/`_xr`, so the
    // result is bit-identical. Written to `dst` once at the end.
    let mut sumf = [[0f32; MR]; NC];
    unsafe {
        let m4b = vdupq_n_u8(0xF);
        for i in 0..nb {
            let blk = &packed[i];
            // d / dmin for the NC channels at c_off; y.d per row.
            let mut cd = [0f32; NC];
            let mut cdmin = [0f32; NC];
            for c in 0..NC {
                cd[c] = blk.d[c_off + c].to_f32();
                cdmin[c] = blk.dmin[c_off + c].to_f32();
            }

            // mins correction: per (c, a)  -= y.d * dmin[c] * Σ_s min[c][s]·q8sums[a][s],
            // q8sums[a][s] = bsums[2s] + bsums[2s+1].  (matches _xr / vec_dot.)
            let mut q8sums = [vdupq_n_s16(0); MR];
            for a in 0..MR {
                let b = rows[a][i].bsums.as_ptr();
                q8sums[a] = vpaddq_s16(vld1q_s16(b), vld1q_s16(b.add(8)));
            }
            for c in 0..NC {
                let mb = blk.mins.as_ptr().add((c_off + c) * 8);
                let mins = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(mb)));
                for a in 0..MR {
                    let prod = vaddq_s32(
                        vmull_s16(vget_low_s16(q8sums[a]), vget_low_s16(mins)),
                        vmull_s16(vget_high_s16(q8sums[a]), vget_high_s16(mins)),
                    );
                    sumf[c][a] -= rows[a][i].d * cdmin[c] * vaddvq_s32(prod) as f32;
                }
            }

            let mut acc = [[vdupq_n_s32(0); MR]; NC];
            for j in 0..QK_K / 64 {
                // chunk j is always a row-interleaved 8×32-byte run; channel c_off+c
                // lives at qs[j*256 + (c_off+c)*32].
                let wbase = j * (PACK_ROWS * 32);
                let sc_lo = 2 * j;
                let sc_hi = 2 * j + 1;

                let mut lo0 = [vdupq_n_s8(0); NC];
                let mut lo1 = [vdupq_n_s8(0); NC];
                for c in 0..NC {
                    let q4b = vld1q_u8_x2(blk.qs.as_ptr().add(wbase + (c_off + c) * 32));
                    lo0[c] = vreinterpretq_s8_u8(vandq_u8(q4b.0, m4b));
                    lo1[c] = vreinterpretq_s8_u8(vandq_u8(q4b.1, m4b));
                }
                for a in 0..MR {
                    let q8 = vld1q_s8_x2(rows[a][i].qs.as_ptr().add(sc_lo * 32));
                    for c in 0..NC {
                        let p = vaddq_s32(vdotq_s32(lo0[c], q8.0), vdotq_s32(lo1[c], q8.1));
                        acc[c][a] =
                            vmlaq_n_s32(acc[c][a], p, blk.scales[(c_off + c) * 8 + sc_lo] as i32);
                    }
                }

                let mut hi0 = [vdupq_n_s8(0); NC];
                let mut hi1 = [vdupq_n_s8(0); NC];
                for c in 0..NC {
                    let q4b = vld1q_u8_x2(blk.qs.as_ptr().add(wbase + (c_off + c) * 32));
                    hi0[c] = vreinterpretq_s8_u8(vshrq_n_u8(q4b.0, 4));
                    hi1[c] = vreinterpretq_s8_u8(vshrq_n_u8(q4b.1, 4));
                }
                for a in 0..MR {
                    let q8 = vld1q_s8_x2(rows[a][i].qs.as_ptr().add(sc_hi * 32));
                    for c in 0..NC {
                        let p = vaddq_s32(vdotq_s32(hi0[c], q8.0), vdotq_s32(hi1[c], q8.1));
                        acc[c][a] =
                            vmlaq_n_s32(acc[c][a], p, blk.scales[(c_off + c) * 8 + sc_hi] as i32);
                    }
                }
            }
            for c in 0..NC {
                for a in 0..MR {
                    sumf[c][a] += rows[a][i].d * cd[c] * vaddvq_s32(acc[c][a]) as f32;
                }
            }
        }
        for c in 0..NC {
            for a in 0..MR {
                dst[c * MR + a] = sumf[c][a];
            }
        }
    }
}

/// Convenience wrapper: all 8 channels of the packed block × `MR` rows.
#[inline(always)]
pub(crate) fn gemm_q4kx8_q8k<const MR: usize>(
    packed: &[BlockQ4Kx8],
    rows: &[&[BlockQ8K]; MR],
    dst: &mut [f32],
) {
    gemm_q4kx_q8k::<8, MR>(packed, 0, rows, dst)
}

#[inline(always)]
pub(crate) fn vec_dot_q3k_q8k(n: usize, xs: &[BlockQ3K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q3k_q8k: {n} is not divisible by {QK_K}"
    );
    let mut sumf = 0f32;
    let mut utmp = [0u32; 4];
    let mut aux = [0u32; 3];
    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;

    unsafe {
        let m3b = vdupq_n_u8(0x3);
        let m0 = vdupq_n_u8(1);
        let m1 = vshlq_n_u8(m0, 1);
        let m2 = vshlq_n_u8(m0, 2);
        let m3 = vshlq_n_u8(m0, 3);
        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = y.d * x.d.to_f32();
            let mut q3 = x.qs.as_ptr();
            let qh = x.hmask.as_ptr();
            let mut q8 = y.qs.as_ptr();

            let mut qhbits = vld1q_u8_x2(qh);

            let mut isum = 0i32;

            // Set up scales
            LittleEndian::read_u32_into(&x.scales, &mut aux);

            utmp[3] = ((aux[1] >> 4) & KMASK2) | (((aux[2] >> 6) & KMASK1) << 4);
            utmp[2] = ((aux[0] >> 4) & KMASK2) | (((aux[2] >> 4) & KMASK1) << 4);
            utmp[1] = (aux[1] & KMASK2) | (((aux[2] >> 2) & KMASK1) << 4);
            utmp[0] = (aux[0] & KMASK2) | ((aux[2] & KMASK1) << 4);

            let mut scale = utmp.as_mut_ptr() as *mut i8;
            for j in 0..16 {
                *scale.add(j) -= 32i8
            }

            for j in 0..QK_K / 128 {
                let q3bits = vld1q_u8_x2(q3);
                q3 = q3.add(32);
                let q8bytes_1 = vld1q_s8_x4(q8);
                q8 = q8.add(64);
                let q8bytes_2 = vld1q_s8_x4(q8);
                q8 = q8.add(64);

                let q3h_0 = vshlq_n_u8(vbicq_u8(m0, qhbits.0), 2);
                let q3h_1 = vshlq_n_u8(vbicq_u8(m0, qhbits.1), 2);
                let q3h_2 = vshlq_n_u8(vbicq_u8(m1, qhbits.0), 1);
                let q3h_3 = vshlq_n_u8(vbicq_u8(m1, qhbits.1), 1);

                let q3bytes_0 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(q3bits.0, m3b)),
                    vreinterpretq_s8_u8(q3h_0),
                );
                let q3bytes_1 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(q3bits.1, m3b)),
                    vreinterpretq_s8_u8(q3h_1),
                );
                let q3bytes_2 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.0, 2), m3b)),
                    vreinterpretq_s8_u8(q3h_2),
                );
                let q3bytes_3 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.1, 2), m3b)),
                    vreinterpretq_s8_u8(q3h_3),
                );

                let p0 = vdotq_s32(q3bytes_0, q8bytes_1.0);
                let p1 = vdotq_s32(q3bytes_1, q8bytes_1.1);
                let p2 = vdotq_s32(q3bytes_2, q8bytes_1.2);
                let p3 = vdotq_s32(q3bytes_3, q8bytes_1.3);
                isum += vaddvq_s32(p0) * *scale as i32
                    + vaddvq_s32(p1) * *scale.add(1) as i32
                    + vaddvq_s32(p2) * *scale.add(2) as i32
                    + vaddvq_s32(p3) * *scale.add(3) as i32;
                scale = scale.add(4);

                let q3h_0 = vbicq_u8(m2, qhbits.0);
                let q3h_1 = vbicq_u8(m2, qhbits.1);
                let q3h_2 = vshrq_n_u8(vbicq_u8(m3, qhbits.0), 1);
                let q3h_3 = vshrq_n_u8(vbicq_u8(m3, qhbits.1), 1);

                let q3bytes_0 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.0, 4), m3b)),
                    vreinterpretq_s8_u8(q3h_0),
                );
                let q3bytes_1 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.1, 4), m3b)),
                    vreinterpretq_s8_u8(q3h_1),
                );
                let q3bytes_2 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.0, 6), m3b)),
                    vreinterpretq_s8_u8(q3h_2),
                );
                let q3bytes_3 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.1, 6), m3b)),
                    vreinterpretq_s8_u8(q3h_3),
                );

                let p0 = vdotq_s32(q3bytes_0, q8bytes_2.0);
                let p1 = vdotq_s32(q3bytes_1, q8bytes_2.1);
                let p2 = vdotq_s32(q3bytes_2, q8bytes_2.2);
                let p3 = vdotq_s32(q3bytes_3, q8bytes_2.3);
                isum += vaddvq_s32(p0) * *scale as i32
                    + vaddvq_s32(p1) * *scale.add(1) as i32
                    + vaddvq_s32(p2) * *scale.add(2) as i32
                    + vaddvq_s32(p3) * *scale.add(3) as i32;
                scale = scale.add(4);

                if j == 0 {
                    qhbits.0 = vshrq_n_u8(qhbits.0, 4);
                    qhbits.1 = vshrq_n_u8(qhbits.1, 4);
                }
            }
            sumf += d * isum as f32;
        }
    }
    sumf
}

#[inline(always)]
pub(crate) fn vec_dot_q2k_q8k(n: usize, xs: &[BlockQ2K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q2k_q8k: {n} is not divisible by {QK_K}"
    );
    let mut sumf = 0f32;
    let mut aux = [0u8; 16];

    unsafe {
        let m3 = vdupq_n_u8(0x3);
        let m4 = vdupq_n_u8(0xF);

        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = y.d * x.d.to_f32();
            let dmin = -y.d * x.dmin.to_f32();

            let mut q2 = x.qs.as_ptr();
            let mut q8 = y.qs.as_ptr();
            let sc = x.scales.as_ptr();

            let mins_and_scales = vld1q_u8(sc);
            let scales = vandq_u8(mins_and_scales, m4);
            vst1q_u8(aux.as_mut_ptr(), scales);

            let mins = vshrq_n_u8(mins_and_scales, 4);
            let q8sums = vld1q_s16_x2(y.bsums.as_ptr());
            let mins16 = int16x8x2_t(
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins))),
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins))),
            );
            let s0 = vaddq_s32(
                vmull_s16(vget_low_s16(mins16.0), vget_low_s16(q8sums.0)),
                vmull_s16(vget_high_s16(mins16.0), vget_high_s16(q8sums.0)),
            );
            let s1 = vaddq_s32(
                vmull_s16(vget_low_s16(mins16.1), vget_low_s16(q8sums.1)),
                vmull_s16(vget_high_s16(mins16.1), vget_high_s16(q8sums.1)),
            );
            sumf += dmin * vaddvq_s32(vaddq_s32(s0, s1)) as f32;

            let mut isum = 0i32;
            let mut is = 0usize;

            // TODO: dotprod
            for _j in 0..QK_K / 128 {
                let q2bits = vld1q_u8_x2(q2);
                q2 = q2.add(32);

                let q8bytes = vld1q_s8_x2(q8);
                q8 = q8.add(32);
                let mut q2bytes = int8x16x2_t(
                    vreinterpretq_s8_u8(vandq_u8(q2bits.0, m3)),
                    vreinterpretq_s8_u8(vandq_u8(q2bits.1, m3)),
                );
                isum += multiply_accum_with_scale(&aux, is, 0, q2bytes, q8bytes);

                let q8bytes = vld1q_s8_x2(q8);
                q8 = q8.add(32);
                q2bytes.0 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.0, 2), m3));
                q2bytes.1 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.1, 2), m3));
                isum += multiply_accum_with_scale(&aux, is, 2, q2bytes, q8bytes);

                let q8bytes = vld1q_s8_x2(q8);
                q8 = q8.add(32);
                q2bytes.0 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.0, 4), m3));
                q2bytes.1 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.1, 4), m3));
                isum += multiply_accum_with_scale(&aux, is, 4, q2bytes, q8bytes);

                let q8bytes = vld1q_s8_x2(q8);
                q8 = q8.add(32);
                q2bytes.0 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.0, 6), m3));
                q2bytes.1 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.1, 6), m3));
                isum += multiply_accum_with_scale(&aux, is, 6, q2bytes, q8bytes);

                is += 8;
            }
            sumf += d * isum as f32;
        }
    }
    sumf
}

#[inline(always)]
unsafe fn multiply_accum_with_scale(
    aux: &[u8; 16],
    is: usize,
    index: usize,
    q2bytes: int8x16x2_t,
    q8bytes: int8x16x2_t,
) -> i32 {
    let p1 = vdotq_s32(q2bytes.0, q8bytes.0);
    let p2 = vdotq_s32(q2bytes.1, q8bytes.1);
    vaddvq_s32(p1) * aux[is + index] as i32 + vaddvq_s32(p2) * aux[is + 1 + index] as i32
}

#[cfg(test)]
mod fused_gemm_tests {
    use super::*;
    use crate::quantized::k_quants::GgmlType;

    // Simple deterministic LCG -> f32 in [-1, 1); avoids a rand dependency.
    fn lcg(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (*state >> 33) as f32 / (1u64 << 31) as f32;
        2.0 * u - 1.0
    }

    #[test]
    fn q4k_mn_matches_scalar_bit_exact() {
        const NR: usize = 8;
        const MR: usize = 8;
        let nb = 3usize;
        let k = nb * QK_K;
        let mut st = 0x1234_5678_9abc_def0u64;

        // Build NR weight columns + MR activation rows once; each shape sub-test
        // slices the first nr/mr of them.
        let mut wq: Vec<Vec<BlockQ4K>> = Vec::new();
        for _ in 0..NR {
            let row: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
            let mut q = vec![BlockQ4K::zeros(); nb];
            BlockQ4K::from_float(&row, &mut q);
            wq.push(q);
        }
        let mut aq: Vec<Vec<BlockQ8K>> = Vec::new();
        for _ in 0..MR {
            let row: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
            let mut q = vec![BlockQ8K::zeros(); nb];
            BlockQ8K::from_float(&row, &mut q);
            aq.push(q);
        }
        let wref: Vec<&[BlockQ4K]> = wq.iter().map(|v| v.as_slice()).collect();
        let aref: Vec<&[BlockQ8K]> = aq.iter().map(|v| v.as_slice()).collect();

        // Every (NR,MR) shape dispatched by BlockQ4K::vec_dot_tile.
        macro_rules! check {
            ($nr:literal, $mr:literal) => {{
                let xs: [&[BlockQ4K]; $nr] = wref[..$nr].try_into().unwrap();
                let ys: [&[BlockQ8K]; $mr] = aref[..$mr].try_into().unwrap();
                let mut got = vec![0f32; $nr * $mr];
                vec_dot_q4k_q8k_mn::<$nr, $mr>(k, &xs, &ys, &mut got);
                for w in 0..$nr {
                    for a in 0..$mr {
                        let want = vec_dot_q4k_q8k(k, &wq[w], &aq[a]);
                        assert_eq!(
                            got[w * $mr + a].to_bits(),
                            want.to_bits(),
                            "shape {}x{} mismatch col {w} row {a}: got {} want {}",
                            $nr, $mr, got[w * $mr + a], want
                        );
                    }
                }
            }};
        }
        check!(4, 4);
        check!(4, 2);
        check!(2, 4);
        check!(2, 2);
        check!(8, 2);
        check!(2, 8);
        check!(4, 1);
        check!(1, 4);
        check!(1, 1);
    }
}
