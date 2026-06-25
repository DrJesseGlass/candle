use super::k_quants::{
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ4_0, BlockQ5K, BlockQ6K, BlockQ8K, BlockQ8_0, QK8_0, QK_K,
};
use super::repack::{BlockQ4Kx8, BlockQ4Kx8L, BlockQ6Kx8, BlockQ8Kx4};
use byteorder::{ByteOrder, LittleEndian};

#[allow(unused_imports)]
#[cfg(target_arch = "arm")]
use core::arch::arm::*;

#[allow(unused_imports)]
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

// Dot of two int8x16 vectors, grouped into four int32 lane-sums. Callers always reduce
// across lanes afterwards, so the lane grouping is irrelevant - only the total matters.
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

// Accumulating SDOT: acc += dot4(a, b) lane-wise via SDOT's native accumulate, so
// chains of sub-block dots reuse one register (vs vdupq_n_s32(0) + a vaddq per
// call). Integer accumulation is associative, so chaining is bit-identical to
// summing separate vdotq_s32 results.
#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
#[inline(always)]
unsafe fn vdotq_s32_acc(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
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
unsafe fn vdotq_s32_acc(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    vaddq_s32(acc, vdotq_s32(a, b))
}

// Software prefetch for the decode (m=1 GEMV) weight stream, `dist` bytes ahead.
// DEFAULT OFF (0): measured on Neoverse-N1, SW prefetch HURTS ~10% at 1-2 vCPU and
// only helps ~10% at 4 vCPU. The weight access is perfectly linear (`+32B`/step),
// which N1's HARDWARE prefetcher already hides at low thread counts - so the extra
// prfm is pure load-port/L1 overhead there. It only pays once 4 cores contend for
// the shared memory system and overwhelm the HW prefetcher. So leave OFF for the
// 1-2 vCPU tiers; set CANDLE_DECODE_PREFETCH=512 only on 4+ vCPU hosts. prfm is a
// fault-safe HINT, so prefetching past the buffer end is sound.
static DECODE_PREFETCH_DIST: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
    std::env::var("CANDLE_DECODE_PREFETCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
});

/// Emit `prfm pldl1keep, [addr]` - hint to pull `addr`'s line into L1 for a load.
/// No `nomem` option so the compiler keeps it (it models a memory side effect).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn prefetch_l1(addr: *const u8) {
    core::arch::asm!(
        "prfm pldl1keep, [{addr}]",
        addr = in(reg) addr,
        options(nostack, preserves_flags),
    );
}

/// Merge two per-lane (abs_max, signed_val) accumulator pairs.
/// Each output lane holds the signed value with the larger absolute value.
#[inline(always)]
unsafe fn merge_signed_max(
    abs_a: float32x4_t,
    smax_a: float32x4_t,
    abs_b: float32x4_t,
    smax_b: float32x4_t,
) -> (float32x4_t, float32x4_t) {
    (
        vmaxq_f32(abs_a, abs_b),
        vbslq_f32(vcgtq_f32(abs_b, abs_a), smax_b, smax_a),
    )
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
        let mzero = vdupq_n_s32(0); // chain seed, hoisted out of the loop

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
                // Chained sdots (ggml-style): both sub-block-half dots accumulate
                // into one register via SDOT native accumulate, then a single
                // scalar reduction. Bit-identical to the zero-init + vaddq form
                // (integer accumulation is associative); ~+9.7% cumulative N1 decode.
                let lo0 = vreinterpretq_s8_u8(vandq_u8(q4bits.0, m4b));
                let lo1 = vreinterpretq_s8_u8(vandq_u8(q4bits.1, m4b));
                let p_lo = vdotq_s32_acc(vdotq_s32_acc(mzero, lo0, q8bytes.0), lo1, q8bytes.1);
                sumi1 += vaddvq_s32(p_lo) * scales[2 * j] as i32;

                let q8bytes = vld1q_s8_x2(q8);
                q8 = q8.add(32);
                let hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.0, 4));
                let hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.1, 4));
                let p_hi = vdotq_s32_acc(vdotq_s32_acc(mzero, hi0, q8bytes.0), hi1, q8bytes.1);
                sumi2 += vaddvq_s32(p_hi) * scales[2 * j + 1] as i32;
            }
            sumf += d * (sumi1 + sumi2) as f32;
        }
    }
    sumf
}

/// Multi-row Q4KxQ8K dot (R <= 4 rows): unpacks each weight superblock once and dots
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
    debug_assert!(R >= 1 && R <= 8 && dst.len() == R);
    let mut utmp = [0u32; 4];
    let mut scales = [0u8; 16];
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let mut sumf = [0f32; R];
    unsafe {
        let m4b = vdupq_n_u8(0xF);
        let mzero = vdupq_n_s32(0); // chain seed for accumulate-SDOT, hoisted

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

            // Per row: the -dmin * sum(mins . bsums) correction.
            for (r, sf) in sumf.iter_mut().enumerate() {
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
            let mut q8p: [*const i8; R] = std::array::from_fn(|r| ys[r][i].qs.as_ptr());
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

                for r in 0..R {
                    let q8 = vld1q_s8_x2(q8p[r]);
                    q8p[r] = q8p[r].add(32);
                    let p = vdotq_s32_acc(vdotq_s32_acc(mzero, lo.0, q8.0), lo.1, q8.1);
                    acc[r] = vmlaq_n_s32(acc[r], p, scales[2 * j] as i32);
                    let q8 = vld1q_s8_x2(q8p[r]);
                    q8p[r] = q8p[r].add(32);
                    let p = vdotq_s32_acc(vdotq_s32_acc(mzero, hi.0, q8.0), hi.1, q8.1);
                    acc[r] = vmlaq_n_s32(acc[r], p, scales[2 * j + 1] as i32);
                }
            }
            for r in 0..R {
                sumf[r] += ys[r][i].d * xd * vaddvq_s32(acc[r]) as f32;
            }
        }
    }
    dst.copy_from_slice(&sumf);
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

/// Quantize a row of f32 activations into Q8K format
#[inline(always)]
pub(crate) fn quantize_row_q8k(xs: &[f32], ys: &mut [BlockQ8K]) {
    debug_assert!(
        xs.len().is_multiple_of(QK_K),
        "quantize_row_q8k: {} is not a multiple of {QK_K}",
        xs.len()
    );
    unsafe {
        for (chunk, y) in xs.chunks_exact(QK_K).zip(ys.iter_mut()) {
            // Find the element with the maximum absolute value, preserving its sign.
            let (mut vabs_max0, mut vsmax0) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
            let (mut vabs_max1, mut vsmax1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
            let (mut vabs_max2, mut vsmax2) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
            let (mut vabs_max3, mut vsmax3) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
            let mut p = chunk.as_ptr();
            for _ in 0..QK_K / 16 {
                let (v0, v1) = (vld1q_f32(p), vld1q_f32(p.add(4)));
                let (v2, v3) = (vld1q_f32(p.add(8)), vld1q_f32(p.add(12)));
                p = p.add(16);
                (vabs_max0, vsmax0) = merge_signed_max(vabs_max0, vsmax0, vabsq_f32(v0), v0);
                (vabs_max1, vsmax1) = merge_signed_max(vabs_max1, vsmax1, vabsq_f32(v1), v1);
                (vabs_max2, vsmax2) = merge_signed_max(vabs_max2, vsmax2, vabsq_f32(v2), v2);
                (vabs_max3, vsmax3) = merge_signed_max(vabs_max3, vsmax3, vabsq_f32(v3), v3);
            }
            // Tree-reduce 4 accumulators to 1.
            let (abs01, smax01) = merge_signed_max(vabs_max0, vsmax0, vabs_max1, vsmax1);
            let (abs23, smax23) = merge_signed_max(vabs_max2, vsmax2, vabs_max3, vsmax3);
            let (abs_v, smax_v) = merge_signed_max(abs01, smax01, abs23, smax23);
            // Cross lane reduce to scalar
            let mask_lohi = vcgt_f32(vget_high_f32(abs_v), vget_low_f32(abs_v));
            let abs_pair = vmax_f32(vget_low_f32(abs_v), vget_high_f32(abs_v));
            let smax_pair = vbsl_f32(mask_lohi, vget_high_f32(smax_v), vget_low_f32(smax_v));
            let max_signed = if vget_lane_f32(abs_pair, 1) > vget_lane_f32(abs_pair, 0) {
                vget_lane_f32(smax_pair, 1)
            } else {
                vget_lane_f32(smax_pair, 0)
            };

            if max_signed == 0.0f32 {
                y.d = 0.0f32;
                y.qs.fill(0);
                y.bsums.fill(0);
                continue;
            }

            let iscale = -128.0f32 / max_signed;
            let vscale = vdupq_n_f32(iscale);

            // Quantize f32 -> i8. Multiply, round-to-nearest, saturating narrow.
            let mut out = y.qs.as_mut_ptr();
            let mut p = chunk.as_ptr();
            for _ in 0..QK_K / 16 {
                let f0 = vmulq_f32(vld1q_f32(p), vscale);
                let f1 = vmulq_f32(vld1q_f32(p.add(4)), vscale);
                let f2 = vmulq_f32(vld1q_f32(p.add(8)), vscale);
                let f3 = vmulq_f32(vld1q_f32(p.add(12)), vscale);
                p = p.add(16);
                let s01 = vcombine_s16(
                    vqmovn_s32(vcvtaq_s32_f32(f0)),
                    vqmovn_s32(vcvtaq_s32_f32(f1)),
                );
                let s23 = vcombine_s16(
                    vqmovn_s32(vcvtaq_s32_f32(f2)),
                    vqmovn_s32(vcvtaq_s32_f32(f3)),
                );
                vst1q_s8(out, vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23)));
                out = out.add(16);
            }

            // Sum of each 16-element group of quantized values
            let qp = y.qs.as_ptr();
            for j in 0..QK_K / 16 {
                let v = vld1q_s8(qp.add(j * 16));
                y.bsums[j] = vaddvq_s32(vpaddlq_s16(vpaddlq_s8(v))) as i16;
            }

            y.d = 1.0f32 / iscale;
        }
    }
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
    // per super-block - the SAME two-op float sequence as `vec_dot`/`_xr`, so the
    // result is bit-identical. Written to `dst` once at the end.
    let mut sumf = [[0f32; MR]; NC];
    // Decode-only weight prefetch distance (bytes); 0 = off. Hoisted out of the
    // hot loop. Only the MR==1 (GEMV) path uses it - prefill (MR>=2) reuses weights
    // and is compute-bound, so prefetch is not its lever.
    let pf_dist = if MR == 1 { *DECODE_PREFETCH_DIST } else { 0 };
    unsafe {
        let m4b = vdupq_n_u8(0xF);
        let mzero = vdupq_n_s32(0); // sdot-chain seed + acc init, hoisted out of the block loop
        for i in 0..nb {
            let blk = &packed[i];
            // Hoist every base pointer ONCE per block. The per-(chunk,channel,row) addresses
            // are then `base + const` adds instead of re-walking `blk.qs.as_ptr().add(...)` and
            // re-indexing `rows[a][i]` each iteration - which the N1 perf annotate showed was
            // the kernel's single biggest cost (`add` ~16% of cycles). `c_off` is folded into
            // the weight/scale/min bases so the inner index drops the `(c_off + c)` term.
            let qsb = blk.qs.as_ptr().add(c_off * 32); // channel c at qsb + wbase + c*32
            let sc_ptr = blk.scales.as_ptr().add(c_off * 8); // channel c sub s at sc_ptr + c*8 + s
            let mins_ptr = blk.mins.as_ptr().add(c_off * 8);
            let mut rq = [core::ptr::null::<i8>(); MR]; // per-row Q8 quant base
            let mut rd = [0f32; MR]; // per-row Q8 scale
            let mut cd = [0f32; NC];
            let mut cdmin = [0f32; NC];
            for a in 0..MR {
                rq[a] = rows[a][i].qs.as_ptr();
                rd[a] = rows[a][i].d;
            }
            for c in 0..NC {
                cd[c] = blk.d[c_off + c].to_f32();
                cdmin[c] = blk.dmin[c_off + c].to_f32();
            }

            // mins correction: per (c, a)  -= y.d * dmin[c] * sum_s min[c][s].q8sums[a][s],
            // q8sums[a][s] = bsums[2s] + bsums[2s+1].  (matches _xr / vec_dot.)
            let mut q8sums = [vdupq_n_s16(0); MR];
            for a in 0..MR {
                let b = rows[a][i].bsums.as_ptr();
                q8sums[a] = vpaddq_s16(vld1q_s16(b), vld1q_s16(b.add(8)));
            }
            for c in 0..NC {
                let mins = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(mins_ptr.add(c * 8))));
                for a in 0..MR {
                    let prod = vaddq_s32(
                        vmull_s16(vget_low_s16(q8sums[a]), vget_low_s16(mins)),
                        vmull_s16(vget_high_s16(q8sums[a]), vget_high_s16(mins)),
                    );
                    sumf[c][a] -= rd[a] * cdmin[c] * vaddvq_s32(prod) as f32;
                }
            }

            let mut acc = [[mzero; MR]; NC];
            for j in 0..QK_K / 64 {
                // chunk j is a row-interleaved 8x32-byte run; channel c at qsb + j*256 + c*32.
                let wbase = j * (PACK_ROWS * 32);
                let sc_lo = 2 * j;
                let sc_hi = 2 * j + 1;

                if MR == 1 {
                    // Decode (GEMV): no row reuse, so streaming per channel keeps register
                    // pressure tiny (no NC-wide weight arrays to spill - which regressed the
                    // memory-bound m=1 path). Still loads each q4 ONCE (vs the old twice).
                    let q8l = vld1q_s8_x2(rq[0].add(sc_lo * 32));
                    let q8h = vld1q_s8_x2(rq[0].add(sc_hi * 32));
                    for c in 0..NC {
                        // Pull the weight line `pf_dist` bytes ahead into L1 while this
                        // channel computes. Address via usize arithmetic (prefetch may
                        // target past the block; prfm is fault-safe, no provenance UB).
                        if pf_dist != 0 {
                            prefetch_l1((qsb as usize + wbase + c * 32 + pf_dist) as *const u8);
                        }
                        let q4 = vld1q_u8_x2(qsb.add(wbase + c * 32));
                        let lo0 = vreinterpretq_s8_u8(vandq_u8(q4.0, m4b));
                        let lo1 = vreinterpretq_s8_u8(vandq_u8(q4.1, m4b));
                        let pl = vdotq_s32_acc(vdotq_s32_acc(mzero, lo0, q8l.0), lo1, q8l.1);
                        acc[c][0] = vmlaq_n_s32(acc[c][0], pl, *sc_ptr.add(c * 8 + sc_lo) as i32);
                        let hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q4.0, 4));
                        let hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q4.1, 4));
                        let ph = vdotq_s32_acc(vdotq_s32_acc(mzero, hi0, q8h.0), hi1, q8h.1);
                        acc[c][0] = vmlaq_n_s32(acc[c][0], ph, *sc_ptr.add(c * 8 + sc_hi) as i32);
                    }
                    continue;
                }

                // Prefill (m >= 2). LOW-REGISTER-PRESSURE two-pass: each pass holds only this
                // pass's NC weight pairs (NC*2 regs) + the NC*MR accumulators, NOT a held `q4`
                // array on top. The held-`q4` single-load variant won on M1's wide core but on
                // N1 (32 NEON regs, narrower) the extra NC*2 live regs spilled and ate the gain -
                // its prefill matmul instr count was identical to the unoptimized kernel. Here the
                // hi pass RE-loads q4, but it is L1-resident (just read in the lo pass) and the
                // address is the hoisted `qsb + const`, so it is cheap - while pressure stays low.
                // `from_fn` still builds each pass's pair array in place (no dead `movi` zero-init).
                let slo: [i32; NC] =
                    core::array::from_fn(|c| unsafe { *sc_ptr.add(c * 8 + sc_lo) as i32 });
                let shi: [i32; NC] =
                    core::array::from_fn(|c| unsafe { *sc_ptr.add(c * 8 + sc_hi) as i32 });

                let lo: [(int8x16_t, int8x16_t); NC] = core::array::from_fn(|c| unsafe {
                    let q = vld1q_u8_x2(qsb.add(wbase + c * 32));
                    (
                        vreinterpretq_s8_u8(vandq_u8(q.0, m4b)),
                        vreinterpretq_s8_u8(vandq_u8(q.1, m4b)),
                    )
                });
                for a in 0..MR {
                    let q8 = vld1q_s8_x2(rq[a].add(sc_lo * 32));
                    for c in 0..NC {
                        let p = vdotq_s32_acc(vdotq_s32_acc(mzero, lo[c].0, q8.0), lo[c].1, q8.1);
                        acc[c][a] = vmlaq_n_s32(acc[c][a], p, slo[c]);
                    }
                }

                let hi: [(int8x16_t, int8x16_t); NC] = core::array::from_fn(|c| unsafe {
                    let q = vld1q_u8_x2(qsb.add(wbase + c * 32));
                    (
                        vreinterpretq_s8_u8(vshrq_n_u8(q.0, 4)),
                        vreinterpretq_s8_u8(vshrq_n_u8(q.1, 4)),
                    )
                });
                for a in 0..MR {
                    let q8 = vld1q_s8_x2(rq[a].add(sc_hi * 32));
                    for c in 0..NC {
                        let p = vdotq_s32_acc(vdotq_s32_acc(mzero, hi[c].0, q8.0), hi[c].1, q8.1);
                        acc[c][a] = vmlaq_n_s32(acc[c][a], p, shi[c]);
                    }
                }
            }
            for c in 0..NC {
                for a in 0..MR {
                    sumf[c][a] += rd[a] * cd[c] * vaddvq_s32(acc[c][a]) as f32;
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

/// Convenience wrapper: all 8 channels of the packed block x `MR` rows.
#[inline(always)]
pub(crate) fn gemm_q4kx8_q8k<const MR: usize>(
    packed: &[BlockQ4Kx8],
    rows: &[&[BlockQ8K]; MR],
    dst: &mut [f32],
) {
    gemm_q4kx_q8k::<8, MR>(packed, 0, rows, dst)
}

// ===========================================================================
// i8mm (SMMLA) packed Q4_K prefill GEMM. Computes a 4-row x 8-col output tile
// per call (one BlockQ8Kx4 activation series x one BlockQ4Kx8L weight series).
// SMMLA does a 2x2 int8 outer-product-accumulate in one instruction (~4 SDOTs of
// work), the lever for large-M prefill on i8mm hardware (Graviton3/4, not N1).
//
// Faithful port of llama's `ggml_gemm_q4_K_8x8_q8_K` NEON i8mm branch, against
// the byte-identical `BlockQ4Kx8L`/`BlockQ8Kx4` layouts. The SMMLA itself is
// gated on `target_feature = "i8mm"`; without it (e.g. the M1 dev host) a SCALAR
// emulation runs that is bit-identical to the hardware instruction (exact integer
// arithmetic), so a local build validates the exact result i8mm hardware yields.
// ===========================================================================

/// SMMLA: `acc + a(2x8) * b(2x8)^T`, a 2x2 int32 result. Lane r*2+c holds the dot
/// of `a` row r with `b` row c - matching the ARM `smmla` destination layout, so
/// the kernel's output reorder (ported verbatim) is correct on both paths.
// dead_code until the i8mm GEMM is wired into the prefill dispatcher (step 2).
#[allow(dead_code)]
#[cfg(all(target_arch = "aarch64", target_feature = "i8mm"))]
#[inline(always)]
unsafe fn smmla(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    // The `vmmlaq_s32` std intrinsic is unstable (stdarch_neon_i8mm), so emit the
    // SMMLA instruction directly. Safe on stable because `i8mm` is a target feature.
    core::arch::asm!(
        "smmla {acc:v}.4s, {a:v}.16b, {b:v}.16b",
        acc = inout(vreg) acc,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack, preserves_flags),
    );
    acc
}

#[allow(dead_code)]
#[cfg(not(all(target_arch = "aarch64", target_feature = "i8mm")))]
#[inline(always)]
unsafe fn smmla(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    // Scalar twin of `smmla`. Integer arithmetic is exact, so this returns the
    // identical int32 result the hardware instruction would - the whole point of
    // step 1: validate layout + scale logic on a host without i8mm.
    let mut av = [0i8; 16];
    let mut bv = [0i8; 16];
    let mut o = [0i32; 4];
    vst1q_s8(av.as_mut_ptr(), a);
    vst1q_s8(bv.as_mut_ptr(), b);
    vst1q_s32(o.as_mut_ptr(), acc);
    for r in 0..2 {
        for c in 0..2 {
            let mut s = 0i32;
            for k in 0..8 {
                s += av[r * 8 + k] as i32 * bv[c * 8 + k] as i32;
            }
            o[r * 2 + c] += s;
        }
    }
    vld1q_s32(o.as_ptr())
}

/// Unpack one sub-block-pair's 6-bit scales/mins from the laneq 12-byte group into
/// 8 i8 scales + an int16x8 of mins. Port of llama's `decode_q_Kx8_6bit_scales`.
#[allow(dead_code)] // until wired into the prefill dispatcher (step 2)
#[inline(always)]
unsafe fn decode_q_kx8_6bit_scales(
    scales_in: &[u8],
    out_mins: &mut int16x8_t,
    out_scales: &mut [i8; 8],
) {
    const KMASK1: u32 = 0x3f3f_3f3f;
    const KMASK2: u32 = 0x0f0f_0f0f;
    const KMASK3: u32 = 0x0303_0303;
    let sm = [
        u32::from_le_bytes([scales_in[0], scales_in[1], scales_in[2], scales_in[3]]),
        u32::from_le_bytes([scales_in[4], scales_in[5], scales_in[6], scales_in[7]]),
        u32::from_le_bytes([scales_in[8], scales_in[9], scales_in[10], scales_in[11]]),
    ];
    let mins_0_3 = sm[1] & KMASK1;
    let mins_4_7 = ((sm[2] >> 4) & KMASK2) | (((sm[1] >> 6) & KMASK3) << 4);
    let mins_u32 = [mins_0_3, mins_4_7];
    let mins_u8 = vreinterpret_u8_u32(vld1_u32(mins_u32.as_ptr()));
    *out_mins = vreinterpretq_s16_u16(vmovl_u8(mins_u8));
    let s0 = (sm[0] & KMASK1).to_le_bytes();
    let s1 = ((sm[2] & KMASK2) | (((sm[0] >> 6) & KMASK3) << 4)).to_le_bytes();
    for l in 0..4 {
        out_scales[l] = s0[l] as i8;
        out_scales[4 + l] = s1[l] as i8;
    }
}

/// Packed Q4_K i8mm GEMM: 4 activation rows x 8 weight columns, over `nb`
/// super-blocks. Writes a row-major 4x8 tile: `dst[row * 8 + col]`. `q4`/`q8`
/// must have the same length `nb`. Bit-exact (mod f32 accumulation order) to the
/// scalar Q4_K dot. See module note for the i8mm gating / scalar-emulation story.
#[allow(dead_code)] // until wired into the prefill dispatcher (step 2)
pub(crate) fn gemm_q4kx8_q8k_i8mm(q4: &[BlockQ4Kx8L], q8: &[BlockQ8Kx4], dst: &mut [f32]) {
    let nb = q4.len();
    debug_assert!(q8.len() == nb && dst.len() == 32);
    unsafe {
        let m4b = vdupq_n_u8(0x0f);
        let mut acc_f32 = [vdupq_n_f32(0.0); 8];
        for b in 0..nb {
            let q4b = &q4[b];
            let q8b = &q8[b];
            // Per q8 row: pairwise-add the 16 bsums into 8 sub-block sums.
            let mut bsums_arr = [[0i16; 8]; 4];
            for r in 0..4 {
                let p = q8b.bsums.as_ptr().add(16 * r);
                let v = vpaddq_s16(vld1q_s16(p), vld1q_s16(p.add(8)));
                vst1q_s16(bsums_arr[r].as_mut_ptr(), v);
            }

            let mut acc = [vdupq_n_s32(0); 8]; // raw i8mm sums (rows 01 in 0..4, 23 in 4..8)
            let mut bias_acc = [vdupq_n_s32(0); 8]; // interleaved mins*bsum bias

            for sb in 0..QK_K / 64 {
                // Scales/mins for the lo (i=0) and hi (i=1) nibble sub-blocks.
                let mut q4sb_scales = [[0i8; 8]; 2];
                let mut q4sb_mins = [vdupq_n_s16(0); 2];
                for i in 0..2 {
                    let offset = sb * 24 + i * 12;
                    decode_q_kx8_6bit_scales(
                        &q4b.scales[offset..],
                        &mut q4sb_mins[i],
                        &mut q4sb_scales[i],
                    );
                }

                // Q8: 8 interleaved (row01 | row23) 16-byte runs for this sub-block.
                let q8_base = q8b.qs.as_ptr().add(sb * 256);
                let mut q8_01 = [vdupq_n_s8(0); 8];
                let mut q8_23 = [vdupq_n_s8(0); 8];
                for i in 0..8 {
                    let off = i * 32;
                    q8_01[i] = vld1q_s8(q8_base.add(off));
                    q8_23[i] = vld1q_s8(q8_base.add(off + 16));
                }
                let q8s = [q8_01, q8_23];

                // Weight columns iterated in pairs (01, 23, 45, 67).
                for cp in 0..4 {
                    let mut sb_acc = [vdupq_n_s32(0); 4];
                    let base = q4b.qs.as_ptr().add(sb * QK_K + 16 * cp);
                    let q0 = vld1q_u8(base);
                    let q1 = vld1q_u8(base.add(64));
                    let q2 = vld1q_u8(base.add(128));
                    let q3 = vld1q_u8(base.add(192));
                    let q4n = [
                        [
                            vreinterpretq_s8_u8(vandq_u8(q0, m4b)),
                            vreinterpretq_s8_u8(vandq_u8(q1, m4b)),
                            vreinterpretq_s8_u8(vandq_u8(q2, m4b)),
                            vreinterpretq_s8_u8(vandq_u8(q3, m4b)),
                        ],
                        [
                            vreinterpretq_s8_u8(vshrq_n_u8(q0, 4)),
                            vreinterpretq_s8_u8(vshrq_n_u8(q1, 4)),
                            vreinterpretq_s8_u8(vshrq_n_u8(q2, 4)),
                            vreinterpretq_s8_u8(vshrq_n_u8(q3, 4)),
                        ],
                    ];
                    for rp in 0..2 {
                        for blk in 0..2 {
                            let q8r = &q8s[rp][4 * blk..];
                            let mut a = sb_acc[2 * rp + blk];
                            for o in 0..4 {
                                a = smmla(a, q4n[blk][o], q8r[o]);
                            }
                            sb_acc[2 * rp + blk] = a;
                        }
                    }
                    let so = cp * 2;
                    let bscale0 = vcombine_s32(
                        vdup_n_s32(q4sb_scales[0][so] as i32),
                        vdup_n_s32(q4sb_scales[0][so + 1] as i32),
                    );
                    let bscale1 = vcombine_s32(
                        vdup_n_s32(q4sb_scales[1][so] as i32),
                        vdup_n_s32(q4sb_scales[1][so + 1] as i32),
                    );
                    acc[cp] = vmlaq_s32(acc[cp], sb_acc[0], bscale0);
                    acc[cp + 4] = vmlaq_s32(acc[cp + 4], sb_acc[2], bscale0);
                    acc[cp] = vmlaq_s32(acc[cp], sb_acc[1], bscale1);
                    acc[cp + 4] = vmlaq_s32(acc[cp + 4], sb_acc[3], bscale1);
                }

                // Bias: mins . bsums, accumulated per (row, lo/hi col half).
                for r in 0..4 {
                    let blo = vdup_n_s16(bsums_arr[sb][r * 2]);
                    let bhi = vdup_n_s16(bsums_arr[sb][r * 2 + 1]);
                    bias_acc[2 * r] = vmlal_s16(bias_acc[2 * r], blo, vget_low_s16(q4sb_mins[0]));
                    bias_acc[2 * r] = vmlal_s16(bias_acc[2 * r], bhi, vget_low_s16(q4sb_mins[1]));
                    bias_acc[2 * r + 1] =
                        vmlal_s16(bias_acc[2 * r + 1], blo, vget_high_s16(q4sb_mins[0]));
                    bias_acc[2 * r + 1] =
                        vmlal_s16(bias_acc[2 * r + 1], bhi, vget_high_s16(q4sb_mins[1]));
                }
            }

            // SMMLA leaves each acc as a 2x2 row-interleaved tile; deinterleave it
            // back to (row, col-quad) order before the f32 scale/bias combine.
            for i in 0..8 {
                let aux = vzip_s32(vget_low_s32(acc[i]), vget_high_s32(acc[i]));
                acc[i] = vcombine_s32(aux.0, aux.1);
            }
            let reorder_acc = [
                vcombine_s32(vget_low_s32(acc[0]), vget_low_s32(acc[1])),
                vcombine_s32(vget_low_s32(acc[2]), vget_low_s32(acc[3])),
                vcombine_s32(vget_high_s32(acc[0]), vget_high_s32(acc[1])),
                vcombine_s32(vget_high_s32(acc[2]), vget_high_s32(acc[3])),
                vcombine_s32(vget_low_s32(acc[4]), vget_low_s32(acc[5])),
                vcombine_s32(vget_low_s32(acc[6]), vget_low_s32(acc[7])),
                vcombine_s32(vget_high_s32(acc[4]), vget_high_s32(acc[5])),
                vcombine_s32(vget_high_s32(acc[6]), vget_high_s32(acc[7])),
            ];
            for i in 0..4 {
                for j in 0..2 {
                    let q8_d = vdupq_n_f32(q8b.d[i]);
                    let q4_dmin_arr: [f32; 4] = core::array::from_fn(|l| q4b.dmin[j * 4 + l].to_f32());
                    let q4_d_arr: [f32; 4] = core::array::from_fn(|l| q4b.d[j * 4 + l].to_f32());
                    let dmins = vmulq_f32(vld1q_f32(q4_dmin_arr.as_ptr()), q8_d);
                    let scale = vmulq_f32(vld1q_f32(q4_d_arr.as_ptr()), q8_d);
                    acc_f32[2 * i + j] =
                        vmlsq_f32(acc_f32[2 * i + j], vcvtq_f32_s32(bias_acc[2 * i + j]), dmins);
                    acc_f32[2 * i + j] =
                        vmlaq_f32(acc_f32[2 * i + j], vcvtq_f32_s32(reorder_acc[2 * i + j]), scale);
                }
            }
        }
        for i in 0..4 {
            for j in 0..2 {
                vst1q_f32(dst.as_mut_ptr().add(i * 8 + j * 4), acc_f32[2 * i + j]);
            }
        }
    }
}

/// Lane-indexed SDOT: `acc += dot4(a[i], b[LANE])` per output lane i, where b's
/// 4-byte group `LANE` is broadcast as the multiplier. Emits `SDOT (by element)`
/// directly (the std `vdotq_laneq_s32` intrinsic is unstable). Safe on stable:
/// `dotprod` is a target feature. Used by the lane=row GEMM where LANE selects the
/// activation ROW, so one weight load feeds all 4 rows of the tile.
#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
#[allow(dead_code)]
#[inline(always)]
unsafe fn vdot_laneq<const LANE: i32>(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    core::arch::asm!(
        "sdot {acc:v}.4s, {a:v}.16b, {b:v}.4b[{lane}]",
        acc = inout(vreg) acc,
        a = in(vreg) a,
        b = in(vreg) b,
        lane = const LANE,
        options(pure, nomem, nostack, preserves_flags),
    );
    acc
}

/// Packed Q4_K prefill GEMM, lane=row SDOT: 8 output channels x 4 activation rows,
/// writing a row-major 4x8 tile `dst[row*8 + col]` (same as `gemm_q4kx8_q8k_i8mm`,
/// so it shares that driver/scatter). This is llama's ACTUAL N1 prefill kernel
/// (`ggml_gemm_q4_K_8x4_q8_K`, DOTPROD branch) - NOT i8mm, runs on any dotprod core
/// incl. N1. The win vs candle's `gemm_q4kx_q8k`: the activation is the row-
/// interleaved `BlockQ8Kx4` (4x4 pack) so the SDOT LANE selects the row - one
/// weight load (`q4_0123_lo`) feeds all 4 rows via 4 lane dots - AND the 6-bit
/// scale is folded to f32 PER SUB-BLOCK, so only ~16 live int accumulators are
/// needed for the 8-col x 4-row tile (candle's kernel holds 32 scaled-int accs
/// across all chunks, which spills the 32 NEON regs at this tile width). Consumes
/// the `BlockQ4Kx8L` weight (interleave 8) + `BlockQ8Kx4` from `quantize_mat_q8_k_4x4`.
/// Bit-exact (mod f32 reassociation) to the scalar Q4_K dot.
#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
#[allow(dead_code)] // wired into the prefill dispatcher in a follow-up step
pub(crate) fn gemm_q4kx8_q8k_lanerow(q4: &[BlockQ4Kx8L], q8: &[BlockQ8Kx4], dst: &mut [f32]) {
    let nb = q4.len();
    debug_assert!(q8.len() == nb && dst.len() == 32);
    unsafe {
        let m4b = vdupq_n_u8(0x0f);
        // f32 output accumulators: acc_f32[2*row + cg], cg 0 = cols 0123, 1 = 4567.
        let mut acc_f32 = [vdupq_n_f32(0.0); 8];
        for b in 0..nb {
            let q4b = &q4[b];
            let q8b = &q8[b];
            let q4d0 = {
                let a: [f32; 4] = core::array::from_fn(|l| q4b.d[l].to_f32());
                vld1q_f32(a.as_ptr())
            };
            let q4d1 = {
                let a: [f32; 4] = core::array::from_fn(|l| q4b.d[4 + l].to_f32());
                vld1q_f32(a.as_ptr())
            };
            let q4m0 = {
                let a: [f32; 4] = core::array::from_fn(|l| q4b.dmin[l].to_f32());
                vld1q_f32(a.as_ptr())
            };
            let q4m1 = {
                let a: [f32; 4] = core::array::from_fn(|l| q4b.dmin[4 + l].to_f32());
                vld1q_f32(a.as_ptr())
            };
            let q8d = vld1q_f32(q8b.d.as_ptr());
            // Per-row super-block scale/min (q4_d * q8_d[row]).
            let sbd_s0 = [
                vmulq_laneq_f32(q4d0, q8d, 0),
                vmulq_laneq_f32(q4d0, q8d, 1),
                vmulq_laneq_f32(q4d0, q8d, 2),
                vmulq_laneq_f32(q4d0, q8d, 3),
            ];
            let sbd_s1 = [
                vmulq_laneq_f32(q4d1, q8d, 0),
                vmulq_laneq_f32(q4d1, q8d, 1),
                vmulq_laneq_f32(q4d1, q8d, 2),
                vmulq_laneq_f32(q4d1, q8d, 3),
            ];
            let sbd_m0 = [
                vmulq_laneq_f32(q4m0, q8d, 0),
                vmulq_laneq_f32(q4m0, q8d, 1),
                vmulq_laneq_f32(q4m0, q8d, 2),
                vmulq_laneq_f32(q4m0, q8d, 3),
            ];
            let sbd_m1 = [
                vmulq_laneq_f32(q4m1, q8d, 0),
                vmulq_laneq_f32(q4m1, q8d, 1),
                vmulq_laneq_f32(q4m1, q8d, 2),
                vmulq_laneq_f32(q4m1, q8d, 3),
            ];
            let mut bsums_arr = [[0i16; 8]; 4];
            for r in 0..4 {
                let p = q8b.bsums.as_ptr().add(16 * r);
                let v = vpaddq_s16(vld1q_s16(p), vld1q_s16(p.add(8)));
                vst1q_s16(bsums_arr[r].as_mut_ptr(), v);
            }
            let mut bias_acc = [vdupq_n_s32(0); 8];
            for sb in 0..QK_K / 64 {
                let mut acc_lo = [vdupq_n_s32(0); 8]; // [row]=cols0123, [row+4]=4567
                let mut acc_hi = [vdupq_n_s32(0); 8];
                let mut q4sb_scales = [vdupq_n_s16(0); 2];
                let mut q4sb_mins = [vdupq_n_s16(0); 2];
                for i in 0..2 {
                    let mut aux = [0i8; 8];
                    decode_q_kx8_6bit_scales(
                        &q4b.scales[sb * 24 + i * 12..],
                        &mut q4sb_mins[i],
                        &mut aux,
                    );
                    q4sb_scales[i] = vmovl_s8(vld1_s8(aux.as_ptr()));
                }
                for k in 0..8 {
                    let q8_blk0 = vld1q_s8(q8b.qs.as_ptr().add(sb * 256 + 16 * k));
                    let q8_blk1 = vld1q_s8(q8b.qs.as_ptr().add(sb * 256 + 16 * k + 128));
                    let q4_0123 = vld1q_u8(q4b.qs.as_ptr().add(sb * QK_K + 32 * k));
                    let q4_4567 = vld1q_u8(q4b.qs.as_ptr().add(sb * QK_K + 32 * k + 16));
                    let lo0 = vreinterpretq_s8_u8(vandq_u8(q4_0123, m4b));
                    let hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q4_0123, 4));
                    let lo1 = vreinterpretq_s8_u8(vandq_u8(q4_4567, m4b));
                    let hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q4_4567, 4));
                    // One weight load feeds all 4 rows; LANE = row.
                    acc_lo[0] = vdot_laneq::<0>(acc_lo[0], lo0, q8_blk0);
                    acc_lo[1] = vdot_laneq::<1>(acc_lo[1], lo0, q8_blk0);
                    acc_lo[2] = vdot_laneq::<2>(acc_lo[2], lo0, q8_blk0);
                    acc_lo[3] = vdot_laneq::<3>(acc_lo[3], lo0, q8_blk0);
                    acc_hi[0] = vdot_laneq::<0>(acc_hi[0], hi0, q8_blk1);
                    acc_hi[1] = vdot_laneq::<1>(acc_hi[1], hi0, q8_blk1);
                    acc_hi[2] = vdot_laneq::<2>(acc_hi[2], hi0, q8_blk1);
                    acc_hi[3] = vdot_laneq::<3>(acc_hi[3], hi0, q8_blk1);
                    acc_lo[4] = vdot_laneq::<0>(acc_lo[4], lo1, q8_blk0);
                    acc_lo[5] = vdot_laneq::<1>(acc_lo[5], lo1, q8_blk0);
                    acc_lo[6] = vdot_laneq::<2>(acc_lo[6], lo1, q8_blk0);
                    acc_lo[7] = vdot_laneq::<3>(acc_lo[7], lo1, q8_blk0);
                    acc_hi[4] = vdot_laneq::<0>(acc_hi[4], hi1, q8_blk1);
                    acc_hi[5] = vdot_laneq::<1>(acc_hi[5], hi1, q8_blk1);
                    acc_hi[6] = vdot_laneq::<2>(acc_hi[6], hi1, q8_blk1);
                    acc_hi[7] = vdot_laneq::<3>(acc_hi[7], hi1, q8_blk1);
                }
                // Fold the 6-bit scale to f32 for THIS sub-block (keeps live int
                // accumulators at 16, so the 8x4 tile fits in registers).
                let sc0_lo = vmovl_s16(vget_low_s16(q4sb_scales[0]));
                let sc1_lo = vmovl_s16(vget_high_s16(q4sb_scales[0]));
                let sc0_hi = vmovl_s16(vget_low_s16(q4sb_scales[1]));
                let sc1_hi = vmovl_s16(vget_high_s16(q4sb_scales[1]));
                for row in 0..4 {
                    let sumf0 = vcvtq_f32_s32(vaddq_s32(
                        vmulq_s32(sc0_lo, acc_lo[row]),
                        vmulq_s32(sc0_hi, acc_hi[row]),
                    ));
                    acc_f32[2 * row] = vfmaq_f32(acc_f32[2 * row], sbd_s0[row], sumf0);
                    let sumf1 = vcvtq_f32_s32(vaddq_s32(
                        vmulq_s32(sc1_lo, acc_lo[row + 4]),
                        vmulq_s32(sc1_hi, acc_hi[row + 4]),
                    ));
                    acc_f32[2 * row + 1] = vfmaq_f32(acc_f32[2 * row + 1], sbd_s1[row], sumf1);
                    let blo = vdup_n_s16(bsums_arr[sb][row * 2]);
                    let bhi = vdup_n_s16(bsums_arr[sb][row * 2 + 1]);
                    bias_acc[2 * row] = vmlal_s16(bias_acc[2 * row], blo, vget_low_s16(q4sb_mins[0]));
                    bias_acc[2 * row] = vmlal_s16(bias_acc[2 * row], bhi, vget_low_s16(q4sb_mins[1]));
                    bias_acc[2 * row + 1] =
                        vmlal_s16(bias_acc[2 * row + 1], blo, vget_high_s16(q4sb_mins[0]));
                    bias_acc[2 * row + 1] =
                        vmlal_s16(bias_acc[2 * row + 1], bhi, vget_high_s16(q4sb_mins[1]));
                }
            }
            for row in 0..4 {
                acc_f32[2 * row] =
                    vmlsq_f32(acc_f32[2 * row], vcvtq_f32_s32(bias_acc[2 * row]), sbd_m0[row]);
                acc_f32[2 * row + 1] = vmlsq_f32(
                    acc_f32[2 * row + 1],
                    vcvtq_f32_s32(bias_acc[2 * row + 1]),
                    sbd_m1[row],
                );
            }
        }
        for row in 0..4 {
            vst1q_f32(dst.as_mut_ptr().add(row * 8), acc_f32[2 * row]);
            vst1q_f32(dst.as_mut_ptr().add(row * 8 + 4), acc_f32[2 * row + 1]);
        }
    }
}

/// Packed Q4_K prefill GEMM via interleaved 8-column SDOT with a duplicated q8
/// load: 8 output channels x MR activation rows, writing `dst[c*MR + a]` (same
/// layout as `gemm_q4kx8_q8k`, so it drops into the same driver/scatter). Each
/// weight column-pair (16 bytes = col 2cp low half, col 2cp+1 high half) is loaded
/// ONCE and reused across the MR rows; the activation 8-byte group is duplicated
/// into both 64-bit halves (`vld1q_dup_s64`) so one `vdotq_s32` dots it against
/// both columns at once. Fewer distinct q8 loads / dots than the per-channel
/// `gemm_q4kx_q8k` - aimed at N1's narrow, register-starved core (M1's wide core
/// sees less). Faithful port of llama's `ggml_gemv_q4_K_8x8_q8_K` DOTPROD inner
/// loop (decode/M=1) generalized to MR rows with the column loads hoisted out of
/// the row loop (the prefill reuse). Consumes the `BlockQ4Kx8L` (interleave=8)
/// weight + standard `BlockQ8K` rows; same Q8K quant as baseline, so it is
/// ~bit-exact (only f32 reassociation differs).
#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
#[allow(dead_code)] // wired into the prefill dispatcher in a follow-up step
pub(crate) fn gemm_q4kx8_q8k_dup<const MR: usize>(
    packed: &[BlockQ4Kx8L],
    rows: &[&[BlockQ8K]; MR],
    dst: &mut [f32],
) {
    debug_assert!(dst.len() == 8 * MR);
    let nb = packed.len();
    unsafe {
        let m4b = vdupq_n_u8(0x0f);
        // f32 output accumulators: per row, 2 col-groups (cols 0123, 4567).
        let mut accf = [[vdupq_n_f32(0.0); 2]; MR];
        for b in 0..nb {
            let q4 = &packed[b];
            // f16 super-block scales/mins -> f32 (shared across sub-blocks/rows).
            let q4d0 = {
                let a: [f32; 4] = core::array::from_fn(|l| q4.d[l].to_f32());
                vld1q_f32(a.as_ptr())
            };
            let q4d1 = {
                let a: [f32; 4] = core::array::from_fn(|l| q4.d[4 + l].to_f32());
                vld1q_f32(a.as_ptr())
            };
            let q4m0 = {
                let a: [f32; 4] = core::array::from_fn(|l| q4.dmin[l].to_f32());
                vld1q_f32(a.as_ptr())
            };
            let q4m1 = {
                let a: [f32; 4] = core::array::from_fn(|l| q4.dmin[4 + l].to_f32());
                vld1q_f32(a.as_ptr())
            };
            // per-row super-block scale (q4_d*q8_d), min (q4_dmin*q8_d), and bsums.
            let mut sb_scale = [[vdupq_n_f32(0.0); 2]; MR];
            let mut sb_min = [[vdupq_n_f32(0.0); 2]; MR];
            let mut bsums_arr = [[0i16; 8]; MR];
            for r in 0..MR {
                let q8d = vdupq_n_f32(rows[r][b].d);
                sb_scale[r][0] = vmulq_f32(q4d0, q8d);
                sb_scale[r][1] = vmulq_f32(q4d1, q8d);
                sb_min[r][0] = vmulq_f32(q4m0, q8d);
                sb_min[r][1] = vmulq_f32(q4m1, q8d);
                let bp = rows[r][b].bsums.as_ptr();
                let bs = vpaddq_s16(vld1q_s16(bp), vld1q_s16(bp.add(8)));
                vst1q_s16(bsums_arr[r].as_mut_ptr(), bs);
            }
            let mut bias = [[vdupq_n_s32(0); 2]; MR]; // [r][0]=cols0123, [1]=4567
            for sb in 0..QK_K / 64 {
                // 6-bit scales/mins for the lo (i=0) / hi (i=1) nibble sub-blocks.
                let mut q4sb_scales = [vdupq_n_s16(0); 2];
                let mut q4sb_mins = [vdupq_n_s16(0); 2];
                for i in 0..2 {
                    let mut aux = [0i8; 8];
                    let off = sb * 24 + i * 12;
                    decode_q_kx8_6bit_scales(&q4.scales[off..], &mut q4sb_mins[i], &mut aux);
                    q4sb_scales[i] = vmovl_s8(vld1_s8(aux.as_ptr()));
                }
                let q4_base = q4.qs.as_ptr().add(sb * QK_K);
                // acc_lo/acc_hi[r][cp]: column pair cp (cols 2cp, 2cp+1) - lanes
                // 0,1 -> col 2cp partials, lanes 2,3 -> col 2cp+1 partials.
                let mut acc_lo = [[vdupq_n_s32(0); 4]; MR];
                let mut acc_hi = [[vdupq_n_s32(0); 4]; MR];
                for cp in 0..4 {
                    // One column-pair's q4: 4x 16-byte reads (col 2cp | col 2cp+1),
                    // hoisted out of the row loop so MR rows reuse them.
                    let c0 = vld1q_u8(q4_base.add(16 * cp));
                    let c1 = vld1q_u8(q4_base.add(16 * cp + 64));
                    let c2 = vld1q_u8(q4_base.add(16 * cp + 128));
                    let c3 = vld1q_u8(q4_base.add(16 * cp + 192));
                    let lo0 = vreinterpretq_s8_u8(vandq_u8(c0, m4b));
                    let lo1 = vreinterpretq_s8_u8(vandq_u8(c1, m4b));
                    let lo2 = vreinterpretq_s8_u8(vandq_u8(c2, m4b));
                    let lo3 = vreinterpretq_s8_u8(vandq_u8(c3, m4b));
                    let hi0 = vreinterpretq_s8_u8(vshrq_n_u8(c0, 4));
                    let hi1 = vreinterpretq_s8_u8(vshrq_n_u8(c1, 4));
                    let hi2 = vreinterpretq_s8_u8(vshrq_n_u8(c2, 4));
                    let hi3 = vreinterpretq_s8_u8(vshrq_n_u8(c3, 4));
                    for r in 0..MR {
                        let q8b = rows[r][b].qs.as_ptr().add(sb * 64);
                        // Each q8 8-byte group duplicated into both halves so one
                        // dot covers both columns of the pair.
                        let d = |o: usize| {
                            vreinterpretq_s8_s64(vld1q_dup_s64(q8b.add(o) as *const i64))
                        };
                        let mut al = acc_lo[r][cp];
                        al = vdotq_s32_acc(al, lo0, d(0));
                        al = vdotq_s32_acc(al, lo1, d(8));
                        al = vdotq_s32_acc(al, lo2, d(16));
                        al = vdotq_s32_acc(al, lo3, d(24));
                        acc_lo[r][cp] = al;
                        let mut ah = acc_hi[r][cp];
                        ah = vdotq_s32_acc(ah, hi0, d(32));
                        ah = vdotq_s32_acc(ah, hi1, d(40));
                        ah = vdotq_s32_acc(ah, hi2, d(48));
                        ah = vdotq_s32_acc(ah, hi3, d(56));
                        acc_hi[r][cp] = ah;
                    }
                }
                // 6-bit scales per col-group (lo/hi nibble), then bias.
                let sc0_lo = vmovl_s16(vget_low_s16(q4sb_scales[0])); // cols 0123 lo
                let sc0_hi = vmovl_s16(vget_low_s16(q4sb_scales[1])); // cols 0123 hi
                let sc1_lo = vmovl_s16(vget_high_s16(q4sb_scales[0])); // cols 4567 lo
                let sc1_hi = vmovl_s16(vget_high_s16(q4sb_scales[1])); // cols 4567 hi
                for r in 0..MR {
                    // vpaddq folds a pair's (col,col+1) lane partials into 4 columns.
                    let lo_0123 = vpaddq_s32(acc_lo[r][0], acc_lo[r][1]);
                    let hi_0123 = vpaddq_s32(acc_hi[r][0], acc_hi[r][1]);
                    accf[r][0] = vfmaq_f32(
                        accf[r][0],
                        sb_scale[r][0],
                        vcvtq_f32_s32(vmulq_s32(sc0_lo, lo_0123)),
                    );
                    accf[r][0] = vfmaq_f32(
                        accf[r][0],
                        sb_scale[r][0],
                        vcvtq_f32_s32(vmulq_s32(sc0_hi, hi_0123)),
                    );
                    let lo_4567 = vpaddq_s32(acc_lo[r][2], acc_lo[r][3]);
                    let hi_4567 = vpaddq_s32(acc_hi[r][2], acc_hi[r][3]);
                    accf[r][1] = vfmaq_f32(
                        accf[r][1],
                        sb_scale[r][1],
                        vcvtq_f32_s32(vmulq_s32(sc1_lo, lo_4567)),
                    );
                    accf[r][1] = vfmaq_f32(
                        accf[r][1],
                        sb_scale[r][1],
                        vcvtq_f32_s32(vmulq_s32(sc1_hi, hi_4567)),
                    );
                    let blo = vdup_n_s16(bsums_arr[r][2 * sb]);
                    let bhi = vdup_n_s16(bsums_arr[r][2 * sb + 1]);
                    bias[r][0] = vmlal_s16(bias[r][0], blo, vget_low_s16(q4sb_mins[0]));
                    bias[r][0] = vmlal_s16(bias[r][0], bhi, vget_low_s16(q4sb_mins[1]));
                    bias[r][1] = vmlal_s16(bias[r][1], blo, vget_high_s16(q4sb_mins[0]));
                    bias[r][1] = vmlal_s16(bias[r][1], bhi, vget_high_s16(q4sb_mins[1]));
                }
            }
            for r in 0..MR {
                accf[r][0] = vmlsq_f32(accf[r][0], vcvtq_f32_s32(bias[r][0]), sb_min[r][0]);
                accf[r][1] = vmlsq_f32(accf[r][1], vcvtq_f32_s32(bias[r][1]), sb_min[r][1]);
            }
        }
        // Scatter: accf[r][0] = cols 0..3, accf[r][1] = cols 4..7; dst[c*MR + r].
        for r in 0..MR {
            let mut c0 = [0f32; 4];
            let mut c1 = [0f32; 4];
            vst1q_f32(c0.as_mut_ptr(), accf[r][0]);
            vst1q_f32(c1.as_mut_ptr(), accf[r][1]);
            for c in 0..4 {
                dst[c * MR + r] = c0[c];
                dst[(c + 4) * MR + r] = c1[c];
            }
        }
    }
}

/// Packed Q6_K GEMV/GEMM: 8 output channels (one `BlockQ6Kx8`) x MR activation
/// rows, writing `dst[c*MR + a]`. Mirrors `vec_dot_q6k_q8k` op-for-op (ql+qh
/// unpack, i8 scales, `-32*isum_mins` bias) so it is bit-identical to the scalar
/// Q6_K dot - but unpacks each channel's q6 once per chunk and reuses it across
/// the MR rows, and reads the 8 interleaved rows of a chunk from one contiguous run.
pub(crate) fn gemm_q6kx8_q8k<const MR: usize>(
    packed: &[BlockQ6Kx8],
    rows: &[&[BlockQ8K]; MR],
    dst: &mut [f32],
) {
    const NC: usize = 8;
    const QLC: usize = QK_K / 4; // 64 ql bytes / chunk
    const QHC: usize = QK_K / 8; // 32 qh bytes / chunk
    const NSC: usize = QK_K / 16; // 16 scales / super-block
    debug_assert!(dst.len() == NC * MR);
    let nb = packed.len();
    let mut sumf = [[0f32; MR]; NC];
    unsafe {
        let m4b = vdupq_n_u8(0xF);
        let mone = vdupq_n_u8(3);
        for i in 0..nb {
            let blk = &packed[i];
            // Hoist base pointers once per block (same address-arithmetic win as the Q4 kernel):
            // inner addresses become `base + const` instead of re-walking `blk.*.as_ptr().add(..)`
            // and re-indexing `rows[a][i]` every chunk.
            let qh_ptr = blk.qh.as_ptr();
            let ql_ptr = blk.ql.as_ptr();
            let sc_ptr = blk.scales.as_ptr();
            let mut rq = [core::ptr::null::<i8>(); MR];
            let mut rd = [0f32; MR];
            for a in 0..MR {
                rq[a] = rows[a][i].qs.as_ptr();
                rd[a] = rows[a][i].d;
            }
            // Bias: isum_mins[c][a] = sum_s scales[c][s] * bsums[a][s] (16 sub-blocks).
            let mut bsums = [(vdupq_n_s16(0), vdupq_n_s16(0)); MR];
            for a in 0..MR {
                let b = rows[a][i].bsums.as_ptr();
                bsums[a] = (vld1q_s16(b), vld1q_s16(b.add(8)));
            }
            let mut isum_mins = [[0i32; MR]; NC];
            for c in 0..NC {
                let sc8 = vld1q_s8(sc_ptr.add(c * NSC));
                let s_lo = vmovl_s8(vget_low_s8(sc8));
                let s_hi = vmovl_s8(vget_high_s8(sc8));
                for a in 0..MR {
                    let prod = vaddq_s32(
                        vaddq_s32(
                            vmull_s16(vget_low_s16(bsums[a].0), vget_low_s16(s_lo)),
                            vmull_s16(vget_high_s16(bsums[a].0), vget_high_s16(s_lo)),
                        ),
                        vaddq_s32(
                            vmull_s16(vget_low_s16(bsums[a].1), vget_low_s16(s_hi)),
                            vmull_s16(vget_high_s16(bsums[a].1), vget_high_s16(s_hi)),
                        ),
                    );
                    isum_mins[c][a] = vaddvq_s32(prod);
                }
            }
            // Main: isum[c][a].
            let mut isum = [[0i32; MR]; NC];
            for j in 0..QK_K / 128 {
                let ql_base = j * (NC * QLC);
                let qh_base = j * (NC * QHC);
                for c in 0..NC {
                    let qhbits = vld1q_u8_x2(qh_ptr.add(qh_base + c * QHC));
                    let q6bits = vld1q_u8_x4(ql_ptr.add(ql_base + c * QLC));
                    let sc = sc_ptr.add(c * NSC + j * 8);
                    // First half: low nibbles + low two-bit highs.
                    let q6h_0 = vshlq_n_u8(vandq_u8(mone, qhbits.0), 4);
                    let q6h_1 = vshlq_n_u8(vandq_u8(mone, qhbits.1), 4);
                    let q6h_2 = vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhbits.0, 2)), 4);
                    let q6h_3 = vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhbits.1, 2)), 4);
                    let b0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.0, m4b), q6h_0));
                    let b1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.1, m4b), q6h_1));
                    let b2 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.2, m4b), q6h_2));
                    let b3 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.3, m4b), q6h_3));
                    for a in 0..MR {
                        let q8 = vld1q_s8_x4(rq[a].add(j * 128));
                        let p0 = vdotq_s32(b0, q8.0);
                        let p1 = vdotq_s32(b1, q8.1);
                        isum[c][a] +=
                            vaddvq_s32(p0) * (*sc as i32) + vaddvq_s32(p1) * (*sc.add(1) as i32);
                        let p2 = vdotq_s32(b2, q8.2);
                        let p3 = vdotq_s32(b3, q8.3);
                        isum[c][a] += vaddvq_s32(p2) * (*sc.add(2) as i32)
                            + vaddvq_s32(p3) * (*sc.add(3) as i32);
                    }
                    // Second half: high nibbles + high two-bit highs.
                    let q6h_0 = vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhbits.0, 4)), 4);
                    let q6h_1 = vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhbits.1, 4)), 4);
                    let q6h_2 = vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhbits.0, 6)), 4);
                    let q6h_3 = vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhbits.1, 6)), 4);
                    let b0 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.0, 4), q6h_0));
                    let b1 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.1, 4), q6h_1));
                    let b2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.2, 4), q6h_2));
                    let b3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.3, 4), q6h_3));
                    for a in 0..MR {
                        let q8 = vld1q_s8_x4(rq[a].add(j * 128 + 64));
                        let p0 = vdotq_s32(b0, q8.0);
                        let p1 = vdotq_s32(b1, q8.1);
                        isum[c][a] += vaddvq_s32(p0) * (*sc.add(4) as i32)
                            + vaddvq_s32(p1) * (*sc.add(5) as i32);
                        let p2 = vdotq_s32(b2, q8.2);
                        let p3 = vdotq_s32(b3, q8.3);
                        isum[c][a] += vaddvq_s32(p2) * (*sc.add(6) as i32)
                            + vaddvq_s32(p3) * (*sc.add(7) as i32);
                    }
                }
            }
            for c in 0..NC {
                let dc = blk.d[c].to_f32();
                for a in 0..MR {
                    sumf[c][a] += dc * rd[a] * ((isum[c][a] - 32 * isum_mins[c][a]) as f32);
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
