// Hand-asm prototype for the prefill Q4_K chunk loop - the 16-accumulator (NC=4 x MR=4)
// configuration that SPILLS under LLVM on N1 and carries the address/movi overhead the
// annotate found. One "lo pass" of the production gemm_q4kx_q8k: per chunk, load 4 channels'
// packed weights, unpack the low nibbles (`and`), and for each of 4 rows do sdot-pair + scaled
// accumulate into 16 pinned accumulators. (lo-only keeps the asm ~half-size; it's the same
// register pressure and instruction mix as the full kernel.)
//
//   intrinsic : what LLVM emits today (spills v0..v15 + temps, recomputes addresses, re-movi)
//   asm       : 16 acc pinned v0..v15, mask pinned v26, pointer auto-increment, scales by lane
//
// Must agree bit-for-bit. The asm should win on N1 (no spills, no address math); M1's wide core
// may not care. Validates the technique before it goes into gemm_q4kx_q8k.
//   cargo run --release --example hand_asm_probe
#![allow(unused)]
use std::arch::aarch64::*;
use std::time::Instant;

const NB: usize = 8; // super-blocks
const CHUNKS: usize = NB * 4; // 4 chunks / super-block
const WBYTES: usize = CHUNKS * 4 * 32; // 4 channels * 32 bytes / chunk
const ABYTES: usize = CHUNKS * 4 * 32; // 4 rows * 32 bytes / chunk
const NSCALE: usize = CHUNKS * 4; // 4 channel scales / chunk

fn fill_u8(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (s >> 56) as u8
        })
        .collect()
}
fn fill_i8(n: usize, seed: u64) -> Vec<i8> {
    fill_u8(n, seed).into_iter().map(|x| (x as i8) >> 1).collect()
}
fn fill_i32(n: usize, seed: u64) -> Vec<i32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 58) as i32) - 16 // small signed scales
        })
        .collect()
}

#[inline(always)]
unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let mut out = acc;
    std::arch::asm!("sdot {0:v}.4s, {1:v}.16b, {2:v}.16b",
        inout(vreg) out, in(vreg) a, in(vreg) b, options(pure, nomem, nostack));
    out
}

// Intrinsic reference: 16 accumulators, nibble-unpack + scaled accumulate.
#[target_feature(enable = "neon,dotprod")]
unsafe fn q4lo_intrinsic(w: *const u8, a: *const i8, s: *const i32) -> [i32; 16] {
    let m4b = vdupq_n_u8(0xF);
    let mut acc = [vdupq_n_s32(0); 16];
    let (mut wp, mut ap, mut sp) = (w, a, s);
    for _ in 0..CHUNKS {
        let mut lo0 = [vdupq_n_s8(0); 4];
        let mut lo1 = [vdupq_n_s8(0); 4];
        for c in 0..4 {
            let q = vld1q_u8_x2(wp.add(c * 32));
            lo0[c] = vreinterpretq_s8_u8(vandq_u8(q.0, m4b));
            lo1[c] = vreinterpretq_s8_u8(vandq_u8(q.1, m4b));
        }
        let sc = [*sp, *sp.add(1), *sp.add(2), *sp.add(3)];
        for r in 0..4 {
            let q8 = vld1q_s8_x2(ap.add(r * 32));
            for c in 0..4 {
                let p = sdot(sdot(vdupq_n_s32(0), lo0[c], q8.0), lo1[c], q8.1);
                acc[c * 4 + r] = vmlaq_n_s32(acc[c * 4 + r], p, sc[c]);
            }
        }
        wp = wp.add(4 * 32);
        ap = ap.add(4 * 32);
        sp = sp.add(4);
    }
    let mut out = [0i32; 16];
    for i in 0..16 {
        out[i] = vaddvq_s32(acc[i]);
    }
    out
}

// Hand-asm: 16 acc pinned v0..v15; lo0[0..3]=v16..v19, lo1[0..3]=v20..v23; q8=v24,v25;
// mask=v26; scales=v27; temps=v28..v31. Pointer auto-increment, mask set once.
#[target_feature(enable = "neon,dotprod")]
unsafe fn q4lo_asm(w: *const u8, a: *const i8, s: *const i32) -> [i32; 16] {
    let mut acc = [0i32; 16];
    let av: *mut int32x4_t = std::ptr::null_mut();
    let (mut wp, mut ap, mut sp, mut n) = (w, a, s, CHUNKS);
    // accumulators live in v0..v15 for the whole loop; init to 0 then SDOT-accumulate.
    let mut a0 = vdupq_n_s32(0); let mut a1 = vdupq_n_s32(0);
    let mut a2 = vdupq_n_s32(0); let mut a3 = vdupq_n_s32(0);
    let mut a4 = vdupq_n_s32(0); let mut a5 = vdupq_n_s32(0);
    let mut a6 = vdupq_n_s32(0); let mut a7 = vdupq_n_s32(0);
    let mut a8 = vdupq_n_s32(0); let mut a9 = vdupq_n_s32(0);
    let mut a10 = vdupq_n_s32(0); let mut a11 = vdupq_n_s32(0);
    let mut a12 = vdupq_n_s32(0); let mut a13 = vdupq_n_s32(0);
    let mut a14 = vdupq_n_s32(0); let mut a15 = vdupq_n_s32(0);
    std::arch::asm!(
        "movi v26.16b, #0x0f",
        "3:",
        "ld1 {{v27.4s}}, [{s}], #16",                 // 4 channel scales
        // unpack 4 channels' low nibbles -> lo0=v16..19, lo1=v20..23
        "ld1 {{v28.16b, v29.16b}}, [{w}], #32",
        "and v16.16b, v28.16b, v26.16b",
        "and v20.16b, v29.16b, v26.16b",
        "ld1 {{v28.16b, v29.16b}}, [{w}], #32",
        "and v17.16b, v28.16b, v26.16b",
        "and v21.16b, v29.16b, v26.16b",
        "ld1 {{v28.16b, v29.16b}}, [{w}], #32",
        "and v18.16b, v28.16b, v26.16b",
        "and v22.16b, v29.16b, v26.16b",
        "ld1 {{v28.16b, v29.16b}}, [{w}], #32",
        "and v19.16b, v28.16b, v26.16b",
        "and v23.16b, v29.16b, v26.16b",
        // Each row: 4 INDEPENDENT channel chains interleaved (4 temps v28-v31) so the sdots
        // issue back-to-back without waiting - exposes ILP for N1's narrower core.
        // row 0 -> acc v0(c0) v4(c1) v8(c2) v12(c3)
        "ld1 {{v24.16b, v25.16b}}, [{a}], #32",
        "movi v28.4s, #0", "movi v29.4s, #0", "movi v30.4s, #0", "movi v31.4s, #0",
        "sdot v28.4s, v16.16b, v24.16b", "sdot v29.4s, v17.16b, v24.16b", "sdot v30.4s, v18.16b, v24.16b", "sdot v31.4s, v19.16b, v24.16b",
        "sdot v28.4s, v20.16b, v25.16b", "sdot v29.4s, v21.16b, v25.16b", "sdot v30.4s, v22.16b, v25.16b", "sdot v31.4s, v23.16b, v25.16b",
        "mla v0.4s, v28.4s, v27.s[0]", "mla v4.4s, v29.4s, v27.s[1]", "mla v8.4s, v30.4s, v27.s[2]", "mla v12.4s, v31.4s, v27.s[3]",
        // row 1 -> v1 v5 v9 v13
        "ld1 {{v24.16b, v25.16b}}, [{a}], #32",
        "movi v28.4s, #0", "movi v29.4s, #0", "movi v30.4s, #0", "movi v31.4s, #0",
        "sdot v28.4s, v16.16b, v24.16b", "sdot v29.4s, v17.16b, v24.16b", "sdot v30.4s, v18.16b, v24.16b", "sdot v31.4s, v19.16b, v24.16b",
        "sdot v28.4s, v20.16b, v25.16b", "sdot v29.4s, v21.16b, v25.16b", "sdot v30.4s, v22.16b, v25.16b", "sdot v31.4s, v23.16b, v25.16b",
        "mla v1.4s, v28.4s, v27.s[0]", "mla v5.4s, v29.4s, v27.s[1]", "mla v9.4s, v30.4s, v27.s[2]", "mla v13.4s, v31.4s, v27.s[3]",
        // row 2 -> v2 v6 v10 v14
        "ld1 {{v24.16b, v25.16b}}, [{a}], #32",
        "movi v28.4s, #0", "movi v29.4s, #0", "movi v30.4s, #0", "movi v31.4s, #0",
        "sdot v28.4s, v16.16b, v24.16b", "sdot v29.4s, v17.16b, v24.16b", "sdot v30.4s, v18.16b, v24.16b", "sdot v31.4s, v19.16b, v24.16b",
        "sdot v28.4s, v20.16b, v25.16b", "sdot v29.4s, v21.16b, v25.16b", "sdot v30.4s, v22.16b, v25.16b", "sdot v31.4s, v23.16b, v25.16b",
        "mla v2.4s, v28.4s, v27.s[0]", "mla v6.4s, v29.4s, v27.s[1]", "mla v10.4s, v30.4s, v27.s[2]", "mla v14.4s, v31.4s, v27.s[3]",
        // row 3 -> v3 v7 v11 v15
        "ld1 {{v24.16b, v25.16b}}, [{a}], #32",
        "movi v28.4s, #0", "movi v29.4s, #0", "movi v30.4s, #0", "movi v31.4s, #0",
        "sdot v28.4s, v16.16b, v24.16b", "sdot v29.4s, v17.16b, v24.16b", "sdot v30.4s, v18.16b, v24.16b", "sdot v31.4s, v19.16b, v24.16b",
        "sdot v28.4s, v20.16b, v25.16b", "sdot v29.4s, v21.16b, v25.16b", "sdot v30.4s, v22.16b, v25.16b", "sdot v31.4s, v23.16b, v25.16b",
        "mla v3.4s, v28.4s, v27.s[0]", "mla v7.4s, v29.4s, v27.s[1]", "mla v11.4s, v30.4s, v27.s[2]", "mla v15.4s, v31.4s, v27.s[3]",
        "subs {n}, {n}, #1",
        "b.ne 3b",
        w = inout(reg) wp,
        a = inout(reg) ap,
        s = inout(reg) sp,
        n = inout(reg) n,
        inout("v0") a0, inout("v1") a1, inout("v2") a2, inout("v3") a3,
        inout("v4") a4, inout("v5") a5, inout("v6") a6, inout("v7") a7,
        inout("v8") a8, inout("v9") a9, inout("v10") a10, inout("v11") a11,
        inout("v12") a12, inout("v13") a13, inout("v14") a14, inout("v15") a15,
        out("v16") _, out("v17") _, out("v18") _, out("v19") _,
        out("v20") _, out("v21") _, out("v22") _, out("v23") _,
        out("v24") _, out("v25") _, out("v26") _, out("v27") _,
        out("v28") _, out("v29") _, out("v30") _, out("v31") _,
        options(nostack),
    );
    let av = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15];
    let mut out = [0i32; 16];
    for i in 0..16 {
        out[i] = vaddvq_s32(av[i]);
    }
    out
}

fn main() {
    let w = fill_u8(WBYTES, 1);
    let a = fill_i8(ABYTES, 7);
    let s = fill_i32(NSCALE, 3);
    let reps = 300_000usize;
    // 16 outputs * CHUNKS * 32 MACs each (two sdots of 16)
    let gmac = 2.0 * 16.0 * CHUNKS as f64 * 32.0 * reps as f64 / 1e9;

    let run = |f: unsafe fn(*const u8, *const i8, *const i32) -> [i32; 16]| -> ([i32; 16], f64) {
        let mut r = [0i32; 16];
        let mut t = f64::MAX;
        for _ in 0..5 {
            let now = Instant::now();
            for _ in 0..reps {
                r = unsafe { f(w.as_ptr(), a.as_ptr(), s.as_ptr()) };
                std::hint::black_box(&r);
            }
            t = t.min(now.elapsed().as_secs_f64());
        }
        (r, t)
    };

    let (r_ref, t_ref) = run(q4lo_intrinsic);
    let (r_asm, t_asm) = run(q4lo_asm);
    let ok = r_ref == r_asm;
    println!("correctness: {}", if ok { "OK" } else { "FAIL" });
    if !ok {
        println!("  ref={:?}\n  asm={:?}", r_ref, r_asm);
    }
    println!("intrinsic : {:.2} ms  {:.1} GMAC/s", t_ref * 1e3, gmac / t_ref);
    println!("asm       : {:.2} ms  {:.1} GMAC/s  ({:.2}x)", t_asm * 1e3, gmac / t_asm, t_ref / t_asm);
}
