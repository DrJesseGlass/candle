//! Candle-native interleaved repacking of Q4_K weights for fast aarch64 GEMM/GEMV.
//!
//! Recent llama.cpp beats a per-row Q4_K dot kernel on aarch64 by repacking the
//! weight matrix into an 8-row-interleaved layout (`block_q4_Kx8`) with the 6-bit
//! scales/mins pre-unpacked, then running dedicated SDOT (and i8mm) GEMM kernels
//! over it. This module is candle's own take on that idea — designed to fit
//! `GgmlType` cleanly rather than mirror ggml's byte layout — with llama.cpp's
//! `make_block_q4_Kx8` kept as a fallback reference if this layout underperforms.
//!
//! Step 1 (this file): the packed type + repack-on-load + integer-exact extraction
//! helpers shared by the repack, its tests, and the kernels to come. The GEMM/GEMV
//! kernels that consume `BlockQ4Kx8` land in later steps.

use super::k_quants::{BlockQ4K, QK_K};
use byteorder::{ByteOrder, LittleEndian};
use half::f16;

/// Output channels interleaved per packed block.
pub(crate) const Q4KX8_ROWS: usize = 8;
/// Sub-blocks of 32 weights per Q4_K super-block (256 / 32).
pub(crate) const SUBBLOCKS: usize = QK_K / 32; // 8
/// 64-weight chunks per super-block; each holds 2 sub-blocks as lo/hi nibbles.
pub(crate) const CHUNKS: usize = QK_K / 64; // 4

/// Eight output channels' worth of one Q4_K super-block, interleaved for SDOT.
///
/// Layout choices (candle-native):
/// - `d`/`dmin`: per-row super-block scales, kept as f16 (no precision change).
/// - `scales`/`mins`: the 6-bit values **pre-unpacked to bytes**, indexed
///   `[row * SUBBLOCKS + sub]`. This removes the per-call 6-bit bit-twiddle from
///   the inner loop — one of the two structural wins of repacking.
/// - `qs`: 4-bit quants kept packed, reorganized to `[chunk][row][32 bytes]`
///   (`j * (Q4KX8_ROWS*32) + r*32 + b`) so the kernel streams all 8 rows' data
///   for a chunk from one contiguous 256-byte run — the second win.
#[repr(C)]
#[derive(Clone)]
pub struct BlockQ4Kx8 {
    pub(crate) d: [f16; Q4KX8_ROWS],
    pub(crate) dmin: [f16; Q4KX8_ROWS],
    pub(crate) scales: [u8; Q4KX8_ROWS * SUBBLOCKS], // 64
    pub(crate) mins: [u8; Q4KX8_ROWS * SUBBLOCKS],   // 64
    pub(crate) qs: [u8; Q4KX8_ROWS * (QK_K / 2)],    // 1024: [chunk][row][32]
}

impl BlockQ4Kx8 {
    fn zeroed() -> Self {
        Self {
            d: [f16::ZERO; Q4KX8_ROWS],
            dmin: [f16::ZERO; Q4KX8_ROWS],
            scales: [0; Q4KX8_ROWS * SUBBLOCKS],
            mins: [0; Q4KX8_ROWS * SUBBLOCKS],
            qs: [0; Q4KX8_ROWS * (QK_K / 2)],
        }
    }
}

/// Unpack a Q4_K super-block's 12-byte `scales` field into the 8 sub-block scales
/// and 8 sub-block mins (each a 6-bit value in a byte). Bit-identical to the
/// extraction in `BlockQ4K::vec_dot_unopt`, so any kernel using the unpacked bytes
/// matches the scalar path exactly.
#[inline]
pub(crate) fn q4k_unpack_scales_mins(packed: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let mut utmp = [0u32; 4];
    LittleEndian::read_u32_into(packed, &mut utmp[0..3]);
    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;

    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    LittleEndian::write_u32_into(&utmp[0..2], &mut scales);
    LittleEndian::write_u32_into(&utmp[2..4], &mut mins);
    (scales, mins)
}

/// Repack `Q4KX8_ROWS` weight rows (each `nb` Q4_K super-blocks long) into `nb`
/// interleaved `BlockQ4Kx8`. Rows are output channels of the weight matrix; all
/// eight must have the same super-block count. Done once at load — not perf
/// critical, so it uses the scalar scale/min unpack.
pub(crate) fn repack_q4k_x8(rows: &[&[BlockQ4K]; Q4KX8_ROWS]) -> Vec<BlockQ4Kx8> {
    let nb = rows[0].len();
    debug_assert!(rows.iter().all(|r| r.len() == nb), "ragged rows in repack_q4k_x8");
    let mut out = Vec::with_capacity(nb);
    for i in 0..nb {
        let mut blk = BlockQ4Kx8::zeroed();
        for r in 0..Q4KX8_ROWS {
            let src = &rows[r][i];
            blk.d[r] = src.d;
            blk.dmin[r] = src.dmin;
            let (sc, mn) = q4k_unpack_scales_mins(&src.scales);
            for s in 0..SUBBLOCKS {
                blk.scales[r * SUBBLOCKS + s] = sc[s];
                blk.mins[r * SUBBLOCKS + s] = mn[s];
            }
            // qs: chunk-major, row-interleaved. src.qs is 4 chunks of 32 bytes.
            for j in 0..CHUNKS {
                let dst0 = j * (Q4KX8_ROWS * 32) + r * 32;
                blk.qs[dst0..dst0 + 32].copy_from_slice(&src.qs[j * 32..j * 32 + 32]);
            }
        }
        out.push(blk);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantized::k_quants::GgmlType;

    fn lcg(s: &mut u64) -> f32 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        2.0 * ((*s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    // The repack must preserve the raw integer payload exactly: per (row,
    // super-block, sub-block) the unpacked scale & min bytes, and per weight the
    // 4-bit nibble, must match what the original BlockQ4K yields. Integer-exact, so
    // it pins the layout independently of any float dequant formula.
    #[test]
    fn repack_q4k_x8_preserves_payload() {
        let nb = 3usize;
        let k = nb * QK_K;
        let mut st = 0xa5a5_1234_dead_0001u64;

        // Build 8 rows of quantized Q4_K.
        let mut rows_q: Vec<Vec<BlockQ4K>> = Vec::new();
        for _ in 0..Q4KX8_ROWS {
            let f: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
            let mut q = vec![BlockQ4K::zeros(); nb];
            BlockQ4K::from_float(&f, &mut q);
            rows_q.push(q);
        }
        let refs: [&[BlockQ4K]; Q4KX8_ROWS] = std::array::from_fn(|r| rows_q[r].as_slice());
        let packed = repack_q4k_x8(&refs);
        assert_eq!(packed.len(), nb);

        // Reference nibble extraction (matches vec_dot_unopt): chunk j -> lo then hi.
        let ref_nibbles = |blk: &BlockQ4K| -> [u8; QK_K] {
            let mut a = [0u8; QK_K];
            let mut p = 0;
            for j in 0..CHUNKS {
                for l in 0..32 {
                    a[p] = blk.qs[j * 32 + l] & 0xF;
                    p += 1;
                }
                for l in 0..32 {
                    a[p] = blk.qs[j * 32 + l] >> 4;
                    p += 1;
                }
            }
            a
        };
        // Same, reconstructed from the packed (chunk-major, row-interleaved) qs.
        let packed_nibbles = |blk: &BlockQ4Kx8, r: usize| -> [u8; QK_K] {
            let mut a = [0u8; QK_K];
            let mut p = 0;
            for j in 0..CHUNKS {
                let base = j * (Q4KX8_ROWS * 32) + r * 32;
                for l in 0..32 {
                    a[p] = blk.qs[base + l] & 0xF;
                    p += 1;
                }
                for l in 0..32 {
                    a[p] = blk.qs[base + l] >> 4;
                    p += 1;
                }
            }
            a
        };

        for i in 0..nb {
            for r in 0..Q4KX8_ROWS {
                let src = &rows_q[r][i];
                let (sc, mn) = q4k_unpack_scales_mins(&src.scales);
                for s in 0..SUBBLOCKS {
                    assert_eq!(packed[i].scales[r * SUBBLOCKS + s], sc[s], "scale r{r} i{i} s{s}");
                    assert_eq!(packed[i].mins[r * SUBBLOCKS + s], mn[s], "min r{r} i{i} s{s}");
                }
                assert_eq!(packed[i].d[r].to_bits(), src.d.to_bits(), "d r{r} i{i}");
                assert_eq!(packed[i].dmin[r].to_bits(), src.dmin.to_bits(), "dmin r{r} i{i}");
                assert_eq!(packed_nibbles(&packed[i], r), ref_nibbles(src), "nibbles r{r} i{i}");
            }
        }
    }

    // The packed SDOT GEMM must be bit-exact vs per-pair vec_dot for every
    // (channel, row), across MR tile heights.
    #[cfg(target_feature = "neon")]
    #[test]
    fn gemm_q4kx8_bit_exact() {
        use crate::quantized::neon::gemm_q4kx8_q8k;
        use crate::quantized::k_quants::BlockQ8K;

        let nb = 3usize;
        let k = nb * QK_K;
        let mut st = 0x1357_9bdf_2468_ace0u64;

        // 8 weight channels -> packed.
        let mut wq: Vec<Vec<BlockQ4K>> = Vec::new();
        for _ in 0..Q4KX8_ROWS {
            let f: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
            let mut q = vec![BlockQ4K::zeros(); nb];
            BlockQ4K::from_float(&f, &mut q);
            wq.push(q);
        }
        let wrefs: [&[BlockQ4K]; Q4KX8_ROWS] = std::array::from_fn(|r| wq[r].as_slice());
        let packed = repack_q4k_x8(&wrefs);

        // Up to 4 activation rows.
        let mut aq: Vec<Vec<BlockQ8K>> = Vec::new();
        for _ in 0..4 {
            let f: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
            let mut q = vec![BlockQ8K::zeros(); nb];
            BlockQ8K::from_float(&f, &mut q);
            aq.push(q);
        }

        macro_rules! check {
            ($mr:literal) => {{
                let ys: [&[BlockQ8K]; $mr] = std::array::from_fn(|a| aq[a].as_slice());
                let mut got = vec![0f32; Q4KX8_ROWS * $mr];
                gemm_q4kx8_q8k::<$mr>(&packed, &ys, &mut got);
                for c in 0..Q4KX8_ROWS {
                    for a in 0..$mr {
                        let want = BlockQ4K::vec_dot(k, &wq[c], &aq[a]);
                        assert_eq!(
                            got[c * $mr + a].to_bits(),
                            want.to_bits(),
                            "MR={} channel {c} row {a}: got {} want {}",
                            $mr, got[c * $mr + a], want
                        );
                    }
                }
            }};
        }
        check!(1);
        check!(2);
        check!(4);
    }

    // Standalone microbench: baseline per-column matmul vs the packed 8-channel
    // SDOT GEMM, for decode (m=1) and prefill (m=512). Ignored by default; run with
    //   RUSTFLAGS="-C target-cpu=native" cargo test -p candle-core --release \
    //     q4kx8_microbench -- --ignored --nocapture
    // Pin cores with taskset + CANDLE_QMATMUL_PREFILL_THREADS to emulate a tier.
    #[cfg(target_feature = "neon")]
    #[test]
    #[ignore]
    fn q4kx8_microbench() {
        use crate::quantized::k_quants::{matmul, BlockQ8K};
        use crate::quantized::neon::gemm_q4kx8_q8k;
        use rayon::prelude::*;
        use std::time::Instant;

        let k = 2048usize;
        let n = 2048usize;
        let nb = k / QK_K;
        let reps = 6usize;
        let mut st = 0xdead_beef_0123_4567u64;
        let mut rnd = |s: &mut u64| {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            2.0 * ((*s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };

        // Weights: n output channels × k, quantized. Baseline rhs_t is row-major
        // (channel r at r*nb). Packed is channel-group-major (group g at g*nb).
        let mut rhs_t = vec![BlockQ4K::zeros(); n * nb];
        for r in 0..n {
            let f: Vec<f32> = (0..k).map(|_| rnd(&mut st)).collect();
            BlockQ4K::from_float(&f, &mut rhs_t[r * nb..(r + 1) * nb]);
        }
        let mut packed: Vec<BlockQ4Kx8> = Vec::with_capacity((n / 8) * nb);
        for g in 0..n / 8 {
            let grp: [&[BlockQ4K]; Q4KX8_ROWS] =
                std::array::from_fn(|r| &rhs_t[(g * 8 + r) * nb..(g * 8 + r + 1) * nb]);
            packed.extend(repack_q4k_x8(&grp));
        }

        struct P(*mut f32);
        unsafe impl Sync for P {}

        for &m in &[1usize, 512usize] {
            let lhs_f: Vec<f32> = (0..m * k).map(|_| rnd(&mut st)).collect();
            let mut dst_base = vec![0f32; m * n];
            let mut dst_pack = vec![0f32; m * n];

            // Baseline (quantizes activations internally).
            let mut t_base = f64::MAX;
            for _ in 0..reps {
                let t = Instant::now();
                matmul::<BlockQ4K>((m, k, n), &lhs_f, &rhs_t, &mut dst_base).unwrap();
                t_base = t_base.min(t.elapsed().as_secs_f64());
            }

            // Packed: quantize activations to Q8K, then parallel 8-channel GEMM,
            // tiling m by MAIN (with a 1-row remainder). Macro lets us sweep the
            // main tile height to find where register pressure stops helping.
            macro_rules! bench_packed {
                ($main:literal) => {{
                    let mut t_pack = f64::MAX;
                    for _ in 0..reps {
                        let t = Instant::now();
                        let mut lhs_q = vec![BlockQ8K::zeros(); m * nb];
                        for a in 0..m {
                            BlockQ8K::from_float(
                                &lhs_f[a * k..(a + 1) * k],
                                &mut lhs_q[a * nb..(a + 1) * nb],
                            );
                        }
                        let dptr = P(dst_pack.as_mut_ptr());
                        (0..n / 8).into_par_iter().for_each(|g| {
                            let w = &packed[g * nb..(g + 1) * nb];
                            let p = &dptr;
                            let mut r = 0;
                            while r + $main <= m {
                                let rows: [&[BlockQ8K]; $main] =
                                    std::array::from_fn(|a| &lhs_q[(r + a) * nb..(r + a + 1) * nb]);
                                let mut tile = [0f32; 8 * $main];
                                gemm_q4kx8_q8k::<$main>(w, &rows, &mut tile);
                                for c in 0..8 {
                                    for a in 0..$main {
                                        unsafe {
                                            *p.0.add((r + a) * n + g * 8 + c) = tile[c * $main + a]
                                        };
                                    }
                                }
                                r += $main;
                            }
                            while r < m {
                                let rows: [&[BlockQ8K]; 1] = [&lhs_q[r * nb..(r + 1) * nb]];
                                let mut tile = [0f32; 8];
                                gemm_q4kx8_q8k::<1>(w, &rows, &mut tile);
                                for c in 0..8 {
                                    unsafe { *p.0.add(r * n + g * 8 + c) = tile[c] };
                                }
                                r += 1;
                            }
                        });
                        t_pack = t_pack.min(t.elapsed().as_secs_f64());
                    }
                    let mut maxdiff = 0f32;
                    for idx in (0..m * n).step_by(((m * n) / 997).max(1)) {
                        maxdiff = maxdiff.max((dst_base[idx] - dst_pack[idx]).abs());
                    }
                    let phase = if m == 1 { "decode " } else { "prefill" };
                    println!(
                        "[{phase}] m={m:<4} MR={:<1} base={:.3}ms packed={:.3}ms speedup={:.2}x maxdiff={maxdiff:.1e}",
                        $main, t_base * 1e3, t_pack * 1e3, t_base / t_pack
                    );
                }};
            }
            if m == 1 {
                bench_packed!(1);
            } else {
                bench_packed!(2);
                bench_packed!(4);
            }
        }
    }
}
