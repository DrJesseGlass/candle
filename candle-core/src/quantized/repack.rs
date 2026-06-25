//! Candle-native interleaved repacking of Q4_K weights for fast aarch64 GEMM/GEMV.
//!
//! Recent llama.cpp beats a per-row Q4_K dot kernel on aarch64 by repacking the
//! weight matrix into an 8-row-interleaved layout (`block_q4_Kx8`) with the 6-bit
//! scales/mins pre-unpacked, then running dedicated SDOT (and i8mm) GEMM kernels
//! over it. This module is candle's own take on that idea - designed to fit
//! `GgmlType` cleanly rather than mirror ggml's byte layout - with llama.cpp's
//! `make_block_q4_Kx8` kept as a fallback reference if this layout underperforms.
//!
//! Step 1 (this file): the packed type + repack-on-load + integer-exact extraction
//! helpers shared by the repack, its tests, and the kernels to come. The GEMM/GEMV
//! kernels that consume `BlockQ4Kx8` land in later steps.

// Several packed helpers (runtime repack-cache dispatch, laneq layout, gguf-requant
// --pack) are consumed by the B-model integration step, not within candle-core yet.
#![allow(dead_code)]
// The repack loops index several parallel per-row arrays by the same counter;
// a range loop is clearer here than zipped iterators.
#![allow(clippy::needless_range_loop)]

use super::k_quants::{BlockQ4K, BlockQ6K, QK_K};
// The packed GEMM kernels (and the trait helpers / rayon they pull in) are
// aarch64/neon-only; the imports they need would be unused on other targets.
#[cfg(target_feature = "neon")]
use super::k_quants::{BlockQ8K, GgmlType};
use byteorder::{ByteOrder, LittleEndian};
use half::f16;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

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
///   the inner loop - one of the two structural wins of repacking.
/// - `qs`: 4-bit quants kept packed, reorganized to `[chunk][row][32 bytes]`
///   (`j * (Q4KX8_ROWS*32) + r*32 + b`) so the kernel streams all 8 rows' data
///   for a chunk from one contiguous 256-byte run - the second win.
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
/// eight must have the same super-block count. Done once at load - not perf
/// critical, so it uses the scalar scale/min unpack.
pub(crate) fn repack_q4k_x8(rows: &[&[BlockQ4K]; Q4KX8_ROWS]) -> Vec<BlockQ4Kx8> {
    let nb = rows[0].len();
    debug_assert!(
        rows.iter().all(|r| r.len() == nb),
        "ragged rows in repack_q4k_x8"
    );
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

// ===========================================================================
// Laneq layout (N1 parity path): 8 columns interleaved 8 bytes at a time, with
// the 6-bit scales/mins repacked into a 96-byte block aligned for a vector load.
// This is the byte layout llama.cpp's arm `ggml_gemm_q4_K_8x8_q8_K` consumes via
// `vdotq_laneq_s32` - chosen because that lane-broadcast kernel reaches llama's
// N1 speed (our earlier separate-channel BlockQ4Kx8 can't feed laneq). Ported
// from llama's `make_block_q4_Kx8` (blck_size_interleave = 8); the matching laneq
// GEMM lands next. Distinct from BlockQ4Kx8 (kept for the shipped +25% path).
// ===========================================================================

/// 8 Q4_K columns interleaved for the `vdotq_laneq` GEMM. Field layout matches
/// llama's `block_q4_Kx8` byte-for-byte so the kernel can mirror its proven one.
#[repr(C)]
#[derive(Clone)]
pub struct BlockQ4Kx8L {
    pub(crate) d: [f16; 8],
    pub(crate) dmin: [f16; 8],
    pub(crate) scales: [u8; 96], // 6-bit scales+mins, repacked (8 cols)
    pub(crate) qs: [u8; 1024],   // 4-bit quants, 8-byte round-robin interleave
}

impl BlockQ4Kx8L {
    fn zeroed() -> Self {
        Self {
            d: [f16::ZERO; 8],
            dmin: [f16::ZERO; 8],
            scales: [0; 96],
            qs: [0; 1024],
        }
    }
}

/// Port of llama's `make_block_q4_Kx8` for one super-block of 8 columns. `bsi` is
/// the `blck_size_interleave` (8 for the i8mm/SMMLA kernel, 4 for the lane=row
/// `8x4` kernel) - only the qs interleave stride depends on it; scales/mins are
/// identical. Pure data rearrangement; deterministic.
fn make_q4kx8_laneq_bsi(cols: &[&[BlockQ4K]; 8], i: usize, bsi: usize) -> BlockQ4Kx8L {
    let mut out = BlockQ4Kx8L::zeroed();
    for c in 0..8 {
        out.d[c] = cols[c][i].d;
        out.dmin[c] = cols[c][i].dmin;
    }
    // qs: take `bsi` bytes at a time, round-robin across the 8 columns.
    // end = QK_K * 4 / bsi; src col = t%8, src off = (t/8)*bsi, dst = t*bsi.
    let end = QK_K * 4 / bsi;
    for t in 0..end {
        let src_id = t % 8;
        let src_off = (t / 8) * bsi;
        let dst_off = t * bsi;
        out.qs[dst_off..dst_off + bsi]
            .copy_from_slice(&cols[src_id][i].qs[src_off..src_off + bsi]);
    }
    // scales: repack the 6-bit packed scales/mins of all 8 columns into 96 bytes.
    let mut s = [0u8; 8];
    let mut m = [0u8; 8];
    for ii in 0..4 {
        for j in 0..8 {
            s[j] = cols[j][i].scales[ii] & 63;
            m[j] = cols[j][i].scales[ii + 4] & 63;
        }
        let b = ii * 12;
        out.scales[b] = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[b + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[b + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[b + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[b + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[b + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[b + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[b + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[b + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[b + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[b + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[b + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
    }
    for ii in 0..4 {
        for j in 0..8 {
            s[j] = ((cols[j][i].scales[ii] & 192) >> 2) | (cols[j][i].scales[ii + 8] & 15);
            m[j] =
                ((cols[j][i].scales[ii + 4] & 192) >> 2) | ((cols[j][i].scales[ii + 8] & 240) >> 4);
        }
        let b = ii * 12 + 48;
        out.scales[b] = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[b + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[b + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[b + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[b + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[b + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[b + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[b + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[b + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[b + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[b + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[b + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
    }
    out
}

/// Repack 8 weight columns into `nb` laneq blocks, interleave 8 (the i8mm/SMMLA
/// weight layout).
pub(crate) fn repack_q4kx8_laneq(cols: &[&[BlockQ4K]; 8]) -> Vec<BlockQ4Kx8L> {
    let nb = cols[0].len();
    (0..nb).map(|i| make_q4kx8_laneq_bsi(cols, i, 8)).collect()
}

/// Repack 8 weight columns into `nb` laneq blocks, interleave 4 (the weight layout
/// the lane=row `8x4` DOTPROD kernel consumes - 4 bytes/column per 16-byte read).
#[allow(dead_code)] // taken by the lane=row prefill kernel
pub(crate) fn repack_q4kx8_laneq4(cols: &[&[BlockQ4K]; 8]) -> Vec<BlockQ4Kx8L> {
    let nb = cols[0].len();
    (0..nb).map(|i| make_q4kx8_laneq_bsi(cols, i, 4)).collect()
}

// ===========================================================================
// Activation side of the i8mm (SMMLA) GEMM: 4 Q8_K rows interleaved into the
// `block_q8_Kx4` layout that llama's `ggml_gemm_q4_K_8x8_q8_K` consumes. The
// SMMLA kernel reads 2 rows at a time as a 2x8 int8 operand, so the 4 rows are
// laid out interleaved (rows 01 then 23) in 8-byte groups. Byte-for-byte port of
// llama's `ggml_quantize_mat_q8_k_4x8` so the matching kernel is a faithful port.
// ===========================================================================

/// Four Q8_K activation rows interleaved for the SMMLA GEMM. Mirrors llama's
/// `block_q8_Kx4` field layout (`d[4]`, interleaved `qs`, grouped `bsums`).
// dead_code until the i8mm GEMM is wired into the prefill dispatcher (step 2).
#[allow(dead_code)]
#[repr(C)]
#[derive(Clone)]
pub(crate) struct BlockQ8Kx4 {
    pub(crate) d: [f32; 4],
    pub(crate) qs: [i8; QK_K * 4],
    pub(crate) bsums: [i16; QK_K / 4],
}

impl BlockQ8Kx4 {
    fn zeroed() -> Self {
        Self {
            d: [0.0; 4],
            qs: [0; QK_K * 4],
            bsums: [0; QK_K / 4],
        }
    }
}

/// Quantize and interleave 4 activation rows (each `nb * QK_K` f32) into the `nb`
/// `BlockQ8Kx4` of `out` (`out.len() == nb`). Pure scalar (runs identically on any
/// host), so the local build produces the exact bytes the i8mm hardware kernel will
/// consume. Each super-block `i` is independent, so this is safe to call on disjoint
/// `out` slices from different threads (the parallel-quant path). Port of llama's
/// `ggml_quantize_mat_q8_k_4x8_generic` (`blck_size_interleave = 8`).
#[allow(dead_code)]
pub(crate) fn quantize_mat_q8_k_4x8_into(rows: &[&[f32]; 4], out: &mut [BlockQ8Kx4]) {
    const BSI: usize = 8; // blck_size_interleave
    for (i, blk) in out.iter_mut().enumerate() {
        *blk = BlockQ8Kx4::zeroed();
        let mut srcv = [[0f32; QK_K]; 4];
        let mut iscale = [0f32; 4];
        for (row, &r) in rows.iter().enumerate() {
            let mut amax = 0f32;
            let mut max = 0f32;
            for j in 0..QK_K {
                let v = r[i * QK_K + j];
                srcv[row][j] = v;
                if amax < v.abs() {
                    amax = v.abs();
                    max = v;
                }
            }
            iscale[row] = if amax != 0.0 { -127.0 / max } else { 0.0 };
            blk.d[row] = if amax != 0.0 { 1.0 / iscale[row] } else { 0.0 };
        }
        // Quants are interleaved in 8-byte runs across the 4 rows; bsums are
        // grouped 4-at-a-time per source super-block (the kernel's bias term).
        for j in 0..QK_K * 4 {
            let mut src_offset = (j / (4 * BSI)) * BSI;
            let src_id = (j % (4 * BSI)) / BSI;
            src_offset += j % BSI;
            let index = (((j & 31) >> 3) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);
            let x0 = srcv[src_id][src_offset] * iscale[src_id];
            let q = x0.round() as i8;
            blk.qs[j] = q;
            blk.bsums[index] += q as i16;
        }
    }
}

/// Vec-returning convenience wrapper over `quantize_mat_q8_k_4x8_into`.
#[allow(dead_code)]
pub(crate) fn quantize_mat_q8_k_4x8(rows: &[&[f32]; 4], nb: usize) -> Vec<BlockQ8Kx4> {
    let mut out = vec![BlockQ8Kx4::zeroed(); nb];
    quantize_mat_q8_k_4x8_into(rows, &mut out);
    out
}

/// Quantize and interleave 4 activation rows into the `nb` `BlockQ8Kx4` of `out`
/// (`out.len() == nb`) with `blck_size_interleave = 4` - the layout llama's DOTPROD
/// `ggml_gemm_q4_K_8x4` consumes, where the SDOT lane index selects the ROW (4
/// bytes/row per 16-byte group). Distinct from `quantize_mat_q8_k_4x8_into`
/// (interleave 8, for the i8mm/SMMLA kernel) only in the interleave stride and the
/// bsums index. Each super-block is independent (safe on disjoint `out` slices from
/// different threads). Port of llama's `ggml_quantize_mat_q8_K_4x4_generic`.
#[allow(dead_code)]
pub(crate) fn quantize_mat_q8_k_4x4_into(rows: &[&[f32]; 4], out: &mut [BlockQ8Kx4]) {
    const BSI: usize = 4; // blck_size_interleave
    for (i, blk) in out.iter_mut().enumerate() {
        *blk = BlockQ8Kx4::zeroed();
        let mut srcv = [[0f32; QK_K]; 4];
        let mut iscale = [0f32; 4];
        for (row, &r) in rows.iter().enumerate() {
            let mut amax = 0f32;
            let mut max = 0f32;
            for j in 0..QK_K {
                let v = r[i * QK_K + j];
                srcv[row][j] = v;
                if amax < v.abs() {
                    amax = v.abs();
                    max = v;
                }
            }
            iscale[row] = if amax != 0.0 { -127.0 / max } else { 0.0 };
            blk.d[row] = if amax != 0.0 { 1.0 / iscale[row] } else { 0.0 };
        }
        // Quants interleaved in 4-byte runs across the 4 rows; bsums grouped
        // 4-at-a-time per source super-block (the kernel's bias term).
        for j in 0..QK_K * 4 {
            let mut src_offset = (j / (4 * BSI)) * BSI;
            let src_id = (j % (4 * BSI)) / BSI;
            src_offset += j % BSI;
            let index = (((j & 15) >> 2) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);
            let x0 = srcv[src_id][src_offset] * iscale[src_id];
            let q = x0.round() as i8;
            blk.qs[j] = q;
            blk.bsums[index] += q as i16;
        }
    }
}

/// Vec-returning convenience wrapper over `quantize_mat_q8_k_4x4_into`.
#[allow(dead_code)]
pub(crate) fn quantize_mat_q8_k_4x4(rows: &[&[f32]; 4], nb: usize) -> Vec<BlockQ8Kx4> {
    let mut out = vec![BlockQ8Kx4::zeroed(); nb];
    quantize_mat_q8_k_4x4_into(rows, &mut out);
    out
}

/// Process-global cache of repacked Q4_K weights, keyed by the source block
/// slice's base pointer. Model weights live for the whole run, so this repacks
/// each weight matrix once on first use. Benchmark-grade: keyed on pointer, so it
/// assumes weights aren't freed and reallocated at the same address (true for a
/// loaded model). Cleared via `clear_packed_cache` if needed.
static PACKED_CACHE: LazyLock<Mutex<HashMap<usize, Arc<Vec<BlockQ4Kx8>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Drop all cached repacked weights (e.g. between unrelated models in a process).
pub fn clear_packed_cache() {
    PACKED_CACHE.lock().unwrap().clear();
    LANEQ_CACHE.lock().unwrap().clear();
    LANEQ4_CACHE.lock().unwrap().clear();
}

/// Eight output channels' worth of one Q6_K super-block, interleaved for SDOT.
/// Mirrors `BlockQ4Kx8` but for Q6_K: `scales` are i8 (already byte-sized - no
/// 6-bit unpack needed, unlike Q4_K), and the quants split into `ql` (low 4 bits)
/// + `qh` (high 2 bits), each reorganized to `[chunk][row][bytes]` so the kernel
/// streams all 8 rows of a 128-value chunk from one contiguous run.
#[repr(C)]
#[derive(Clone)]
pub struct BlockQ6Kx8 {
    pub(crate) d: [f16; Q4KX8_ROWS],
    pub(crate) scales: [i8; Q4KX8_ROWS * (QK_K / 16)], // 128
    pub(crate) ql: [u8; Q4KX8_ROWS * (QK_K / 2)],      // 1024: [chunk(2)][row(8)][64]
    pub(crate) qh: [u8; Q4KX8_ROWS * (QK_K / 4)],      // 512:  [chunk(2)][row(8)][32]
}

impl BlockQ6Kx8 {
    fn zeroed() -> Self {
        Self {
            d: [f16::ZERO; Q4KX8_ROWS],
            scales: [0; Q4KX8_ROWS * (QK_K / 16)],
            ql: [0; Q4KX8_ROWS * (QK_K / 2)],
            qh: [0; Q4KX8_ROWS * (QK_K / 4)],
        }
    }
}

/// Repack `Q4KX8_ROWS` Q6_K weight rows (each `nb` super-blocks) into `nb`
/// interleaved `BlockQ6Kx8`. Scalar (load-time, not perf-critical); the i8 scales
/// copy directly and the 128B `ql`/64B `qh` split into two 64B/32B chunks.
pub(crate) fn repack_q6k_x8(rows: &[&[BlockQ6K]; Q4KX8_ROWS]) -> Vec<BlockQ6Kx8> {
    let nb = rows[0].len();
    debug_assert!(
        rows.iter().all(|r| r.len() == nb),
        "ragged rows in repack_q6k_x8"
    );
    const QLC: usize = QK_K / 4; // 64 ql bytes / 128-value chunk
    const QHC: usize = QK_K / 8; // 32 qh bytes / chunk
    let mut out = Vec::with_capacity(nb);
    for i in 0..nb {
        let mut blk = BlockQ6Kx8::zeroed();
        for r in 0..Q4KX8_ROWS {
            let src = &rows[r][i];
            blk.d[r] = src.d;
            for s in 0..(QK_K / 16) {
                blk.scales[r * (QK_K / 16) + s] = src.scales[s];
            }
            for j in 0..2 {
                let qld = j * (Q4KX8_ROWS * QLC) + r * QLC;
                blk.ql[qld..qld + QLC].copy_from_slice(&src.ql[j * QLC..j * QLC + QLC]);
                let qhd = j * (Q4KX8_ROWS * QHC) + r * QHC;
                blk.qh[qhd..qhd + QHC].copy_from_slice(&src.qh[j * QHC..j * QHC + QHC]);
            }
        }
        out.push(blk);
    }
    out
}

/// Repack a full row-major Q6_K weight matrix into the interleaved `BlockQ6Kx8`
/// layout for all `n/8` channel groups. Mirror of `repack_q4k_weight`.
pub fn repack_q6k_weight(rows: &[BlockQ6K], n: usize, nb: usize) -> Vec<BlockQ6Kx8> {
    debug_assert_eq!(n % Q4KX8_ROWS, 0, "repack_q6k_weight: n {n} not /8");
    debug_assert_eq!(rows.len(), n * nb, "repack_q6k_weight: rows len mismatch");
    let mut packed = Vec::with_capacity((n / 8) * nb);
    for g in 0..n / 8 {
        let grp: [&[BlockQ6K]; Q4KX8_ROWS] =
            std::array::from_fn(|r| &rows[(g * 8 + r) * nb..(g * 8 + r + 1) * nb]);
        packed.extend(repack_q6k_x8(&grp));
    }
    packed
}

/// Repack a full row-major Q4_K weight matrix (`n` output channels, each `nb`
/// super-blocks, laid out at row `r` from `r*nb`) into the interleaved
/// `BlockQ4Kx8` layout for all `n/8` channel groups. Single source of truth for
/// the row grouping, shared by `packed_cache_get` (runtime repack-on-load) and the
/// offline pre-packer; requires `n % Q4KX8_ROWS == 0`.
pub fn repack_q4k_weight(rows: &[BlockQ4K], n: usize, nb: usize) -> Vec<BlockQ4Kx8> {
    debug_assert_eq!(
        n % Q4KX8_ROWS,
        0,
        "repack_q4k_weight: n {n} not a multiple of 8"
    );
    debug_assert_eq!(rows.len(), n * nb, "repack_q4k_weight: rows len mismatch");
    let mut packed = Vec::with_capacity((n / 8) * nb);
    for g in 0..n / 8 {
        let grp: [&[BlockQ4K]; Q4KX8_ROWS] =
            std::array::from_fn(|r| &rows[(g * 8 + r) * nb..(g * 8 + r + 1) * nb]);
        packed.extend(repack_q4k_x8(&grp));
    }
    packed
}

fn packed_cache_get(rhs: &[BlockQ4K], n: usize, nb: usize) -> Arc<Vec<BlockQ4Kx8>> {
    let key = rhs.as_ptr() as usize;
    {
        if let Some(p) = PACKED_CACHE.lock().unwrap().get(&key) {
            return p.clone();
        }
    }
    let mut map = PACKED_CACHE.lock().unwrap();
    if let Some(p) = map.get(&key) {
        return p.clone();
    }
    let arc = Arc::new(repack_q4k_weight(rhs, n, nb));
    map.insert(key, arc.clone());
    arc
}

/// Whether a `(k, n)` Q4_K matmul can use the packed path: k a multiple of the
/// super-block (256) and n a multiple of the 8-channel pack width.
pub(crate) fn packed_q4k_applicable(k: usize, n: usize) -> bool {
    k.is_multiple_of(QK_K) && n.is_multiple_of(Q4KX8_ROWS)
}

/// `dst(m,n) = lhs(m,k) x rhs_q4k(n,k)^T` via the interleaved packed SDOT kernels.
/// Activations are quantized to Q8K here (as the baseline does). Weights are
/// repacked once and cached. Parallelized over 8-channel groups on `pool`. Decode
/// (m=1) uses the MR=1 kernel; prefill tiles rows by MR=2 (the N1 sweet spot -
/// MR=4 spills) with a 1-row remainder. Bit-identical to baseline `matmul`.
/// Prefill tile strategy for the packed GEMM, the m>=2 compute-bound path.
/// `nc4mr4` (DEFAULT): two 4-channel x 4-row calls (16 acc each) - measured BEST on
/// Neoverse-N1. `nc8mr2`: one 8-channel x 2-row call (16 acc). `nc8mr4`: one
/// 8-channel x 4-row call (32 acc) - wider reuse but REGRESSES ~20% on N1 (32 acc
/// spills the 32 NEON regs once weights/activations are added; N1's narrower core is
/// hurt more than M1's wide OoO). Kept selectable for other cores only. Override
/// with `CANDLE_PACKED_PREFILL={nc4mr4,nc8mr2,nc8mr4}`.
#[derive(Clone, Copy, PartialEq)]
enum PrefillTile {
    Nc8mr4,
    Nc4mr4,
    Nc8mr2,
}
static PACKED_PREFILL_TILE: LazyLock<PrefillTile> = LazyLock::new(|| {
    match std::env::var("CANDLE_PACKED_PREFILL")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "nc8mr4" => PrefillTile::Nc8mr4,
        "nc8mr2" => PrefillTile::Nc8mr2,
        _ => PrefillTile::Nc4mr4,
    }
});

#[cfg(target_feature = "neon")]
pub(crate) fn matmul_q4k_packed(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs_q4k: &[BlockQ4K],
    dst: &mut [f32],
) {
    let nb = k / QK_K;
    // Prefill (m>=4) on i8mm hardware: route to the SMMLA GEMM, which does ~4 SDOTs
    // of work per instruction (the large-M prefill lever). Only compiled when the
    // binary is built with `+i8mm` (else the real `smmla` is absent); decode (m<4)
    // and non-i8mm builds keep the SDOT path. Separate laneq weight cache (the SMMLA
    // layout differs from the SDOT `BlockQ4Kx8`), so an i8mm host holds both copies -
    // acceptable for now; a baked laneq format / laneq GEMV decode is a later step.
    #[cfg(target_feature = "i8mm")]
    {
        if m >= 4 && *PREFILL_I8MM {
            let packed = packed_laneq_cache_get(rhs_q4k, n, nb);
            matmul_q4kx8l_prepacked_i8mm((m, k, n), lhs, &packed, dst);
            return;
        }
    }
    // Prefill (m>=4) on a dotprod core WITHOUT i8mm (Graviton2/N1, the deploy
    // target): route to the lane=row 8x4 SDOT GEMM - llama's actual N1 prefill
    // kernel. It folds the Q4_K scale to f32 per sub-block so the wide 8-col x
    // 4-row tile fits in 32 NEON regs (candle's SDOT kernel can't - it spills),
    // measured 1.46x the SDOT kernel on M1. Gated `not(i8mm)` so i8mm builds keep
    // the SMMLA path above. Uses the interleave-4 weight cache (distinct layout).
    #[cfg(all(target_feature = "dotprod", not(target_feature = "i8mm")))]
    {
        if m >= 4 && *PREFILL_LANEROW {
            let packed = packed_laneq4_cache_get(rhs_q4k, n, nb);
            matmul_q4kx8l_lanerow((m, k, n), lhs, &packed, dst);
            return;
        }
    }
    let packed = packed_cache_get(rhs_q4k, n, nb);
    matmul_q4kx8_prepacked((m, k, n), lhs, &packed, dst);
}

// A/B switch for the i8mm prefill path (default ON when compiled with `+i8mm`).
// Set CANDLE_PREFILL_I8MM=0 to force the SDOT path for same-binary benchmarking.
#[cfg(target_feature = "i8mm")]
static PREFILL_I8MM: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("CANDLE_PREFILL_I8MM")
        .map(|s| s != "0")
        .unwrap_or(true)
});

// A/B switch for the lane=row prefill path (default ON for dotprod-without-i8mm
// builds). Set CANDLE_PREFILL_LANEROW=0 to force the legacy SDOT kernel for
// same-binary benchmarking on N1.
#[cfg(all(target_feature = "dotprod", not(target_feature = "i8mm")))]
static PREFILL_LANEROW: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("CANDLE_PREFILL_LANEROW")
        .map(|s| s != "0")
        .unwrap_or(true)
});

// A/B switch for parallelizing the prefill activation quantization across the
// barrier pool (default ON). The lane=row / i8mm drivers quantize all m rows to Q8
// before the parallel GEMM; doing it serially is an Amdahl ceiling on multi-thread
// prefill scaling. Set CANDLE_PREFILL_PARQUANT=0 to force the serial quant for
// same-binary A/B on N1. Result is bit-identical either way (per-tile independent).
#[cfg(target_feature = "neon")]
#[allow(dead_code)]
static PREFILL_PARQUANT: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("CANDLE_PREFILL_PARQUANT")
        .map(|s| s != "0")
        .unwrap_or(true)
});

/// GEMM over an already-interleaved `BlockQ4Kx8` weight (no repack/cache step):
/// `dst(m,n) = lhs(m,k) x W^T`, `packed` holding the `n/8` channel groups
/// (`nb = k/256` blocks each, group g at `g*nb`). Activations are quantized to Q8K
/// here. Identical inner kernels/tiling to `matmul_q4k_packed` - split out so a
/// pre-packed (offline-baked) weight can drive the same SDOT GEMM. Bit-exact.
#[cfg(target_feature = "neon")]
pub(crate) fn matmul_q4kx8_prepacked(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    packed: &[BlockQ4Kx8],
    dst: &mut [f32],
) {
    use super::neon::{gemm_q4kx8_q8k, gemm_q4kx_q8k};
    let nb = k / QK_K;
    let groups = n / Q4KX8_ROWS;
    let tile = *PACKED_PREFILL_TILE;

    // Activation -> Q8K into a thread-local scratch (matmul is called many times per token;
    // this avoids a per-call Vec alloc). Cheap vs the matmul, kept serial.
    thread_local! {
        static LHS_Q: std::cell::RefCell<Vec<BlockQ8K>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    LHS_Q.with(|cell| {
        let mut scratch = cell.borrow_mut();
        if scratch.len() < m * nb {
            scratch.resize(m * nb, BlockQ8K::zeros());
        }
        for a in 0..m {
            BlockQ8K::from_float(&lhs[a * k..(a + 1) * k], &mut scratch[a * nb..(a + 1) * nb]);
        }
        let lhs_q: &[BlockQ8K] = &scratch;

        struct DstPtr(*mut f32);
        unsafe impl Sync for DstPtr {}
        let dptr = DstPtr(dst.as_mut_ptr());
        let process = |g: usize| {
            let w = &packed[g * nb..(g + 1) * nb];
            let p = &dptr;
            let row = |r: usize| -> &[BlockQ8K] { &lhs_q[r * nb..(r + 1) * nb] };
            let mut r = 0;
            match tile {
                PrefillTile::Nc8mr4 => {
                    while r + 4 <= m {
                        let rows: [&[BlockQ8K]; 4] = std::array::from_fn(|a| row(r + a));
                        let mut t = [0f32; Q4KX8_ROWS * 4];
                        gemm_q4kx8_q8k::<4>(w, &rows, &mut t);
                        for c in 0..Q4KX8_ROWS {
                            for a in 0..4 {
                                unsafe { *p.0.add((r + a) * n + g * 8 + c) = t[c * 4 + a] };
                            }
                        }
                        r += 4;
                    }
                }
                PrefillTile::Nc4mr4 => {
                    while r + 4 <= m {
                        let rows: [&[BlockQ8K]; 4] = std::array::from_fn(|a| row(r + a));
                        for half in 0..2 {
                            let c0 = half * 4;
                            let mut t = [0f32; 4 * 4];
                            gemm_q4kx_q8k::<4, 4>(w, c0, &rows, &mut t);
                            for c in 0..4 {
                                for a in 0..4 {
                                    unsafe {
                                        *p.0.add((r + a) * n + g * 8 + c0 + c) = t[c * 4 + a]
                                    };
                                }
                            }
                        }
                        r += 4;
                    }
                }
                PrefillTile::Nc8mr2 => {}
            }
            while r + 2 <= m {
                let rows: [&[BlockQ8K]; 2] = std::array::from_fn(|a| row(r + a));
                let mut t = [0f32; Q4KX8_ROWS * 2];
                gemm_q4kx8_q8k::<2>(w, &rows, &mut t);
                for c in 0..Q4KX8_ROWS {
                    for a in 0..2 {
                        unsafe { *p.0.add((r + a) * n + g * 8 + c) = t[c * 2 + a] };
                    }
                }
                r += 2;
            }
            while r < m {
                let rows: [&[BlockQ8K]; 1] = [row(r)];
                let mut t = [0f32; Q4KX8_ROWS];
                gemm_q4kx8_q8k::<1>(w, &rows, &mut t);
                for c in 0..Q4KX8_ROWS {
                    unsafe { *p.0.add(r * n + g * 8 + c) = t[c] };
                }
                r += 1;
            }
        };

        // CONTIGUOUS per-thread group partition on the barrier pool (replaces rayon's
        // fine-grained par_iter, whose work-stealing thrashed the shared cache on N1 and
        // collapsed multi-thread decode scaling - see bench/perf_bytes.sh). `execute` with a
        // 0-worker pool just runs f(0) inline, so the 1-vCPU path has zero pool overhead.
        let pool = crate::utils::barrier_pool();
        let n_total = pool.n_workers() + 1;
        let gpt = groups.div_ceil(n_total);
        pool.execute(|tid| {
            let start = tid * gpt;
            if start < groups {
                let end = groups.min((tid + 1) * gpt);
                for g in start..end {
                    process(g);
                }
            }
        });
    });
}

/// Repack a Q4_K weight (`n` channels x `nb` blocks, row-major) into the laneq
/// `BlockQ4Kx8L` groups the i8mm GEMM consumes. Mirrors `repack_q4k_weight` but
/// emits the SMMLA layout (`repack_q4kx8_laneq` per 8-channel group).
#[allow(dead_code)] // used only by the i8mm dispatch (`+i8mm` builds)
pub(crate) fn repack_q4k_weight_laneq(rows: &[BlockQ4K], n: usize, nb: usize) -> Vec<BlockQ4Kx8L> {
    debug_assert_eq!(n % Q4KX8_ROWS, 0, "repack_q4k_weight_laneq: n {n} not /8");
    debug_assert_eq!(rows.len(), n * nb, "repack_q4k_weight_laneq: rows len mismatch");
    let mut packed = Vec::with_capacity((n / 8) * nb);
    for g in 0..n / 8 {
        let grp: [&[BlockQ4K]; Q4KX8_ROWS] =
            std::array::from_fn(|r| &rows[(g * 8 + r) * nb..(g * 8 + r + 1) * nb]);
        packed.extend(repack_q4kx8_laneq(&grp));
    }
    packed
}

/// Laneq-layout repack cache, keyed on the source Q4_K pointer (parallel to
/// `PACKED_CACHE`). Populated on first i8mm prefill of each weight.
#[allow(dead_code)]
static LANEQ_CACHE: LazyLock<Mutex<HashMap<usize, Arc<Vec<BlockQ4Kx8L>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[allow(dead_code)]
fn packed_laneq_cache_get(rhs: &[BlockQ4K], n: usize, nb: usize) -> Arc<Vec<BlockQ4Kx8L>> {
    let key = rhs.as_ptr() as usize;
    {
        if let Some(p) = LANEQ_CACHE.lock().unwrap().get(&key) {
            return p.clone();
        }
    }
    let mut map = LANEQ_CACHE.lock().unwrap();
    if let Some(p) = map.get(&key) {
        return p.clone();
    }
    let arc = Arc::new(repack_q4k_weight_laneq(rhs, n, nb));
    map.insert(key, arc.clone());
    arc
}

/// i8mm prefill GEMM driver: `dst(m,n) = lhs(m,k) x W^T` over laneq-packed weights.
/// Activations are packed to `BlockQ8Kx4` in 4-row tiles (last tile zero-padded),
/// then each 8-channel weight group x 4-row tile is an SMMLA `gemm_q4kx8_q8k_i8mm`
/// call writing a 4x8 sub-tile. Parallelized over channel groups on the barrier
/// pool, exactly like the SDOT driver. Compiles on any neon host (the SMMLA op
/// falls back to its scalar twin), so the tiling/scatter is testable without i8mm.
#[cfg(target_feature = "neon")]
#[allow(dead_code)] // taken only by the i8mm dispatch (`+i8mm` builds) + tests
pub(crate) fn matmul_q4kx8l_prepacked_i8mm(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    packed: &[BlockQ4Kx8L],
    dst: &mut [f32],
) {
    use super::neon::gemm_q4kx8_q8k_i8mm;
    let nb = k / QK_K;
    let groups = n / Q4KX8_ROWS;
    let row_tiles = m.div_ceil(4);

    // Pack activations once (reused across all weight groups). The last tile's
    // missing rows are zero - they contribute 0 and are not scattered to `dst`.
    // Parallelized over row-tiles on the barrier pool (each tile independent), so
    // the quant doesn't serialize multi-thread prefill; CANDLE_PREFILL_PARQUANT=0
    // forces serial. Identical bytes either way.
    let zeros = vec![0f32; k];
    let mut q8: Vec<BlockQ8Kx4> = vec![BlockQ8Kx4::zeroed(); row_tiles * nb];
    let tile_rows = |rt: usize| -> [&[f32]; 4] {
        std::array::from_fn(|a| {
            let r = rt * 4 + a;
            if r < m {
                &lhs[r * k..(r + 1) * k]
            } else {
                zeros.as_slice()
            }
        })
    };
    if *PREFILL_PARQUANT {
        crate::utils::par_chunks_mut(&mut q8, nb, |rt, chunk| {
            quantize_mat_q8_k_4x8_into(&tile_rows(rt), chunk);
        });
    } else {
        for rt in 0..row_tiles {
            quantize_mat_q8_k_4x8_into(&tile_rows(rt), &mut q8[rt * nb..(rt + 1) * nb]);
        }
    }

    struct DstPtr(*mut f32);
    unsafe impl Sync for DstPtr {}
    let dptr = DstPtr(dst.as_mut_ptr());

    let process = |g: usize| {
        let w = &packed[g * nb..(g + 1) * nb];
        let p = &dptr;
        for rt in 0..row_tiles {
            let q8t = &q8[rt * nb..(rt + 1) * nb];
            let mut t = [0f32; 32];
            gemm_q4kx8_q8k_i8mm(w, q8t, &mut t);
            for a in 0..4 {
                let r = rt * 4 + a;
                if r >= m {
                    break;
                }
                for c in 0..Q4KX8_ROWS {
                    unsafe { *p.0.add(r * n + g * 8 + c) = t[a * 8 + c] };
                }
            }
        }
    };

    let pool = crate::utils::barrier_pool();
    let n_total = pool.n_workers() + 1;
    let gpt = groups.div_ceil(n_total);
    pool.execute(|tid| {
        let start = tid * gpt;
        if start < groups {
            let end = groups.min((tid + 1) * gpt);
            for g in start..end {
                process(g);
            }
        }
    });
}

/// Repack a Q4_K weight into the interleave-4 laneq `BlockQ4Kx8L` groups the
/// lane=row `8x4` kernel consumes (`repack_q4kx8_laneq4` per 8-channel group).
#[allow(dead_code)] // used only by the lane=row dispatch (dotprod-no-i8mm builds)
pub(crate) fn repack_q4k_weight_laneq4(rows: &[BlockQ4K], n: usize, nb: usize) -> Vec<BlockQ4Kx8L> {
    debug_assert_eq!(n % Q4KX8_ROWS, 0, "repack_q4k_weight_laneq4: n {n} not /8");
    debug_assert_eq!(rows.len(), n * nb, "repack_q4k_weight_laneq4: rows len mismatch");
    let mut packed = Vec::with_capacity((n / 8) * nb);
    for g in 0..n / 8 {
        let grp: [&[BlockQ4K]; Q4KX8_ROWS] =
            std::array::from_fn(|r| &rows[(g * 8 + r) * nb..(g * 8 + r + 1) * nb]);
        packed.extend(repack_q4kx8_laneq4(&grp));
    }
    packed
}

/// Interleave-4 laneq repack cache, keyed on the source Q4_K pointer (parallel to
/// `LANEQ_CACHE`). Populated on first lane=row prefill of each weight.
#[allow(dead_code)]
static LANEQ4_CACHE: LazyLock<Mutex<HashMap<usize, Arc<Vec<BlockQ4Kx8L>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[allow(dead_code)]
fn packed_laneq4_cache_get(rhs: &[BlockQ4K], n: usize, nb: usize) -> Arc<Vec<BlockQ4Kx8L>> {
    let key = rhs.as_ptr() as usize;
    {
        if let Some(p) = LANEQ4_CACHE.lock().unwrap().get(&key) {
            return p.clone();
        }
    }
    let mut map = LANEQ4_CACHE.lock().unwrap();
    if let Some(p) = map.get(&key) {
        return p.clone();
    }
    let arc = Arc::new(repack_q4k_weight_laneq4(rhs, n, nb));
    map.insert(key, arc.clone());
    arc
}

/// Lane=row prefill GEMM driver: `dst(m,n) = lhs(m,k) x W^T` over interleave-4
/// laneq weights. Mirrors `matmul_q4kx8l_prepacked_i8mm` exactly (4-row activation
/// tiles via `quantize_mat_q8_k_4x4`, last tile zero-padded, barrier-pool over
/// channel groups, same 4x8 sub-tile scatter), but calls the lane=row SDOT kernel
/// instead of SMMLA - so it runs on any dotprod core (Graviton2/N1).
#[cfg(all(target_feature = "neon", target_feature = "dotprod"))]
#[allow(dead_code)] // taken by the lane=row dispatch (dotprod-no-i8mm) + tests
pub(crate) fn matmul_q4kx8l_lanerow(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    packed: &[BlockQ4Kx8L],
    dst: &mut [f32],
) {
    use super::neon::gemm_q4kx8_q8k_lanerow;
    let nb = k / QK_K;
    let groups = n / Q4KX8_ROWS;
    let row_tiles = m.div_ceil(4);

    // Quantize all m activation rows to Q8 (4x4 interleave) into 4-row tiles, the
    // last zero-padded. Parallelized over row-tiles on the barrier pool (each tile
    // is independent), so it does not serialize multi-thread prefill; identical
    // bytes to the serial path. CANDLE_PREFILL_PARQUANT=0 forces serial for A/B.
    let zeros = vec![0f32; k];
    let mut q8: Vec<BlockQ8Kx4> = vec![BlockQ8Kx4::zeroed(); row_tiles * nb];
    let tile_rows = |rt: usize| -> [&[f32]; 4] {
        std::array::from_fn(|a| {
            let r = rt * 4 + a;
            if r < m {
                &lhs[r * k..(r + 1) * k]
            } else {
                zeros.as_slice()
            }
        })
    };
    if *PREFILL_PARQUANT {
        crate::utils::par_chunks_mut(&mut q8, nb, |rt, chunk| {
            quantize_mat_q8_k_4x4_into(&tile_rows(rt), chunk);
        });
    } else {
        for rt in 0..row_tiles {
            quantize_mat_q8_k_4x4_into(&tile_rows(rt), &mut q8[rt * nb..(rt + 1) * nb]);
        }
    }

    struct DstPtr(*mut f32);
    unsafe impl Sync for DstPtr {}
    let dptr = DstPtr(dst.as_mut_ptr());

    let process = |g: usize| {
        let w = &packed[g * nb..(g + 1) * nb];
        let p = &dptr;
        for rt in 0..row_tiles {
            let q8t = &q8[rt * nb..(rt + 1) * nb];
            let mut t = [0f32; 32];
            gemm_q4kx8_q8k_lanerow(w, q8t, &mut t);
            for a in 0..4 {
                let r = rt * 4 + a;
                if r >= m {
                    break;
                }
                for c in 0..Q4KX8_ROWS {
                    unsafe { *p.0.add(r * n + g * 8 + c) = t[a * 8 + c] };
                }
            }
        }
    };

    let pool = crate::utils::barrier_pool();
    let n_total = pool.n_workers() + 1;
    let gpt = groups.div_ceil(n_total);
    pool.execute(|tid| {
        let start = tid * gpt;
        if start < groups {
            let end = groups.min((tid + 1) * gpt);
            for g in start..end {
                process(g);
            }
        }
    });
}

/// GEMM over an already-interleaved `BlockQ6Kx8` weight: `dst(m,n) = lhs(m,k) x W^T`.
/// Mirrors `matmul_q4kx8_prepacked` (same Q8K activation quant, same group
/// parallelism, same serial 1-vCPU fast-path), but the Q6 packed kernel only has
/// the 8-channel form (`gemm_q6kx8_q8k<MR>`), so the tile strategy is simply MR=2
/// for prefill with a 1-row remainder (no nc4 variant). Bit-exact to `vec_dot_q6k`.
#[cfg(target_feature = "neon")]
pub(crate) fn matmul_q6kx8_prepacked(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    packed: &[BlockQ6Kx8],
    dst: &mut [f32],
) {
    use super::neon::gemm_q6kx8_q8k;
    let nb = k / QK_K;
    let groups = n / Q4KX8_ROWS;
    // Q6 has only the 8-channel kernel; widen the prefill tile to MR=4 (the nc8mr4
    // default) for the same channel+row reuse win, else MR=2.
    let mr4 = *PACKED_PREFILL_TILE == PrefillTile::Nc8mr4;

    thread_local! {
        static LHS_Q: std::cell::RefCell<Vec<BlockQ8K>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    LHS_Q.with(|cell| {
        let mut scratch = cell.borrow_mut();
        if scratch.len() < m * nb {
            scratch.resize(m * nb, BlockQ8K::zeros());
        }
        for a in 0..m {
            BlockQ8K::from_float(&lhs[a * k..(a + 1) * k], &mut scratch[a * nb..(a + 1) * nb]);
        }
        let lhs_q: &[BlockQ8K] = &scratch;

        struct DstPtr(*mut f32);
        unsafe impl Sync for DstPtr {}
        let dptr = DstPtr(dst.as_mut_ptr());
        let process = |g: usize| {
            let w = &packed[g * nb..(g + 1) * nb];
            let p = &dptr;
            let row = |r: usize| -> &[BlockQ8K] { &lhs_q[r * nb..(r + 1) * nb] };
            let mut r = 0;
            if mr4 {
                while r + 4 <= m {
                    let rows: [&[BlockQ8K]; 4] = std::array::from_fn(|a| row(r + a));
                    let mut t = [0f32; Q4KX8_ROWS * 4];
                    gemm_q6kx8_q8k::<4>(w, &rows, &mut t);
                    for c in 0..Q4KX8_ROWS {
                        for a in 0..4 {
                            unsafe { *p.0.add((r + a) * n + g * 8 + c) = t[c * 4 + a] };
                        }
                    }
                    r += 4;
                }
            }
            while r + 2 <= m {
                let rows: [&[BlockQ8K]; 2] = std::array::from_fn(|a| row(r + a));
                let mut t = [0f32; Q4KX8_ROWS * 2];
                gemm_q6kx8_q8k::<2>(w, &rows, &mut t);
                for c in 0..Q4KX8_ROWS {
                    for a in 0..2 {
                        unsafe { *p.0.add((r + a) * n + g * 8 + c) = t[c * 2 + a] };
                    }
                }
                r += 2;
            }
            while r < m {
                let rows: [&[BlockQ8K]; 1] = [row(r)];
                let mut t = [0f32; Q4KX8_ROWS];
                gemm_q6kx8_q8k::<1>(w, &rows, &mut t);
                for c in 0..Q4KX8_ROWS {
                    unsafe { *p.0.add(r * n + g * 8 + c) = t[c] };
                }
                r += 1;
            }
        };

        let pool = crate::utils::barrier_pool();
        let n_total = pool.n_workers() + 1;
        let gpt = groups.div_ceil(n_total);
        pool.execute(|tid| {
            let start = tid * gpt;
            if start < groups {
                let end = groups.min((tid + 1) * gpt);
                for g in start..end {
                    process(g);
                }
            }
        });
    });
}

// ===========================================================================
// Pre-packed Q4_Kx8 as a first-class quantized storage type (GgmlDType::Q4Kx8).
//
// The interleaved BlockQ4Kx8 layout - normally produced by repack-on-load and
// cached - can instead be baked into a GGUF offline (see the gguf-requant
// `--pack` flag) and loaded as a SINGLE copy with no runtime repack. Owned or
// mmap-backed; the mmap path is zero-copy (GGUF tensors are 32-aligned, which
// covers BlockQ4Kx8's f16/u8 alignment of 2).
// ===========================================================================

/// Backing store for a `PackedQ4Kx8`: either an owned `Vec` (built from raw bytes)
/// or a view into a memory-mapped GGUF region (zero-copy, read-only).
enum PackedStore {
    Owned(Vec<BlockQ4Kx8>),
    Mmap {
        mmap: Arc<memmap2::Mmap>,
        offset: usize, // byte offset of the first block within the mmap
        count: usize,  // number of BlockQ4Kx8
    },
}

// SAFETY: the Mmap variant references an immutable, file-backed byte region kept
// alive by the Arc; BlockQ4Kx8 is a POD (#[repr(C)] of f16/u8). Access is
// read-only and the Arc makes the mapping outlive the view.
unsafe impl Send for PackedStore {}
unsafe impl Sync for PackedStore {}

/// A pre-packed Q4_K weight: `n` output channels (`n % 8 == 0`) stored as the
/// interleaved `BlockQ4Kx8` groups, ready for the SDOT GEMM with no repack.
pub struct PackedQ4Kx8 {
    store: PackedStore,
    n: usize,
}

impl PackedQ4Kx8 {
    #[inline]
    fn as_slice(&self) -> &[BlockQ4Kx8] {
        match &self.store {
            PackedStore::Owned(v) => v.as_slice(),
            PackedStore::Mmap {
                mmap,
                offset,
                count,
            } => {
                // SAFETY: offset/count/alignment checked in `from_mmap`; Arc keeps
                // the mapping alive for the lifetime of the returned slice.
                unsafe {
                    std::slice::from_raw_parts(
                        mmap.as_ptr().add(*offset) as *const BlockQ4Kx8,
                        *count,
                    )
                }
            }
        }
    }

    /// Build an owned `PackedQ4Kx8` by copying raw interleaved bytes (e.g. from a
    /// GGUF tensor read into a Vec). `raw` length must be a whole number of blocks.
    pub fn from_bytes(raw: &[u8], n: usize) -> Self {
        let bs = std::mem::size_of::<BlockQ4Kx8>();
        assert_eq!(
            raw.len() % bs,
            0,
            "PackedQ4Kx8::from_bytes: {} bytes not a multiple of block size {bs}",
            raw.len()
        );
        let count = raw.len() / bs;
        // Block is #[repr(C)] POD; a raw byte copy reproduces it exactly, so copy
        // straight into uninitialized capacity rather than zero-filling first.
        let mut v: Vec<BlockQ4Kx8> = Vec::with_capacity(count);
        unsafe {
            std::ptr::copy_nonoverlapping(raw.as_ptr(), v.as_mut_ptr() as *mut u8, raw.len());
            v.set_len(count);
        }
        Self {
            store: PackedStore::Owned(v),
            n,
        }
    }

    /// Build a zero-copy `PackedQ4Kx8` viewing `byte_len` bytes at `offset` in an
    /// mmap'd GGUF. Validates bounds and alignment.
    pub fn from_mmap(
        mmap: Arc<memmap2::Mmap>,
        offset: usize,
        byte_len: usize,
        n: usize,
    ) -> crate::Result<Self> {
        let bs = std::mem::size_of::<BlockQ4Kx8>();
        if offset + byte_len > mmap.len() {
            crate::bail!(
                "Q4Kx8 mmap region end {} exceeds map len {}",
                offset + byte_len,
                mmap.len()
            );
        }
        if !byte_len.is_multiple_of(bs) {
            crate::bail!("Q4Kx8 mmap byte_len {byte_len} not a multiple of block size {bs}");
        }
        let base = mmap.as_ptr() as usize + offset;
        if !base.is_multiple_of(std::mem::align_of::<BlockQ4Kx8>()) {
            crate::bail!(
                "Q4Kx8 mmap tensor at offset {offset} not aligned to {}",
                std::mem::align_of::<BlockQ4Kx8>()
            );
        }
        Ok(Self {
            store: PackedStore::Mmap {
                mmap,
                offset,
                count: byte_len / bs,
            },
            n,
        })
    }

    /// Dequantize the packed weight back to f32 of shape `[n, k]` (row-major), where
    /// `k = elem_count / n`. Mirrors `BlockQ4K::to_float` per channel: per sub-block
    /// scale/min are the pre-unpacked bytes; nibbles come from the chunk-major,
    /// row-interleaved `qs`. Correctness-only (not perf).
    fn dequantize_to(&self, elem_count: usize, ys: &mut [f32]) {
        let n = self.n;
        let k = elem_count / n;
        let nb = k / QK_K;
        let blocks = self.as_slice();
        debug_assert_eq!(blocks.len(), (n / Q4KX8_ROWS) * nb);
        for g in 0..n / Q4KX8_ROWS {
            for r in 0..Q4KX8_ROWS {
                let out_row = g * Q4KX8_ROWS + r;
                for i in 0..nb {
                    let blk = &blocks[g * nb + i];
                    let d = blk.d[r].to_f32();
                    let dmin = blk.dmin[r].to_f32();
                    let base = out_row * k + i * QK_K;
                    // 4 chunks of 64 weights; each chunk = 2 sub-blocks (lo, hi).
                    for j in 0..CHUNKS {
                        let sub_lo = 2 * j;
                        let sub_hi = 2 * j + 1;
                        let d1 = d * blk.scales[r * SUBBLOCKS + sub_lo] as f32;
                        let m1 = dmin * blk.mins[r * SUBBLOCKS + sub_lo] as f32;
                        let d2 = d * blk.scales[r * SUBBLOCKS + sub_hi] as f32;
                        let m2 = dmin * blk.mins[r * SUBBLOCKS + sub_hi] as f32;
                        let qoff = j * (Q4KX8_ROWS * 32) + r * 32;
                        let qs = &blk.qs[qoff..qoff + 32];
                        let cbase = base + j * 64;
                        for l in 0..32 {
                            ys[cbase + l] = d1 * (qs[l] & 0xF) as f32 - m1;
                        }
                        for l in 0..32 {
                            ys[cbase + 32 + l] = d2 * (qs[l] >> 4) as f32 - m2;
                        }
                    }
                }
            }
        }
    }
}

impl super::QuantizedType for PackedQ4Kx8 {
    #[cfg(target_feature = "neon")]
    fn matmul_t(
        &self,
        mkn: (usize, usize, usize),
        lhs: &[f32],
        dst: &mut [f32],
    ) -> crate::Result<()> {
        matmul_q4kx8_prepacked(mkn, lhs, self.as_slice(), dst);
        Ok(())
    }

    #[cfg(not(target_feature = "neon"))]
    fn matmul_t(
        &self,
        _mkn: (usize, usize, usize),
        _lhs: &[f32],
        _dst: &mut [f32],
    ) -> crate::Result<()> {
        crate::bail!("Q4Kx8 packed matmul requires the neon target feature")
    }

    fn matmul_t_f16(
        &self,
        mkn: (usize, usize, usize),
        lhs: &[f16],
        dst: &mut [f16],
    ) -> crate::Result<()> {
        // qwen3 runs f32 activations; provide an f32-detour for the rare f16 caller.
        let lhs_f: Vec<f32> = lhs.iter().map(|x| x.to_f32()).collect();
        let mut dst_f = vec![0f32; dst.len()];
        self.matmul_t(mkn, &lhs_f, &mut dst_f)?;
        for (o, v) in dst.iter_mut().zip(dst_f.iter()) {
            *o = f16::from_f32(*v);
        }
        Ok(())
    }

    fn dequantize(&self, elem_count: usize) -> crate::Result<super::CpuStorage> {
        let mut ys = vec![0f32; elem_count];
        self.dequantize_to(elem_count, &mut ys);
        Ok(super::CpuStorage::F32(ys))
    }

    fn storage_size_in_bytes(&self) -> usize {
        std::mem::size_of_val(self.as_slice())
    }

    fn size(&self) -> usize {
        self.storage_size_in_bytes()
    }

    fn as_ptr(&self) -> *const u8 {
        self.as_slice().as_ptr() as *const u8
    }

    fn block_size(&self) -> usize {
        QK_K
    }

    fn dtype(&self) -> super::GgmlDType {
        super::GgmlDType::Q4Kx8
    }

    fn from_float(&mut self, _xs: &[f32]) {
        unreachable!("Q4Kx8 is bake-only; quantize-to is not supported")
    }

    fn from_float_imatrix(&mut self, _xs: &[f32], _imatrix_weights: &[f32], _n_per_row: usize) {
        unreachable!("Q4Kx8 is bake-only; quantize-to is not supported")
    }
}

// ===========================================================================
// Pre-packed Q6_Kx8 as a first-class quantized storage type (GgmlDType::Q6Kx8).
// The Q6_K analogue of PackedQ4Kx8: the residual attn_v/ffn_down weights that
// Q4_K_M keeps at Q6_K, baked into the interleaved BlockQ6Kx8 layout so they ride
// the same SDOT GEMM with no runtime repack. Same Owned/mmap backing story.
// ===========================================================================

/// Backing store for a `PackedQ6Kx8`: owned `Vec` or a zero-copy mmap view.
enum PackedStoreQ6 {
    Owned(Vec<BlockQ6Kx8>),
    Mmap {
        mmap: Arc<memmap2::Mmap>,
        offset: usize,
        count: usize,
    },
}

// SAFETY: identical to PackedStore - the Mmap variant references an immutable,
// file-backed region kept alive by the Arc; BlockQ6Kx8 is a #[repr(C)] POD.
unsafe impl Send for PackedStoreQ6 {}
unsafe impl Sync for PackedStoreQ6 {}

/// A pre-packed Q6_K weight: `n` output channels (`n % 8 == 0`) stored as the
/// interleaved `BlockQ6Kx8` groups, ready for the SDOT GEMM with no repack.
pub struct PackedQ6Kx8 {
    store: PackedStoreQ6,
    n: usize,
}

impl PackedQ6Kx8 {
    #[inline]
    fn as_slice(&self) -> &[BlockQ6Kx8] {
        match &self.store {
            PackedStoreQ6::Owned(v) => v.as_slice(),
            PackedStoreQ6::Mmap {
                mmap,
                offset,
                count,
            } => {
                // SAFETY: offset/count/alignment checked in `from_mmap`; Arc keeps
                // the mapping alive for the lifetime of the returned slice.
                unsafe {
                    std::slice::from_raw_parts(
                        mmap.as_ptr().add(*offset) as *const BlockQ6Kx8,
                        *count,
                    )
                }
            }
        }
    }

    /// Build an owned `PackedQ6Kx8` by copying raw interleaved bytes. `raw` length
    /// must be a whole number of blocks.
    pub fn from_bytes(raw: &[u8], n: usize) -> Self {
        let bs = std::mem::size_of::<BlockQ6Kx8>();
        assert_eq!(
            raw.len() % bs,
            0,
            "PackedQ6Kx8::from_bytes: {} bytes not a multiple of block size {bs}",
            raw.len()
        );
        let count = raw.len() / bs;
        let mut v: Vec<BlockQ6Kx8> = Vec::with_capacity(count);
        unsafe {
            std::ptr::copy_nonoverlapping(raw.as_ptr(), v.as_mut_ptr() as *mut u8, raw.len());
            v.set_len(count);
        }
        Self {
            store: PackedStoreQ6::Owned(v),
            n,
        }
    }

    /// Build a zero-copy `PackedQ6Kx8` viewing `byte_len` bytes at `offset` in an
    /// mmap'd GGUF. Validates bounds and alignment.
    pub fn from_mmap(
        mmap: Arc<memmap2::Mmap>,
        offset: usize,
        byte_len: usize,
        n: usize,
    ) -> crate::Result<Self> {
        let bs = std::mem::size_of::<BlockQ6Kx8>();
        if offset + byte_len > mmap.len() {
            crate::bail!(
                "Q6Kx8 mmap region end {} exceeds map len {}",
                offset + byte_len,
                mmap.len()
            );
        }
        if !byte_len.is_multiple_of(bs) {
            crate::bail!("Q6Kx8 mmap byte_len {byte_len} not a multiple of block size {bs}");
        }
        let base = mmap.as_ptr() as usize + offset;
        if !base.is_multiple_of(std::mem::align_of::<BlockQ6Kx8>()) {
            crate::bail!(
                "Q6Kx8 mmap tensor at offset {offset} not aligned to {}",
                std::mem::align_of::<BlockQ6Kx8>()
            );
        }
        Ok(Self {
            store: PackedStoreQ6::Mmap {
                mmap,
                offset,
                count: byte_len / bs,
            },
            n,
        })
    }

    /// Dequantize the packed weight back to f32 of shape `[n, k]` (row-major).
    /// Mirrors `BlockQ6K::to_float` per channel: the i8 sub-block scales and the
    /// `ql`/`qh` quants come from the chunk-major, row-interleaved layout produced
    /// by `repack_q6k_x8`. Correctness-only (not perf).
    fn dequantize_to(&self, elem_count: usize, ys: &mut [f32]) {
        const QLC: usize = QK_K / 4; // 64 ql bytes / 128-value chunk
        const QHC: usize = QK_K / 8; // 32 qh bytes / chunk
        const NSC: usize = QK_K / 16; // 16 i8 scales / row
        let n = self.n;
        let k = elem_count / n;
        let nb = k / QK_K;
        let blocks = self.as_slice();
        debug_assert_eq!(blocks.len(), (n / Q4KX8_ROWS) * nb);
        for g in 0..n / Q4KX8_ROWS {
            for r in 0..Q4KX8_ROWS {
                let out_row = g * Q4KX8_ROWS + r;
                for i in 0..nb {
                    let blk = &blocks[g * nb + i];
                    let d = blk.d[r].to_f32();
                    let base = out_row * k + i * QK_K;
                    // Two 128-value chunks, exactly as BlockQ6K::to_float steps by 128.
                    for idx in 0..2 {
                        let sc = &blk.scales[r * NSC + 8 * idx..];
                        let ql = &blk.ql[idx * (Q4KX8_ROWS * QLC) + r * QLC..];
                        let qh = &blk.qh[idx * (Q4KX8_ROWS * QHC) + r * QHC..];
                        let cbase = base + idx * 128;
                        for l in 0..32 {
                            let is = l / 16;
                            let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
                            let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                            ys[cbase + l] = d * sc[is] as f32 * q1 as f32;
                            ys[cbase + l + 32] = d * sc[is + 2] as f32 * q2 as f32;
                            ys[cbase + l + 64] = d * sc[is + 4] as f32 * q3 as f32;
                            ys[cbase + l + 96] = d * sc[is + 6] as f32 * q4 as f32;
                        }
                    }
                }
            }
        }
    }
}

impl super::QuantizedType for PackedQ6Kx8 {
    #[cfg(target_feature = "neon")]
    fn matmul_t(
        &self,
        mkn: (usize, usize, usize),
        lhs: &[f32],
        dst: &mut [f32],
    ) -> crate::Result<()> {
        matmul_q6kx8_prepacked(mkn, lhs, self.as_slice(), dst);
        Ok(())
    }

    #[cfg(not(target_feature = "neon"))]
    fn matmul_t(
        &self,
        _mkn: (usize, usize, usize),
        _lhs: &[f32],
        _dst: &mut [f32],
    ) -> crate::Result<()> {
        crate::bail!("Q6Kx8 packed matmul requires the neon target feature")
    }

    fn matmul_t_f16(
        &self,
        mkn: (usize, usize, usize),
        lhs: &[f16],
        dst: &mut [f16],
    ) -> crate::Result<()> {
        let lhs_f: Vec<f32> = lhs.iter().map(|x| x.to_f32()).collect();
        let mut dst_f = vec![0f32; dst.len()];
        self.matmul_t(mkn, &lhs_f, &mut dst_f)?;
        for (o, v) in dst.iter_mut().zip(dst_f.iter()) {
            *o = f16::from_f32(*v);
        }
        Ok(())
    }

    fn dequantize(&self, elem_count: usize) -> crate::Result<super::CpuStorage> {
        let mut ys = vec![0f32; elem_count];
        self.dequantize_to(elem_count, &mut ys);
        Ok(super::CpuStorage::F32(ys))
    }

    fn storage_size_in_bytes(&self) -> usize {
        std::mem::size_of_val(self.as_slice())
    }

    fn size(&self) -> usize {
        self.storage_size_in_bytes()
    }

    fn as_ptr(&self) -> *const u8 {
        self.as_slice().as_ptr() as *const u8
    }

    fn block_size(&self) -> usize {
        QK_K
    }

    fn dtype(&self) -> super::GgmlDType {
        super::GgmlDType::Q6Kx8
    }

    fn from_float(&mut self, _xs: &[f32]) {
        unreachable!("Q6Kx8 is bake-only; quantize-to is not supported")
    }

    fn from_float_imatrix(&mut self, _xs: &[f32], _imatrix_weights: &[f32], _n_per_row: usize) {
        unreachable!("Q6Kx8 is bake-only; quantize-to is not supported")
    }
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

    // gemm_q6kx8_q8k (packed Q6_K GEMV) must be BIT-IDENTICAL to the scalar
    // vec_dot_q6k_q8k applied per row - it mirrors the same integer ops/order.
    #[cfg(target_feature = "neon")]
    #[test]
    fn gemm_q6kx8_matches_vec_dot() {
        use crate::quantized::k_quants::{BlockQ6K, BlockQ8K};
        use crate::quantized::neon::{gemm_q6kx8_q8k, vec_dot_q6k_q8k};

        let nb = 4usize;
        let k = nb * QK_K;
        let mut st = 0x1357_9bdf_2468_ace0u64;
        let mut rows_q: Vec<Vec<BlockQ6K>> = Vec::new();
        for _ in 0..Q4KX8_ROWS {
            let f: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
            let mut q = vec![BlockQ6K::zeros(); nb];
            BlockQ6K::from_float(&f, &mut q);
            rows_q.push(q);
        }
        let af: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
        let mut q8 = vec![BlockQ8K::zeros(); nb];
        BlockQ8K::from_float(&af, &mut q8);

        let reference: [f32; 8] = std::array::from_fn(|r| vec_dot_q6k_q8k(k, &rows_q[r], &q8));
        let refs: [&[BlockQ6K]; Q4KX8_ROWS] = std::array::from_fn(|r| rows_q[r].as_slice());
        let packed = repack_q6k_x8(&refs);
        let mut dst = [0f32; 8];
        gemm_q6kx8_q8k::<1>(&packed, &[q8.as_slice()], &mut dst);
        for c in 0..8 {
            assert_eq!(
                dst[c].to_bits(),
                reference[c].to_bits(),
                "channel {c}: packed {} vs ref {}",
                dst[c],
                reference[c]
            );
        }
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
                    assert_eq!(
                        packed[i].scales[r * SUBBLOCKS + s],
                        sc[s],
                        "scale r{r} i{i} s{s}"
                    );
                    assert_eq!(
                        packed[i].mins[r * SUBBLOCKS + s],
                        mn[s],
                        "min r{r} i{i} s{s}"
                    );
                }
                assert_eq!(packed[i].d[r].to_bits(), src.d.to_bits(), "d r{r} i{i}");
                assert_eq!(
                    packed[i].dmin[r].to_bits(),
                    src.dmin.to_bits(),
                    "dmin r{r} i{i}"
                );
                assert_eq!(
                    packed_nibbles(&packed[i], r),
                    ref_nibbles(src),
                    "nibbles r{r} i{i}"
                );
            }
        }
    }

    // The packed SDOT GEMM must be bit-exact vs per-pair vec_dot for every
    // (channel, row), across MR tile heights.
    #[cfg(target_feature = "neon")]
    #[test]
    fn gemm_q4kx8_bit_exact() {
        use crate::quantized::k_quants::BlockQ8K;
        use crate::quantized::neon::gemm_q4kx8_q8k;

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
                            $mr,
                            got[c * $mr + a],
                            want
                        );
                    }
                }
            }};
        }
        check!(1);
        check!(2);
        check!(4);

        // NC=4 halves (c_off 0 and 4), MR=4 - the nc4mr4 prefill path.
        use crate::quantized::neon::gemm_q4kx_q8k;
        for c_off in [0usize, 4usize] {
            let ys: [&[BlockQ8K]; 4] = std::array::from_fn(|a| aq[a].as_slice());
            let mut got = vec![0f32; 4 * 4];
            gemm_q4kx_q8k::<4, 4>(&packed, c_off, &ys, &mut got);
            for c in 0..4 {
                for a in 0..4 {
                    let want = BlockQ4K::vec_dot(k, &wq[c_off + c], &aq[a]);
                    assert_eq!(
                        got[c * 4 + a].to_bits(),
                        want.to_bits(),
                        "NC4 c_off={c_off} channel {c} row {a}: got {} want {}",
                        got[c * 4 + a],
                        want
                    );
                }
            }
        }
    }

    // Laneq repack sanity: qs must be a permutation of the 8 columns' qs bytes
    // (every source byte present once), and d/dmin copied exactly. Full numeric
    // correctness comes with the laneq GEMM (compared to vec_dot) in the next step.
    #[test]
    fn repack_q4kx8_laneq_qs_is_permutation() {
        let nb = 2usize;
        let k = nb * QK_K;
        let mut st = 0x7777_3333_1111_9999u64;
        let mut wq: Vec<Vec<BlockQ4K>> = Vec::new();
        for _ in 0..8 {
            let f: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
            let mut q = vec![BlockQ4K::zeros(); nb];
            BlockQ4K::from_float(&f, &mut q);
            wq.push(q);
        }
        let cols: [&[BlockQ4K]; 8] = std::array::from_fn(|c| wq[c].as_slice());
        let packed = repack_q4kx8_laneq(&cols);
        assert_eq!(packed.len(), nb);
        for i in 0..nb {
            // Multiset of qs bytes equal across source columns and packed block.
            let mut src: Vec<u8> = Vec::new();
            for c in 0..8 {
                src.extend_from_slice(&wq[c][i].qs);
            }
            let mut got: Vec<u8> = packed[i].qs.to_vec();
            src.sort_unstable();
            got.sort_unstable();
            assert_eq!(src, got, "qs not a permutation at block {i}");
            for c in 0..8 {
                assert_eq!(
                    packed[i].d[c].to_bits(),
                    wq[c][i].d.to_bits(),
                    "d c{c} i{i}"
                );
                assert_eq!(
                    packed[i].dmin[c].to_bits(),
                    wq[c][i].dmin.to_bits(),
                    "dmin c{c} i{i}"
                );
            }
        }
    }

    // The i8mm/SMMLA 4x8 GEMM tile must match the f32 ground truth (dequant Q4_K
    // weight . dequant Q8 activation). Validates the laneq weight layout, the
    // q8_Kx4 activation pack, the 6-bit scale unpack, and the SMMLA tiling/reorder
    // end to end. The SMMLA op runs via the scalar twin on a non-i8mm host, which
    // is bit-identical to the hardware instruction - so this is the i8mm result.
    #[cfg(target_feature = "neon")]
    #[test]
    fn gemm_q4kx8_i8mm_matches_reference() {
        use crate::quantized::neon::gemm_q4kx8_q8k_i8mm;

        let nb = 3usize;
        let k = nb * QK_K;
        let mut st = 0x1234_5678_9abc_def0u64;

        // 8 weight columns; keep the dequantized f32 for the reference.
        let mut wq: Vec<Vec<BlockQ4K>> = Vec::new();
        let mut wf: Vec<Vec<f32>> = Vec::new();
        for _ in 0..8 {
            let f: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
            let mut q = vec![BlockQ4K::zeros(); nb];
            BlockQ4K::from_float(&f, &mut q);
            let mut deq = vec![0f32; k];
            BlockQ4K::to_float(&q, &mut deq);
            wq.push(q);
            wf.push(deq);
        }
        let cols: [&[BlockQ4K]; 8] = std::array::from_fn(|c| wq[c].as_slice());
        let packed = repack_q4kx8_laneq(&cols);

        // 4 activation rows.
        let af: Vec<Vec<f32>> = (0..4).map(|_| (0..k).map(|_| lcg(&mut st)).collect()).collect();
        let arows: [&[f32]; 4] = std::array::from_fn(|r| af[r].as_slice());
        let q8 = quantize_mat_q8_k_4x8(&arows, nb);

        let mut dst = [0f32; 32];
        gemm_q4kx8_q8k_i8mm(&packed, &q8, &mut dst);

        // Reference: dequant weight . dequant activation, same products the kernel
        // forms (only the f32 accumulation order differs). Reconstruct each row's
        // q8 from the interleaved pack: element e of row r lives at qs[j].
        for r in 0..4 {
            for col in 0..8 {
                let mut sum = 0f64;
                for blk in 0..nb {
                    let qb = &q8[blk];
                    for e in 0..QK_K {
                        let j = (e / 8) * 32 + r * 8 + (e % 8);
                        let actd = qb.d[r] as f64 * qb.qs[j] as f64;
                        sum += wf[col][blk * QK_K + e] as f64 * actd;
                    }
                }
                let got = dst[r * 8 + col] as f64;
                let rel = (got - sum).abs() / sum.abs().max(1e-6);
                assert!(rel < 1e-2, "r{r} col{col}: got {got} ref {sum} rel {rel}");
            }
        }
    }

    // The lane=row SDOT 4x8 GEMM tile (llama's actual N1 prefill kernel) must match
    // the f32 ground truth: validates the laneq weight layout, the q8_Kx4 (4x4)
    // activation pack, the 6-bit scale unpack, and the lane-row tiling/fold. Runs on
    // any dotprod host (incl. M1), so this is a real check of the N1 kernel.
    #[cfg(all(target_feature = "neon", target_feature = "dotprod"))]
    #[test]
    fn gemm_q4kx8_lanerow_matches_reference() {
        use crate::quantized::neon::gemm_q4kx8_q8k_lanerow;

        let nb = 3usize;
        let k = nb * QK_K;
        let mut st = 0x9e37_79b9_7f4a_7c15u64;

        let mut wq: Vec<Vec<BlockQ4K>> = Vec::new();
        let mut wf: Vec<Vec<f32>> = Vec::new();
        for _ in 0..8 {
            let f: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
            let mut q = vec![BlockQ4K::zeros(); nb];
            BlockQ4K::from_float(&f, &mut q);
            let mut deq = vec![0f32; k];
            BlockQ4K::to_float(&q, &mut deq);
            wq.push(q);
            wf.push(deq);
        }
        let cols: [&[BlockQ4K]; 8] = std::array::from_fn(|c| wq[c].as_slice());
        let packed = repack_q4kx8_laneq4(&cols);

        let af: Vec<Vec<f32>> = (0..4).map(|_| (0..k).map(|_| lcg(&mut st)).collect()).collect();
        let arows: [&[f32]; 4] = std::array::from_fn(|r| af[r].as_slice());
        let q8 = quantize_mat_q8_k_4x4(&arows, nb);

        let mut dst = [0f32; 32];
        gemm_q4kx8_q8k_lanerow(&packed, &q8, &mut dst);

        // Reconstruct each row's q8 from the 4x4 interleave: element e of row r
        // lives at qs[j], j = (e/4)*16 + r*4 + (e%4).
        for r in 0..4 {
            for col in 0..8 {
                let mut sum = 0f64;
                for blk in 0..nb {
                    let qb = &q8[blk];
                    for e in 0..QK_K {
                        let j = (e / 4) * 16 + r * 4 + (e % 4);
                        let actd = qb.d[r] as f64 * qb.qs[j] as f64;
                        sum += wf[col][blk * QK_K + e] as f64 * actd;
                    }
                }
                let got = dst[r * 8 + col] as f64;
                let rel = (got - sum).abs() / sum.abs().max(1e-6);
                assert!(rel < 1e-2, "r{r} col{col}: got {got} ref {sum} rel {rel}");
            }
        }
    }

    // The packed driver must match baseline matmul end-to-end (full dst, multiple
    // 8-channel groups), for prefill and decode.
    #[cfg(target_feature = "neon")]
    #[test]
    fn matmul_q4k_packed_matches_baseline() {
        use crate::quantized::k_quants::matmul;

        // This test asserts BIT-exactness of the SDOT packed path. On a `+i8mm`
        // build m>=4 would dispatch to the i8mm path, and on a dotprod build to the
        // lane=row path - both correct but NOT bit-exact (different q8 quant). Force
        // the legacy SDOT kernel. Safe: this is the only test reaching those gates,
        // so the env is read here first.
        std::env::set_var("CANDLE_PREFILL_I8MM", "0");
        std::env::set_var("CANDLE_PREFILL_LANEROW", "0");

        let k = 512usize; // 2 super-blocks
        let n = 24usize; // 3 groups of 8
        let nb = k / QK_K;
        let mut st = 0x2222_4444_6666_8888u64;

        let mut rhs_t = vec![BlockQ4K::zeros(); n * nb];
        for r in 0..n {
            let f: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
            BlockQ4K::from_float(&f, &mut rhs_t[r * nb..(r + 1) * nb]);
        }
        let rhs_q4k: &[BlockQ4K] = &rhs_t;

        for &m in &[1usize, 5usize, 16usize] {
            let lhs: Vec<f32> = (0..m * k).map(|_| lcg(&mut st)).collect();
            let mut dst_base = vec![0f32; m * n];
            let mut dst_pack = vec![0f32; m * n];
            matmul::<BlockQ4K>((m, k, n), &lhs, &rhs_t, &mut dst_base).unwrap();
            clear_packed_cache();
            matmul_q4k_packed((m, k, n), &lhs, rhs_q4k, &mut dst_pack);
            for i in 0..m * n {
                assert_eq!(
                    dst_base[i].to_bits(),
                    dst_pack[i].to_bits(),
                    "m={m} idx {i}: base {} packed {}",
                    dst_base[i],
                    dst_pack[i]
                );
            }
        }
    }

    // The i8mm prefill driver (laneq weights + q8_Kx4 activations + SMMLA) must
    // match the baseline Q4_K matmul. NOT bit-exact: the i8mm path quantizes
    // activations with llama's signed-max convention vs the baseline's abs-max
    // BlockQ8K, so results differ by q8 rounding noise - a relative tolerance
    // catches layout/tiling bugs (which blow up >100%) while allowing that. Covers
    // m not a multiple of 4 (zero-padded remainder tile). Runs on a non-i8mm host
    // via the scalar SMMLA twin (bit-identical to hardware), so it is a real check.
    #[cfg(target_feature = "neon")]
    #[test]
    fn matmul_q4kx8l_i8mm_matches_baseline() {
        use crate::quantized::k_quants::matmul;

        let k = 512usize; // 2 super-blocks
        let n = 24usize; // 3 groups of 8
        let nb = k / QK_K;
        let mut st = 0x0bad_f00d_dead_beefu64;

        let mut rhs_t = vec![BlockQ4K::zeros(); n * nb];
        for r in 0..n {
            let f: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
            BlockQ4K::from_float(&f, &mut rhs_t[r * nb..(r + 1) * nb]);
        }
        let laneq = repack_q4k_weight_laneq(&rhs_t, n, nb);

        for &m in &[4usize, 7usize, 16usize] {
            let lhs: Vec<f32> = (0..m * k).map(|_| lcg(&mut st)).collect();
            let mut dst_base = vec![0f32; m * n];
            let mut dst_i8mm = vec![0f32; m * n];
            matmul::<BlockQ4K>((m, k, n), &lhs, &rhs_t, &mut dst_base).unwrap();
            matmul_q4kx8l_prepacked_i8mm((m, k, n), &lhs, &laneq, &mut dst_i8mm);
            // Relative L2 error over the whole output: robust to q8 noise on small
            // individual entries, while a layout/tiling bug blows it past 1.0.
            let mut err = 0f64;
            let mut sig = 0f64;
            for i in 0..m * n {
                let d = dst_i8mm[i] as f64 - dst_base[i] as f64;
                err += d * d;
                sig += (dst_base[i] as f64) * (dst_base[i] as f64);
            }
            let rel = (err / sig.max(1e-12)).sqrt();
            assert!(rel < 2e-2, "m={m}: relative L2 error {rel} too high");
        }
    }

    // The lane=row prefill driver (interleave-4 laneq weights + q8_Kx4 4x4
    // activations + lane=row SDOT) must match the baseline Q4_K matmul. NOT
    // bit-exact (signed-max vs abs-max q8 quant), so relative L2; covers m not a
    // multiple of 4 (zero-padded remainder). Validates the full driver: tiling,
    // scatter, parallel group partition. dotprod host (incl. M1), so a real check.
    #[cfg(all(target_feature = "neon", target_feature = "dotprod"))]
    #[test]
    fn matmul_q4kx8l_lanerow_matches_baseline() {
        use crate::quantized::k_quants::matmul;

        let k = 512usize; // 2 super-blocks
        let n = 24usize; // 3 groups of 8
        let nb = k / QK_K;
        let mut st = 0xc0ff_ee00_1234_abcdu64;

        let mut rhs_t = vec![BlockQ4K::zeros(); n * nb];
        for r in 0..n {
            let f: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
            BlockQ4K::from_float(&f, &mut rhs_t[r * nb..(r + 1) * nb]);
        }
        let laneq = repack_q4k_weight_laneq4(&rhs_t, n, nb);

        for &m in &[4usize, 7usize, 16usize] {
            let lhs: Vec<f32> = (0..m * k).map(|_| lcg(&mut st)).collect();
            let mut dst_base = vec![0f32; m * n];
            let mut dst_lr = vec![0f32; m * n];
            matmul::<BlockQ4K>((m, k, n), &lhs, &rhs_t, &mut dst_base).unwrap();
            matmul_q4kx8l_lanerow((m, k, n), &lhs, &laneq, &mut dst_lr);
            let mut err = 0f64;
            let mut sig = 0f64;
            for i in 0..m * n {
                let d = dst_lr[i] as f64 - dst_base[i] as f64;
                err += d * d;
                sig += (dst_base[i] as f64) * (dst_base[i] as f64);
            }
            let rel = (err / sig.max(1e-12)).sqrt();
            assert!(rel < 2e-2, "m={m}: relative L2 error {rel} too high");
        }
    }

    // The parallel activation-quant (par_chunks_mut over row-tiles) must produce
    // BYTE-IDENTICAL Q8 to the serial loop - each tile is independent, so this is
    // structural, but assert it so a future refactor can't silently break the
    // CANDLE_PREFILL_PARQUANT=0/1 equivalence the bench A/B relies on.
    #[test]
    fn parquant_matches_serial() {
        let nb = 4usize;
        let k = nb * QK_K;
        let m = 13usize; // not a multiple of 4 -> exercises the zero-pad tile
        let row_tiles = m.div_ceil(4);
        let mut st = 0x3141_5926_5358_9793u64;
        let lhs: Vec<f32> = (0..m * k).map(|_| lcg(&mut st)).collect();
        let zeros = vec![0f32; k];
        let tile_rows = |rt: usize| -> [&[f32]; 4] {
            std::array::from_fn(|a| {
                let r = rt * 4 + a;
                if r < m {
                    &lhs[r * k..(r + 1) * k]
                } else {
                    zeros.as_slice()
                }
            })
        };
        for use_4x8 in [false, true] {
            let mut serial = vec![BlockQ8Kx4::zeroed(); row_tiles * nb];
            for rt in 0..row_tiles {
                let dst = &mut serial[rt * nb..(rt + 1) * nb];
                if use_4x8 {
                    quantize_mat_q8_k_4x8_into(&tile_rows(rt), dst);
                } else {
                    quantize_mat_q8_k_4x4_into(&tile_rows(rt), dst);
                }
            }
            let mut par = vec![BlockQ8Kx4::zeroed(); row_tiles * nb];
            crate::utils::par_chunks_mut(&mut par, nb, |rt, chunk| {
                if use_4x8 {
                    quantize_mat_q8_k_4x8_into(&tile_rows(rt), chunk);
                } else {
                    quantize_mat_q8_k_4x4_into(&tile_rows(rt), chunk);
                }
            });
            for t in 0..row_tiles * nb {
                assert_eq!(serial[t].d, par[t].d, "d mismatch tile {t} 4x8={use_4x8}");
                assert_eq!(serial[t].qs, par[t].qs, "qs mismatch tile {t} 4x8={use_4x8}");
                assert_eq!(
                    serial[t].bsums, par[t].bsums,
                    "bsums mismatch tile {t} 4x8={use_4x8}"
                );
            }
        }
    }

    // A pre-packed Q4_Kx8 loaded as raw bytes (PackedQ4Kx8::from_bytes) must drive
    // a matmul bit-identical to the standard Q4_K matmul over the original weight,
    // for decode (m=1) and prefill (m=4). This exercises the offline-bake path:
    // repack -> bytes -> PackedQ4Kx8 -> QuantizedType::matmul_t.
    #[cfg(target_feature = "neon")]
    #[test]
    fn packed_q4kx8_from_bytes_matches_baseline() {
        use crate::quantized::k_quants::matmul;
        use crate::quantized::QuantizedType;

        let k = 512usize; // 2 super-blocks
        let n = 24usize; // 3 groups of 8
        let nb = k / QK_K;
        let mut st = 0x0bad_f00d_1234_5678u64;

        // Row-major Q4_K weight: channel r at r*nb.
        let mut rhs_t = vec![BlockQ4K::zeros(); n * nb];
        for r in 0..n {
            let f: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
            BlockQ4K::from_float(&f, &mut rhs_t[r * nb..(r + 1) * nb]);
        }

        // Offline bake: repack -> raw bytes -> PackedQ4Kx8 (owned, from bytes).
        let packed = repack_q4k_weight(&rhs_t, n, nb);
        let raw: &[u8] = unsafe {
            std::slice::from_raw_parts(
                packed.as_ptr() as *const u8,
                std::mem::size_of_val(packed.as_slice()),
            )
        };
        let pq = PackedQ4Kx8::from_bytes(raw, n);

        for &m in &[1usize, 4usize] {
            let lhs: Vec<f32> = (0..m * k).map(|_| lcg(&mut st)).collect();
            let mut dst_base = vec![0f32; m * n];
            let mut dst_pack = vec![0f32; m * n];
            matmul::<BlockQ4K>((m, k, n), &lhs, &rhs_t, &mut dst_base).unwrap();
            pq.matmul_t((m, k, n), &lhs, &mut dst_pack).unwrap();
            let mut maxdiff_bits = 0u32;
            for i in 0..m * n {
                let db = dst_base[i].to_bits() as i64;
                let dp = dst_pack[i].to_bits() as i64;
                maxdiff_bits = maxdiff_bits.max((db - dp).unsigned_abs() as u32);
                assert_eq!(
                    dst_base[i].to_bits(),
                    dst_pack[i].to_bits(),
                    "m={m} idx {i}: base {} packed {}",
                    dst_base[i],
                    dst_pack[i]
                );
            }
            assert_eq!(maxdiff_bits, 0, "m={m} not bit-exact");
        }

        // dequantize() must reproduce the original Q4_K dequant (same formula).
        let mut want = vec![0f32; n * k];
        BlockQ4K::to_float(&rhs_t, &mut want);
        let got = match pq.dequantize(n * k).unwrap() {
            crate::CpuStorage::F32(v) => v,
            _ => panic!("expected f32"),
        };
        for i in 0..n * k {
            assert_eq!(want[i].to_bits(), got[i].to_bits(), "dequant idx {i}");
        }
    }

    #[cfg(target_feature = "neon")]
    #[test]
    fn packed_q6kx8_from_bytes_matches_baseline() {
        use crate::quantized::k_quants::matmul;
        use crate::quantized::QuantizedType;

        let k = 512usize; // 2 super-blocks
        let n = 24usize; // 3 groups of 8
        let nb = k / QK_K;
        let mut st = 0x51c6_ba9d_2244_8821u64;

        // Row-major Q6_K weight: channel r at r*nb.
        let mut rhs_t = vec![BlockQ6K::zeros(); n * nb];
        for r in 0..n {
            let f: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
            BlockQ6K::from_float(&f, &mut rhs_t[r * nb..(r + 1) * nb]);
        }

        // Offline bake: repack -> raw bytes -> PackedQ6Kx8 (owned, from bytes).
        let packed = repack_q6k_weight(&rhs_t, n, nb);
        let raw: &[u8] = unsafe {
            std::slice::from_raw_parts(
                packed.as_ptr() as *const u8,
                std::mem::size_of_val(packed.as_slice()),
            )
        };
        let pq = PackedQ6Kx8::from_bytes(raw, n);

        for &m in &[1usize, 4usize] {
            let lhs: Vec<f32> = (0..m * k).map(|_| lcg(&mut st)).collect();
            let mut dst_base = vec![0f32; m * n];
            let mut dst_pack = vec![0f32; m * n];
            matmul::<BlockQ6K>((m, k, n), &lhs, &rhs_t, &mut dst_base).unwrap();
            pq.matmul_t((m, k, n), &lhs, &mut dst_pack).unwrap();
            for i in 0..m * n {
                assert_eq!(
                    dst_base[i].to_bits(),
                    dst_pack[i].to_bits(),
                    "m={m} idx {i}: base {} packed {}",
                    dst_base[i],
                    dst_pack[i]
                );
            }
        }

        // dequantize() must reproduce the original Q6_K dequant (same formula).
        let mut want = vec![0f32; n * k];
        BlockQ6K::to_float(&rhs_t, &mut want);
        let got = match pq.dequantize(n * k).unwrap() {
            crate::CpuStorage::F32(v) => v,
            _ => panic!("expected f32"),
        };
        for i in 0..n * k {
            assert_eq!(want[i].to_bits(), got[i].to_bits(), "dequant idx {i}");
        }
    }

    // Standalone microbench: baseline per-column matmul vs the packed 8-channel
    // SDOT GEMM, for decode (m=1) and prefill (m=512). Ignored by default; run with
    //   RUSTFLAGS="-C target-cpu=native" cargo test -p candle-core --release \
    //     q4kx8_microbench -- --ignored --nocapture
    // Pin cores with taskset + CANDLE_QMATMUL_PREFILL_THREADS to emulate a tier.
    #[cfg(target_feature = "neon")]
    #[test]
    #[ignore]
    // The tile-height sweep macro expands `r + $main <= m` for $main == 1.
    #[allow(clippy::int_plus_one)]
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
        let rnd = |s: &mut u64| {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            2.0 * ((*s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };

        // Weights: n output channels x k, quantized. Baseline rhs_t is row-major
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
                    // GFLOP/s for the matmul: 2*m*k*n MACs.
                    let gflop = 2.0 * (m as f64) * (k as f64) * (n as f64) / 1e9;
                    println!(
                        "[{phase}] m={m:<4} MR={:<1} base={:.3}ms ({:.1} GFLOP/s) packed={:.3}ms ({:.1} GFLOP/s) speedup={:.2}x maxdiff={maxdiff:.1e}",
                        $main, t_base * 1e3, gflop / t_base, t_pack * 1e3, gflop / t_pack, t_base / t_pack
                    );
                }};
            }
            if m == 1 {
                bench_packed!(1);
            } else {
                bench_packed!(2);
                bench_packed!(4);
                bench_packed!(6);
                bench_packed!(8);
            }
        }
    }

    // The dup-load interleaved-SDOT prefill kernel must match the shipped SDOT
    // packed kernel: same Q8K activations, same Q4_K math, only f32 reassociation
    // differs - so a tiny relative L2 error. Catches any layout / scale / reduce bug.
    #[cfg(all(target_feature = "neon", target_feature = "dotprod"))]
    #[test]
    fn gemm_q4kx8_dup_matches_sdot() {
        use crate::quantized::neon::{gemm_q4kx8_q8k, gemm_q4kx8_q8k_dup};

        let nb = 3usize;
        let k = nb * QK_K;
        let mut st = 0xfeed_face_cafe_d00du64;
        let mut wq: Vec<Vec<BlockQ4K>> = Vec::new();
        for _ in 0..8 {
            let f: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
            let mut q = vec![BlockQ4K::zeros(); nb];
            BlockQ4K::from_float(&f, &mut q);
            wq.push(q);
        }
        let cols: [&[BlockQ4K]; 8] = std::array::from_fn(|c| wq[c].as_slice());
        let laneq = repack_q4kx8_laneq(&cols);
        let sdot = repack_q4k_x8(&cols);

        const MR: usize = 4;
        let q8r: Vec<Vec<BlockQ8K>> = (0..MR)
            .map(|_| {
                let f: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
                let mut q = vec![BlockQ8K::zeros(); nb];
                BlockQ8K::from_float(&f, &mut q);
                q
            })
            .collect();
        let rows: [&[BlockQ8K]; MR] = std::array::from_fn(|r| q8r[r].as_slice());

        let mut d_dup = [0f32; 8 * MR];
        let mut d_sdot = [0f32; 8 * MR];
        gemm_q4kx8_q8k_dup::<MR>(&laneq, &rows, &mut d_dup);
        gemm_q4kx8_q8k::<MR>(&sdot, &rows, &mut d_sdot);

        let mut err = 0f64;
        let mut sig = 0f64;
        for i in 0..8 * MR {
            let d = (d_dup[i] - d_sdot[i]) as f64;
            err += d * d;
            sig += (d_sdot[i] as f64) * (d_sdot[i] as f64);
        }
        let rel = (err / sig.max(1e-12)).sqrt();
        assert!(rel < 1e-4, "dup vs sdot rel L2 {rel} (d_dup {d_dup:?})");
    }

    // Single-thread kernel A/B: dup-load interleaved SDOT vs the shipped SDOT
    // prefill kernel, same MR=4 tiling over the same packed weights / Q8K
    // activations, so the time difference is purely the inner loop. Run with:
    //   cargo test -p candle-core --release dup_vs_sdot_microbench \
    //     -- --ignored --nocapture
    // (build +dotprod, e.g. RUSTFLAGS="-C target-cpu=native"). M1 is not the
    // verdict for this kernel - the column reuse targets N1's narrow core.
    #[cfg(all(target_feature = "neon", target_feature = "dotprod"))]
    #[test]
    #[ignore]
    fn dup_vs_sdot_microbench() {
        use crate::quantized::neon::{gemm_q4kx8_q8k, gemm_q4kx8_q8k_dup};
        use std::time::Instant;

        let (n, k, m) = (2048usize, 2048usize, 512usize);
        let nb = k / QK_K;
        let reps = 8usize;
        let mut st = 0x0123_4567_89ab_cdefu64;

        let mut wq: Vec<Vec<BlockQ4K>> = Vec::new();
        for _ in 0..n {
            let f: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
            let mut q = vec![BlockQ4K::zeros(); nb];
            BlockQ4K::from_float(&f, &mut q);
            wq.push(q);
        }
        // Both packs, group-major (group g at g*nb).
        let mut p_sdot: Vec<BlockQ4Kx8> = Vec::with_capacity((n / 8) * nb);
        let mut p_lane: Vec<BlockQ4Kx8L> = Vec::with_capacity((n / 8) * nb);
        for g in 0..n / 8 {
            let grp: [&[BlockQ4K]; 8] = std::array::from_fn(|r| wq[g * 8 + r].as_slice());
            p_sdot.extend(repack_q4k_x8(&grp));
            p_lane.extend(repack_q4kx8_laneq(&grp));
        }
        // m activation rows -> Q8K.
        let q8r: Vec<Vec<BlockQ8K>> = (0..m)
            .map(|_| {
                let f: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
                let mut q = vec![BlockQ8K::zeros(); nb];
                BlockQ8K::from_float(&f, &mut q);
                q
            })
            .collect();

        let mut out_s = vec![0f32; m * n];
        let mut out_l = vec![0f32; m * n];
        let gflop = 2.0 * (m as f64) * (k as f64) * (n as f64) / 1e9;

        // Sweep the row-tile height. MR=2 is the N1 sweet spot for the shipped
        // kernel (MR=4 spills); the dup kernel may differ, so measure both.
        macro_rules! sweep {
            ($mr:literal) => {{
                const MR: usize = $mr;
                let run = |dup: bool, out: &mut [f32]| {
                    for g in 0..n / 8 {
                        let mut r = 0;
                        while r + MR <= m {
                            let rows: [&[BlockQ8K]; MR] =
                                std::array::from_fn(|a| q8r[r + a].as_slice());
                            let mut t = [0f32; 8 * MR];
                            if dup {
                                gemm_q4kx8_q8k_dup::<MR>(&p_lane[g * nb..(g + 1) * nb], &rows, &mut t);
                            } else {
                                gemm_q4kx8_q8k::<MR>(&p_sdot[g * nb..(g + 1) * nb], &rows, &mut t);
                            }
                            for c in 0..8 {
                                for a in 0..MR {
                                    out[(r + a) * n + g * 8 + c] = t[c * MR + a];
                                }
                            }
                            r += MR;
                        }
                    }
                };
                let mut t_s = f64::MAX;
                let mut t_l = f64::MAX;
                for _ in 0..reps {
                    let t = Instant::now();
                    run(false, &mut out_s);
                    t_s = t_s.min(t.elapsed().as_secs_f64());
                }
                for _ in 0..reps {
                    let t = Instant::now();
                    run(true, &mut out_l);
                    t_l = t_l.min(t.elapsed().as_secs_f64());
                }
                let mut maxdiff = 0f32;
                for i in (0..m * n).step_by(((m * n) / 997).max(1)) {
                    maxdiff = maxdiff.max((out_s[i] - out_l[i]).abs());
                }
                println!(
                    "[prefill m={m} MR={MR}] sdot={:.3}ms ({:.1} GFLOP/s)  dup={:.3}ms ({:.1} GFLOP/s)  dup/sdot={:.2}x  maxdiff={maxdiff:.1e}",
                    t_s * 1e3,
                    gflop / t_s,
                    t_l * 1e3,
                    gflop / t_l,
                    t_s / t_l
                );
            }};
        }
        sweep!(2);
        sweep!(4);
    }

    // Single-thread kernel A/B: llama's lane=row 8x4 GEMM vs the shipped SDOT
    // prefill kernel (nc4mr4-style 4x4 tiling), same n/8 groups over the same
    // weights, MR=4 row tiles. The lane=row kernel processes a fixed 4-row x 8-col
    // tile per call (its structural advantage: 8-col reuse at 16 live int accs).
    //   cargo test -p candle-core --release lanerow_vs_sdot_microbench \
    //     -- --ignored --nocapture
    // (build +dotprod, e.g. RUSTFLAGS="-C target-cpu=native"). M1 is NOT the
    // verdict - the per-sub-block fold targets N1's narrow, register-starved core.
    #[cfg(all(target_feature = "neon", target_feature = "dotprod"))]
    #[test]
    #[ignore]
    fn lanerow_vs_sdot_microbench() {
        use crate::quantized::neon::{gemm_q4kx8_q8k, gemm_q4kx8_q8k_lanerow};
        use std::time::Instant;

        let (n, k, m) = (2048usize, 2048usize, 512usize);
        let nb = k / QK_K;
        let reps = 8usize;
        let mut st = 0x5151_2323_8787_abcdu64;

        let mut wq: Vec<Vec<BlockQ4K>> = Vec::new();
        for _ in 0..n {
            let f: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
            let mut q = vec![BlockQ4K::zeros(); nb];
            BlockQ4K::from_float(&f, &mut q);
            wq.push(q);
        }
        // SDOT pack (BlockQ4Kx8) + lane=row pack (BlockQ4Kx8L interleave 4), per group.
        let mut p_sdot: Vec<BlockQ4Kx8> = Vec::with_capacity((n / 8) * nb);
        let mut p_lr: Vec<BlockQ4Kx8L> = Vec::with_capacity((n / 8) * nb);
        for g in 0..n / 8 {
            let grp: [&[BlockQ4K]; 8] = std::array::from_fn(|r| wq[g * 8 + r].as_slice());
            p_sdot.extend(repack_q4k_x8(&grp));
            p_lr.extend(repack_q4kx8_laneq4(&grp));
        }
        // m activation rows: BlockQ8K (SDOT) and BlockQ8Kx4 (lane=row, 4-row tiles).
        let af: Vec<Vec<f32>> = (0..m).map(|_| (0..k).map(|_| lcg(&mut st)).collect()).collect();
        let q8_sdot: Vec<Vec<BlockQ8K>> = af
            .iter()
            .map(|f| {
                let mut q = vec![BlockQ8K::zeros(); nb];
                BlockQ8K::from_float(f, &mut q);
                q
            })
            .collect();
        let mut q8_lr: Vec<BlockQ8Kx4> = Vec::with_capacity((m / 4) * nb);
        for rt in 0..m / 4 {
            let rows: [&[f32]; 4] = std::array::from_fn(|a| af[rt * 4 + a].as_slice());
            q8_lr.extend(quantize_mat_q8_k_4x4(&rows, nb));
        }

        let mut out_s = vec![0f32; m * n];
        let mut out_l = vec![0f32; m * n];

        // Shipped SDOT path: MR=4 row tiles, nc4mr4 = two 4-channel halves.
        let run_sdot = |out: &mut [f32]| {
            for g in 0..n / 8 {
                let w = &p_sdot[g * nb..(g + 1) * nb];
                let mut r = 0;
                while r + 4 <= m {
                    let rows: [&[BlockQ8K]; 4] = std::array::from_fn(|a| q8_sdot[r + a].as_slice());
                    let mut t = [0f32; 8 * 4];
                    gemm_q4kx8_q8k::<4>(w, &rows, &mut t);
                    for c in 0..8 {
                        for a in 0..4 {
                            out[(r + a) * n + g * 8 + c] = t[c * 4 + a];
                        }
                    }
                    r += 4;
                }
            }
        };
        // Lane=row path: one 4x8 tile per (group, row-tile).
        let run_lr = |out: &mut [f32]| {
            for g in 0..n / 8 {
                let w = &p_lr[g * nb..(g + 1) * nb];
                for rt in 0..m / 4 {
                    let q8t = &q8_lr[rt * nb..(rt + 1) * nb];
                    let mut t = [0f32; 32];
                    gemm_q4kx8_q8k_lanerow(w, q8t, &mut t);
                    for a in 0..4 {
                        for c in 0..8 {
                            out[(rt * 4 + a) * n + g * 8 + c] = t[a * 8 + c];
                        }
                    }
                }
            }
        };

        let mut t_s = f64::MAX;
        let mut t_l = f64::MAX;
        for _ in 0..reps {
            let t = Instant::now();
            run_sdot(&mut out_s);
            t_s = t_s.min(t.elapsed().as_secs_f64());
        }
        for _ in 0..reps {
            let t = Instant::now();
            run_lr(&mut out_l);
            t_l = t_l.min(t.elapsed().as_secs_f64());
        }

        let gflop = 2.0 * (m as f64) * (k as f64) * (n as f64) / 1e9;
        // Both correct but different q8 rounding (abs-max vs signed-max) -> compare
        // by relative L2, not bit-exact.
        let mut err = 0f64;
        let mut sig = 0f64;
        for i in 0..m * n {
            let d = (out_l[i] - out_s[i]) as f64;
            err += d * d;
            sig += (out_s[i] as f64) * (out_s[i] as f64);
        }
        let rel = (err / sig.max(1e-12)).sqrt();
        println!(
            "[prefill m={m}] sdot(nc4mr4)={:.3}ms ({:.1} GFLOP/s)  lanerow(8x4)={:.3}ms ({:.1} GFLOP/s)  lanerow/sdot={:.2}x  relL2={rel:.1e}",
            t_s * 1e3,
            gflop / t_s,
            t_l * 1e3,
            gflop / t_l,
            t_s / t_l
        );
    }
}
