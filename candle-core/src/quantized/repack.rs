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

use super::k_quants::{BlockQ4K, QK_K};
// The packed GEMM kernels (and the trait helpers / rayon they pull in) are
// aarch64/neon-only; the imports they need would be unused on other targets.
#[cfg(target_feature = "neon")]
use super::k_quants::{BlockQ8K, GgmlType};
use byteorder::{ByteOrder, LittleEndian};
use half::f16;
#[cfg(target_feature = "neon")]
use rayon::prelude::*;
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

/// Port of llama's `make_block_q4_Kx8` for one super-block of 8 columns
/// (`blck_size_interleave = 8`). Pure data rearrangement; deterministic.
fn make_q4kx8_laneq(cols: &[&[BlockQ4K]; 8], i: usize) -> BlockQ4Kx8L {
    let mut out = BlockQ4Kx8L::zeroed();
    for c in 0..8 {
        out.d[c] = cols[c][i].d;
        out.dmin[c] = cols[c][i].dmin;
    }
    // qs: take 8 bytes at a time, round-robin across the 8 columns.
    // end = QK_K * 4 / 8 = 128; src col = t%8, src off = (t/8)*8, dst = t*8.
    let end = QK_K * 4 / 8;
    for t in 0..end {
        let src_id = t % 8;
        let src_off = (t / 8) * 8;
        let dst_off = t * 8;
        out.qs[dst_off..dst_off + 8].copy_from_slice(&cols[src_id][i].qs[src_off..src_off + 8]);
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

/// Repack 8 weight columns (each `nb` Q4_K super-blocks) into `nb` laneq blocks.
pub(crate) fn repack_q4kx8_laneq(cols: &[&[BlockQ4K]; 8]) -> Vec<BlockQ4Kx8L> {
    let nb = cols[0].len();
    (0..nb).map(|i| make_q4kx8_laneq(cols, i)).collect()
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
/// Prefill tile strategy for the packed GEMM. `nc4mr4` (DEFAULT): two 4-channel x
/// 4-row calls per group (16 acc each, most activation reuse without spilling -
/// measured best on N1). `nc8mr2`: one 8-channel x 2-row call (16 acc). Override
/// with `CANDLE_PACKED_PREFILL=nc8mr2`.
static PACKED_PREFILL_NC4: LazyLock<bool> = LazyLock::new(|| {
    !std::env::var("CANDLE_PACKED_PREFILL")
        .map(|s| s.eq_ignore_ascii_case("nc8mr2"))
        .unwrap_or(false)
});

#[cfg(target_feature = "neon")]
pub(crate) fn matmul_q4k_packed(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs_q4k: &[BlockQ4K],
    dst: &mut [f32],
    pool: &rayon::ThreadPool,
) {
    let nb = k / QK_K;
    let packed = packed_cache_get(rhs_q4k, n, nb);
    matmul_q4kx8_prepacked((m, k, n), lhs, &packed, dst, pool);
}

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
    pool: &rayon::ThreadPool,
) {
    use super::neon::{gemm_q4kx8_q8k, gemm_q4kx_q8k};
    let nb = k / QK_K;
    let groups = n / Q4KX8_ROWS;
    let use_nc4 = *PACKED_PREFILL_NC4;
    // Single-thread decode (the Lambda 1-vCPU tier): pool.install + into_par_iter is
    // pure crossbeam split/join overhead with no parallelism to gain (~3% of decode),
    // so run serially - mirrors the m==1 serial GEMV fast-path in k_quants::matmul.
    let serial = pool.current_num_threads() <= 1;

    let mut lhs_q = vec![BlockQ8K::zeros(); m * nb];
    if serial {
        for a in 0..m {
            BlockQ8K::from_float(&lhs[a * k..(a + 1) * k], &mut lhs_q[a * nb..(a + 1) * nb]);
        }
    } else {
        pool.install(|| {
            lhs_q
                .par_chunks_mut(nb)
                .enumerate()
                .with_min_len(4)
                .for_each(|(a, row)| BlockQ8K::from_float(&lhs[a * k..(a + 1) * k], row));
        });
    }

    struct DstPtr(*mut f32);
    unsafe impl Sync for DstPtr {}
    let dptr = DstPtr(dst.as_mut_ptr());
    let process = |g: usize| {
        let w = &packed[g * nb..(g + 1) * nb];
        let p = &dptr;
        let row = |r: usize| -> &[BlockQ8K] { &lhs_q[r * nb..(r + 1) * nb] };
        let mut r = 0;
        if use_nc4 {
            // nc4mr4: 4 rows x {channels 0..4, 4..8} per call.
            while r + 4 <= m {
                let rows: [&[BlockQ8K]; 4] = std::array::from_fn(|a| row(r + a));
                for half in 0..2 {
                    let c0 = half * 4;
                    let mut tile = [0f32; 4 * 4];
                    gemm_q4kx_q8k::<4, 4>(w, c0, &rows, &mut tile);
                    for c in 0..4 {
                        for a in 0..4 {
                            unsafe { *p.0.add((r + a) * n + g * 8 + c0 + c) = tile[c * 4 + a] };
                        }
                    }
                }
                r += 4;
            }
        }
        // nc8mr2 main (default), or the 2/3-row remainder of the nc4 path.
        while r + 2 <= m {
            let rows: [&[BlockQ8K]; 2] = std::array::from_fn(|a| row(r + a));
            let mut tile = [0f32; Q4KX8_ROWS * 2];
            gemm_q4kx8_q8k::<2>(w, &rows, &mut tile);
            for c in 0..Q4KX8_ROWS {
                for a in 0..2 {
                    unsafe { *p.0.add((r + a) * n + g * 8 + c) = tile[c * 2 + a] };
                }
            }
            r += 2;
        }
        while r < m {
            let rows: [&[BlockQ8K]; 1] = [row(r)];
            let mut tile = [0f32; Q4KX8_ROWS];
            gemm_q4kx8_q8k::<1>(w, &rows, &mut tile);
            for c in 0..Q4KX8_ROWS {
                unsafe { *p.0.add(r * n + g * 8 + c) = tile[c] };
            }
            r += 1;
        }
    };

    if serial {
        (0..groups).for_each(&process);
    } else {
        pool.install(|| {
            (0..groups).into_par_iter().for_each(&process);
        });
    }
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
        let (m, _k, _n) = mkn;
        let pool = if m == 1 {
            &*super::k_quants::QMATMUL_DECODE_POOL
        } else {
            &*super::k_quants::QMATMUL_PREFILL_POOL
        };
        matmul_q4kx8_prepacked(mkn, lhs, self.as_slice(), dst, pool);
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

    // The packed driver must match baseline matmul end-to-end (full dst, multiple
    // 8-channel groups), for prefill and decode.
    #[cfg(target_feature = "neon")]
    #[test]
    fn matmul_q4k_packed_matches_baseline() {
        use crate::quantized::k_quants::matmul;

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
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();

        for &m in &[1usize, 5usize, 16usize] {
            let lhs: Vec<f32> = (0..m * k).map(|_| lcg(&mut st)).collect();
            let mut dst_base = vec![0f32; m * n];
            let mut dst_pack = vec![0f32; m * n];
            matmul::<BlockQ4K>((m, k, n), &lhs, &rhs_t, &mut dst_base).unwrap();
            clear_packed_cache();
            matmul_q4k_packed((m, k, n), &lhs, rhs_q4k, &mut dst_pack, &pool);
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
            }
        }
    }
}
