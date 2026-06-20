// Kernel-level multithread-scaling probe for Q4_K prefill GEMM.
//
// Compares two weight kernels on the SAME shapes, activations and thread pool:
//   A) the current unpacked Q4_K `_xr` matmul (A1's row-tiled SDOT kernel), and
//   B) B-core's interleaved Q4_Kx8 packed GEMM (`gemm_q4kx8_q8k`), whose weights
//      stream contiguously with scales/mins pre-unpacked.
//
// Both go through A1's QMATMUL_PREFILL_POOL, so thread count is controlled
// externally by CANDLE_QMATMUL_PREFILL_THREADS. Sweep that knob 1/2/4/8 to see
// which kernel keeps scaling: if the unpacked path plateaus (bandwidth-bound on
// the per-call nibble-unpack + weight stream) while the packed path keeps
// climbing, the multithread gap is the repack, and B-model is the scaling fix.
//
// Run (on the Graviton2 box, native target for NEON+dotprod):
//   for t in 1 2 4 8; do
//     CANDLE_QMATMUL_PREFILL_THREADS=$t M=128 \
//       cargo run --release --example q4k_pack_sweep
//   done
//
// Env knobs: M = prefill rows (default 128), ITERS = timed reps (default 40).

use candle_core::quantized::k_quants::{BlockQ4K, GgmlType, QK_K};
use candle_core::quantized::repack::{repack_q4k_weight, PackedQ4Kx8, BlockQ4Kx8};
use candle_core::quantized::QuantizedType;
use std::time::Instant;

// Deterministic pseudo-random values in [-1, 1] so weights/activations are
// stable across runs without pulling in an rng dependency.
fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = (i.wrapping_mul(2654435761).wrapping_add(seed)) % 1000;
            (h as f32) / 500.0 - 1.0
        })
        .collect()
}

// Byte view of a packed-block slice for PackedQ4Kx8::from_bytes. BlockQ4Kx8 is
// #[repr(C)] POD, so this reinterprets the same bytes the GGUF loader would.
fn blocks_as_bytes(blocks: &[BlockQ4Kx8]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr() as *const u8,
            std::mem::size_of_val(blocks),
        )
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn bench(iters: usize, mut f: impl FnMut() -> f32) -> f64 {
    // one warm-up (also forces lazy pool init) then `iters` timed reps
    std::hint::black_box(f());
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        std::hint::black_box(f());
        samples.push(t.elapsed().as_secs_f64());
    }
    median(samples)
}

fn main() {
    let m: usize = std::env::var("M").ok().and_then(|s| s.parse().ok()).unwrap_or(128);
    let iters: usize = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(40);
    let threads = std::env::var("CANDLE_QMATMUL_PREFILL_THREADS").unwrap_or_else(|_| "default".into());

    // Representative Qwen3-0.6B prefill GEMM shapes: (name, k=in_features, n=out_features).
    let shapes = [
        ("attn_qkvo", 1024, 1024),
        ("ffn_up_gate", 1024, 3072),
        ("ffn_down", 3072, 1024),
    ];

    println!(
        "# m={m} iters={iters} prefill_threads={threads}  (median of {iters} reps)"
    );
    println!(
        "{:<12} {:>6} {:>6} {:>10} {:>10} {:>9} {:>9} {:>9} {:>10}",
        "shape", "k", "n", "xr_us", "pack_us", "xr_GFs", "pk_GFs", "speedup", "maxdiff"
    );

    for (name, k, n) in shapes {
        assert!(k % QK_K == 0, "k must be a multiple of {QK_K}");
        assert!(n % 8 == 0, "n must be a multiple of 8");
        let nb = k / QK_K;

        // Quantize a random [n, k] weight (row-major, n output channels) to Q4_K,
        // then interleave to Q4_Kx8.
        let w = fill(n * k, 1);
        let mut q4k = vec![BlockQ4K::zeros(); n * nb];
        <BlockQ4K as GgmlType>::from_float(&w, &mut q4k);
        let packed_blocks = repack_q4k_weight(&q4k, n, nb);
        let packed = PackedQ4Kx8::from_bytes(blocks_as_bytes(&packed_blocks), n);

        let lhs = fill(m * k, 7);
        let mut dst_xr = vec![0f32; m * n];
        let mut dst_pk = vec![0f32; m * n];

        // Correctness: the packed GEMM is meant to be bit-exact to `_xr`.
        q4k.matmul_t((m, k, n), &lhs, &mut dst_xr).unwrap();
        packed.matmul_t((m, k, n), &lhs, &mut dst_pk).unwrap();
        let maxdiff = dst_xr
            .iter()
            .zip(&dst_pk)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);

        let t_xr = bench(iters, || {
            q4k.matmul_t((m, k, n), &lhs, &mut dst_xr).unwrap();
            dst_xr[0]
        });
        let t_pk = bench(iters, || {
            packed.matmul_t((m, k, n), &lhs, &mut dst_pk).unwrap();
            dst_pk[0]
        });

        let flop = 2.0 * (m * n * k) as f64;
        let gf_xr = flop / t_xr / 1e9;
        let gf_pk = flop / t_pk / 1e9;
        println!(
            "{:<12} {:>6} {:>6} {:>10.1} {:>10.1} {:>9.1} {:>9.1} {:>8.2}x {:>10.2e}",
            name,
            k,
            n,
            t_xr * 1e6,
            t_pk * 1e6,
            gf_xr,
            gf_pk,
            t_xr / t_pk,
            maxdiff,
        );
    }
}
