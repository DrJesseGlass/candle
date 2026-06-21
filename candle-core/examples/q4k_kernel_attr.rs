// Instruction attribution: per-kernel instr/MAC for the Q4_K decode path.
//
//   MODE=vecdot : plain Q4_K vec_dot kernel (nibble-unpack + SDOT + scale).
//   MODE=q6k    : plain Q6_K vec_dot kernel (the lm_head / residual attn_v,ffn_down).
//   MODE=packed : the interleaved Q4Kx8 packed matmul (PackedQ4Kx8::matmul_t).
//   MODE=gemm   : the full plain matmul_t (activation quant + blocking + kernel).
//
// Run each under `perf stat -e instructions` and divide by total_macs for instr/MAC.
//
//   MODE=vecdot|q6k|packed|gemm M=1 ITERS=2000 cargo run --release --example q4k_kernel_attr

use candle_core::quantized::k_quants::{BlockQ4K, BlockQ6K, BlockQ8K, GgmlType, QK_K};
use candle_core::quantized::repack::{repack_q4k_weight, BlockQ4Kx8, PackedQ4Kx8};
use candle_core::quantized::QuantizedType;

fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = (i.wrapping_mul(2654435761).wrapping_add(seed)) % 1000;
            (h as f32) / 500.0 - 1.0
        })
        .collect()
}

fn blocks_as_bytes(blocks: &[BlockQ4Kx8]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(blocks.as_ptr() as *const u8, std::mem::size_of_val(blocks))
    }
}

fn main() {
    let mode = std::env::var("MODE").unwrap_or_else(|_| "vecdot".into());
    let m: usize = std::env::var("M").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    // Qwen3-0.6B ffn_up shape: k=1024 in, n=3072 out.
    let (k, n) = (1024usize, 3072usize);
    let nb = k / QK_K;

    let w = fill(n * k, 1);
    let lhs = fill(m * k, 7);
    // pre-quantized Q8K activation for the kernel modes
    let mut q8k = vec![BlockQ8K::zeros(); m * nb];
    <BlockQ8K as GgmlType>::from_float(&lhs, &mut q8k);
    let mut dst = vec![0f32; m * n];
    let macs_per_iter = (m * n * k) as u64;

    match mode.as_str() {
        "vecdot" => {
            let mut q4k = vec![BlockQ4K::zeros(); n * nb];
            <BlockQ4K as GgmlType>::from_float(&w, &mut q4k);
            for _ in 0..iters {
                for r in 0..m {
                    let a = &q8k[r * nb..(r + 1) * nb];
                    for row in 0..n {
                        dst[r * n + row] =
                            <BlockQ4K as GgmlType>::vec_dot(k, &q4k[row * nb..(row + 1) * nb], a);
                    }
                }
                std::hint::black_box(dst[0]);
            }
        }
        "q6k" => {
            let mut q6k = vec![BlockQ6K::zeros(); n * nb];
            <BlockQ6K as GgmlType>::from_float(&w, &mut q6k);
            for _ in 0..iters {
                for r in 0..m {
                    let a = &q8k[r * nb..(r + 1) * nb];
                    for row in 0..n {
                        dst[r * n + row] =
                            <BlockQ6K as GgmlType>::vec_dot(k, &q6k[row * nb..(row + 1) * nb], a);
                    }
                }
                std::hint::black_box(dst[0]);
            }
        }
        "packed" => {
            let mut q4k = vec![BlockQ4K::zeros(); n * nb];
            <BlockQ4K as GgmlType>::from_float(&w, &mut q4k);
            let packed_blocks = repack_q4k_weight(&q4k, n, nb);
            let packed = PackedQ4Kx8::from_bytes(blocks_as_bytes(&packed_blocks), n);
            for _ in 0..iters {
                packed.matmul_t((m, k, n), &lhs, &mut dst).unwrap();
                std::hint::black_box(dst[0]);
            }
        }
        "gemm" => {
            let mut q4k = vec![BlockQ4K::zeros(); n * nb];
            <BlockQ4K as GgmlType>::from_float(&w, &mut q4k);
            for _ in 0..iters {
                q4k.matmul_t((m, k, n), &lhs, &mut dst).unwrap();
                std::hint::black_box(dst[0]);
            }
        }
        other => panic!("MODE must be vecdot|q6k|packed|gemm, got {other}"),
    }

    println!(
        "mode={mode} m={m} k={k} n={n} iters={iters} macs/iter={macs_per_iter} total_macs={}",
        macs_per_iter * iters as u64
    );
}
