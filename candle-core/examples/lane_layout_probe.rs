// Probe: does a lane=CHANNEL SDOT GEMM beat candle's lane=SUB-DOT layout?
//
// candle's packed kernel (gemm_q4kx_q8k) keeps one int32x4 accumulator per
// (channel,row) whose 4 lanes are SDOT's internal partials, then does a vaddvq
// HORIZONTAL REDUCTION per output. llama's 8x4 keeps 4 CHANNELS in the 4 lanes of
// one accumulator (lane=channel), so no per-output reduction and the scale applies
// to 4 channels at once - at the cost of pre-replicating the activation 4x.
//
// This isolates the layout question with a pure int8 GEMM (out[m][n] = sum_k
// w[n][k]*a[m][k]) - the Q4_K nibble/scale work is identical for both layouts, so
// the relative kernel speed here predicts the Q4_K case. 8 channels x 4 rows tile.
//
//   cargo run --release --example lane_layout_probe
use std::arch::aarch64::*;
use std::time::Instant;

#[cfg(target_feature = "neon")]
#[inline(always)]
unsafe fn vdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    // stable-compiler workaround (std vdotq is unstable): emit SDOT directly.
    let mut out = acc;
    std::arch::asm!(
        "sdot {0:v}.4s, {1:v}.16b, {2:v}.16b",
        inout(vreg) out, in(vreg) a, in(vreg) b, options(pure, nomem, nostack)
    );
    out
}

fn fill_i8(n: usize, seed: u64) -> Vec<i8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 56) as i8) >> 1 // small-ish signed
        })
        .collect()
}

fn main() {
    let (n, k, m) = (2048usize, 2048usize, 512usize);
    let reps = 8;
    let w = fill_i8(n * k, 1); // w[c*k + j]
    let a = fill_i8(m * k, 7); // a[r*k + j]
    let macs = 2.0 * (m * n * k) as f64 / 1e9;

    // Reference int32 result for correctness (sampled).
    let refdot = |c: usize, r: usize| -> i32 {
        let mut s = 0i32;
        for j in 0..k {
            s += w[c * k + j] as i32 * a[r * k + j] as i32;
        }
        s
    };

    // ---- Layout A: lane=sub-dot (candle style). 8ch x 4row, vaddvq per output. ----
    let mut out_a = vec![0i32; m * n];
    let mut ta = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        unsafe {
            for n0 in (0..n).step_by(8) {
                for m0 in (0..m).step_by(4) {
                    let mut acc = [[vdupq_n_s32(0); 4]; 8];
                    for k0 in (0..k).step_by(16) {
                        let mut wv = [vdupq_n_s8(0); 8];
                        for c in 0..8 {
                            wv[c] = vld1q_s8(w.as_ptr().add((n0 + c) * k + k0));
                        }
                        for r in 0..4 {
                            let av = vld1q_s8(a.as_ptr().add((m0 + r) * k + k0));
                            for c in 0..8 {
                                acc[c][r] = vdot(acc[c][r], wv[c], av);
                            }
                        }
                    }
                    for c in 0..8 {
                        for r in 0..4 {
                            out_a[(m0 + r) * n + n0 + c] = vaddvq_s32(acc[c][r]);
                        }
                    }
                }
            }
        }
        ta = ta.min(t.elapsed().as_secs_f64());
    }

    // ---- Layout B: lane=channel (llama style). Repack weights to 4ch-interleaved
    // by k-group, pre-replicate activation 4x. 4 lanes = 4 channels, NO vaddvq. ----
    // wp[cg][kg*16..]: for channel group cg (4 ch), k-group kg (4 k): bytes
    //   [c0k0 c0k1 c0k2 c0k3  c1k0..3  c2..  c3..]
    let kg = k / 4;
    let ncg = n / 4;
    let mut wp = vec![0i8; n * k]; // same total size, reinterleaved
    for cg in 0..ncg {
        for g in 0..kg {
            let base = (cg * kg + g) * 16;
            for cc in 0..4 {
                let c = cg * 4 + cc;
                for kk in 0..4 {
                    wp[base + cc * 4 + kk] = w[c * k + g * 4 + kk];
                }
            }
        }
    }
    // ap[r]: a[r][g*4..g*4+4] replicated 4x per k-group -> stride 4k (=kg*16).
    let astride = kg * 16;
    let mut ap = vec![0i8; m * astride];
    for r in 0..m {
        for g in 0..kg {
            let base = r * astride + g * 16;
            for lane in 0..4 {
                for kk in 0..4 {
                    ap[base + lane * 4 + kk] = a[r * k + g * 4 + kk];
                }
            }
        }
    }
    let mut out_b = vec![0i32; m * n];
    let mut tb = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        unsafe {
            for cg in 0..ncg {
                for m0 in (0..m).step_by(4) {
                    let mut acc = [vdupq_n_s32(0); 4]; // one per row, lanes=channels
                    for g in 0..kg {
                        let wv = vld1q_s8(wp.as_ptr().add((cg * kg + g) * 16));
                        for r in 0..4 {
                            let av = vld1q_s8(ap.as_ptr().add((m0 + r) * astride + g * 16));
                            acc[r] = vdot(acc[r], wv, av);
                        }
                    }
                    for r in 0..4 {
                        let o = ((m0 + r) * n + cg * 4) as usize;
                        vst1q_s32(out_b.as_mut_ptr().add(o), acc[r]);
                    }
                }
            }
        }
        tb = tb.min(t.elapsed().as_secs_f64());
    }

    // correctness (sampled)
    let mut bad = 0;
    for &(c, r) in &[(0, 0), (7, 3), (1023, 17), (2047, 511), (500, 200)] {
        let want = refdot(c, r);
        if out_a[r * n + c] != want {
            bad += 1;
            println!("A mismatch c={c} r={r}: {} vs {want}", out_a[r * n + c]);
        }
        if out_b[r * n + c] != want {
            bad += 1;
            println!("B mismatch c={c} r={r}: {} vs {want}", out_b[r * n + c]);
        }
    }
    println!(
        "lane=sub-dot (candle):  {:.3} ms  {:.1} GFLOP/s",
        ta * 1e3,
        macs / ta
    );
    println!(
        "lane=channel (llama):   {:.3} ms  {:.1} GFLOP/s  ({:.2}x)",
        tb * 1e3,
        macs / tb,
        ta / tb
    );
    println!("correctness: {}", if bad == 0 { "OK" } else { "FAILED" });
}
