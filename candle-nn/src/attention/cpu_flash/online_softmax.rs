//! Online softmax step shared by CPU flash-attention kernels.
//!
//! Uses the exact libm `f32::exp` (bit-exact). A polynomial `fast_expf` was
//! trialed (gated `fast-softmax`) but measured ~0% on both M1 (Apple libm already
//! fast) and N1/glibc, so it was dropped. The f16 q·k approximation (`f16-attn-dot`,
//! +~1.4% N1 decode) is the only attention precision tradeoff we keep.

/// Stream one (score, v_row) into (m, ssum, acc); v_apply adds v_row * w to acc.
#[inline(always)]
pub(crate) fn online_softmax_step(
    score: f32,
    m: &mut f32,
    ssum: &mut f32,
    acc: &mut [f32],
    v_apply: impl FnOnce(&mut [f32], f32),
) {
    if score > *m {
        let scale_old = (*m - score).exp();
        for a in acc.iter_mut() {
            *a *= scale_old;
        }
        *ssum = *ssum * scale_old + 1.0;
        *m = score;
        v_apply(acc, 1.0);
    } else {
        let w = (score - *m).exp();
        v_apply(acc, w);
        *ssum += w;
    }
}
