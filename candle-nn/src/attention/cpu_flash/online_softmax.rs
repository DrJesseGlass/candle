//! Online softmax step shared by CPU flash-attention kernels.

// NB: decode runs this ONE score at a time (an online reduction), so its exp can't vectorize.
// A scalar poly-exp was tried (CANDLE_VEC_SOFTMAX_EXP) and REVERTED here: it cut transcendental
// instructions but ran slower - its Horner FMA dependency chain has worse latency than libm expf,
// and decode is memory-bound anyway (38.7% backend stall) so instruction cuts don't help. libm
// `.exp()` stays. (The vectorized poly is kept for PREFILL attention in causal.rs, where it
// vectorizes across the score vector with no serial chain.)

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
