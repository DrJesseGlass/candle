// Measure the per-fork-join cost of candle's BarrierPool, to decide whether
// candle's fine-grained per-op parallel regions (~400/prefill) cost meaningful
// SYNC time, or whether multi-thread prefill scaling loss is serial COMPUTATION
// instead. Two regimes:
//   - HOT: back-to-back execute() - workers stay in the spin window (best case).
//   - PARKED: a serial gap between execute() longer than the spin window, so
//     workers park() and the next execute pays a futex unpark (realistic when
//     there's real serial work between barriers).
// Run: CANDLE_NUM_THREADS=4 cargo run --release --example barrier_cost
use candle_core::utils::barrier_pool;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static SINK: AtomicU64 = AtomicU64::new(0);

// Busy-wait ~`us` microseconds so workers exceed the 10k spin window and park.
fn busy_us(us: u64) {
    let t = Instant::now();
    let mut x = 0u64;
    while t.elapsed().as_micros() < us as u128 {
        x = x.wrapping_add(1);
        std::hint::spin_loop();
    }
    SINK.fetch_add(x, Ordering::Relaxed);
}

fn main() {
    let pool = barrier_pool();
    let w = pool.n_workers();
    println!("barrier_pool n_workers = {w} (set CANDLE_NUM_THREADS to change)");

    let n = 200_000usize;
    // tiny payload: each thread bumps a sink so the closure isn't optimized away
    let payload = |tid: usize| {
        SINK.fetch_add(tid as u64 + 1, Ordering::Relaxed);
    };

    // warm up
    for _ in 0..1000 {
        pool.execute(payload);
    }

    // HOT: back-to-back
    let t = Instant::now();
    for _ in 0..n {
        pool.execute(payload);
    }
    let hot = t.elapsed().as_secs_f64() / n as f64 * 1e6;
    println!("HOT    execute(): {hot:.3} us/call  ({n} calls)");

    // PARKED: ~150us serial gap between calls (exceeds spin window -> workers park)
    let gap_us = 150u64;
    let m = 5_000usize;
    let t = Instant::now();
    for _ in 0..m {
        pool.execute(payload);
        busy_us(gap_us);
    }
    let total = t.elapsed().as_secs_f64() / m as f64 * 1e6;
    let parked = total - gap_us as f64;
    println!("PARKED execute(): {parked:.3} us/call  (total {total:.2} us incl {gap_us}us gap, {m} calls)");

    // Put it in context: ~400 barriers/prefill.
    let bpp = 400.0;
    println!(
        "\n~{bpp:.0} barriers/prefill => hot {:.2} ms,  parked {:.2} ms of pure sync",
        bpp * hot / 1e3,
        bpp * parked / 1e3
    );
    println!("sink={}", SINK.load(Ordering::Relaxed));
}
