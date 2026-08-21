use nominal::examples::karpathy;

fn main() {
    // warm-up pass
    karpathy::manual();

    const RUNS: u32 = 1000000;
    let start = std::time::Instant::now();
    for _ in 0..RUNS {
        karpathy::manual();
    }
    let duration = start.elapsed();
    println!("Time elapsed: {:?}", duration);
    println!("Average time per run: {:?}", duration / RUNS);
}
