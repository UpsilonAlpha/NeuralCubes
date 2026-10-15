//! Rough step-time benchmark: `cargo run --release -p nc-core --example bench [rule] [size]`.

use nc_core::{from_notation, Dims, Seeding};
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let rule = args.next().unwrap_or_else(|| "4/4/5/M".into());
    let size: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let mut sim = from_notation(&rule, Dims::cube(size)).expect("valid rule");
    sim.seed(Seeding { extent: 0.5, density: 0.5, seed: 1 });
    let steps = 50;
    let t = Instant::now();
    for _ in 0..steps {
        sim.step();
    }
    let per = t.elapsed().as_secs_f64() / steps as f64;
    println!(
        "{} ({}) on {size}^3: {:.2} ms/step, {:.0} Mcells/s, population {}",
        sim.notation(),
        sim.family(),
        per * 1e3,
        (size * size * size) as f64 / per / 1e6,
        sim.population()
    );
}
