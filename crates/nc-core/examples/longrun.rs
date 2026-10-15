//! Grows a neural CA for many steps, carves a ball out halfway, and reports how far the
//! shape drifts from its form at step 150: `longrun <weights.json> [steps]`.
use nc_core::{from_nca_json, Dims, Seeding};

fn alive(s: &[f32], c: usize) -> Vec<bool> {
    s.chunks(c).map(|v| v[3] >= 0.2).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let json = std::fs::read_to_string(&args[1]).unwrap();
    let steps: u64 = args.get(2).map_or(3000, |s| s.parse().unwrap());
    let c: usize = serde_json::from_str::<serde_json::Value>(&json).unwrap()["channels"]
        .as_u64()
        .unwrap() as usize;
    let size = 24;
    {
        let mut sim = from_nca_json(&json, Dims::cube(size)).unwrap();
        sim.seed(Seeding::default());
        let mut reference = Vec::new();
        let mut line = String::new();
        for t in 1..=steps {
            sim.step();
            if t == 150 {
                reference = alive(&sim.export_state(), c);
            }
            if t == steps / 2 {
                // Carve a ball out of the upper body to check it still regrows.
                let (mut m, mut k) = ([0.0f32; 3], 0.0);
                for (i, _) in reference.iter().enumerate().filter(|(_, a)| **a) {
                    let p = [i % size, (i / size) % size, i / (size * size)];
                    for a in 0..3 {
                        m[a] += p[a] as f32 + 0.5;
                    }
                    k += 1.0;
                }
                let m = m.map(|v| v / k);
                let removed = sim.erase([m[0], m[1] + 3.0, m[2]], 4.0);
                line += &format!("  [cut {removed}]");
            }
            if t >= 150 && (t == 150 || t % (steps / 5) == 0) {
                let a = alive(&sim.export_state(), c);
                let inter = a.iter().zip(&reference).filter(|(x, y)| **x && **y).count();
                let uni = a.iter().zip(&reference).filter(|(x, y)| **x || **y).count();
                let n = a.iter().filter(|x| **x).count();
                line += &format!(
                    "  t{t}: {n} cells, IoU {:.2}",
                    inter as f32 / uni.max(1) as f32
                );
            }
        }
        println!("{line}");
    }
}
