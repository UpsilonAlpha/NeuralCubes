//! Grows a neural CA from its seed, damages it, and reports how closely it matches a target.
//! `cargo run --release -p nc-core --example nca_eval -- web/models/bunny.json target.alpha`
//! where `target.alpha` is one byte per cell (z, y, x order), nonzero inside the shape.

use nc_core::{from_nca_json, Dims, Seeding};

fn main() {
    let mut args = std::env::args().skip(1);
    let model = std::fs::read_to_string(args.next().expect("weights path")).unwrap();
    let target: Vec<bool> = std::fs::read(args.next().expect("target alpha path"))
        .unwrap()
        .iter()
        .map(|&b| b > 0)
        .collect();
    let n = (target.len() as f64).cbrt().round() as usize;
    let mut sim = from_nca_json(&model, Dims::cube(n)).unwrap();
    sim.seed(Seeding::default());
    let iou = |sim: &dyn nc_core::Automaton| {
        let d = sim.dims();
        let mut cells = vec![false; d.len()];
        // A ray starting inside a cell hits that cell exactly when it is filled.
        for z in 0..d.z {
            for y in 0..d.y {
                for x in 0..d.x {
                    let i = d.index(x, y, z);
                    cells[i] = sim
                        .raycast(
                            [x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5],
                            [1.0, 0.0, 0.0],
                        )
                        .is_some_and(|h| h.cell == [x, y, z]);
                }
            }
        }
        let inter = cells
            .iter()
            .zip(&target)
            .filter(|(a, b)| **a && **b)
            .count();
        let union = cells
            .iter()
            .zip(&target)
            .filter(|(a, b)| **a || **b)
            .count();
        inter as f64 / union.max(1) as f64
    };
    for step in 1..=200 {
        sim.step();
        if [24, 48, 72, 100, 150, 200].contains(&step) {
            println!(
                "grow step {step}: IoU {:.3}, population {}",
                iou(&*sim),
                sim.population()
            );
        }
    }
    let removed = sim.erase(
        [n as f32 * 0.55, n as f32 * 0.6, n as f32 * 0.5],
        n as f32 * 0.25,
    );
    println!("damaged: removed {removed} cells, IoU {:.3}", iou(&*sim));
    for step in 1..=120 {
        sim.step();
        if [20, 40, 80, 120].contains(&step) {
            println!("regrow step {step}: IoU {:.3}", iou(&*sim));
        }
    }
}
