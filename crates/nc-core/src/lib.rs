//! Neural Cubes simulation core.
//!
//! A [`Grid`] holds cells, a [`Rule`] says how each cell changes, and a [`Simulation`]
//! steps one over the other. New behaviour is added by writing a new [`Rule`]
//! implementation (a "rule family"); see [`rules`] for the built-in ones.

pub mod grid;
pub mod neighborhood;
pub mod rng;
pub mod rule;
pub mod rules;
pub mod sim;

pub use grid::{Dims, Grid};
pub use neighborhood::{Neighborhood, Neighbors};
pub use rng::Rng;
pub use rule::{Rule, StepContext};
pub use rules::ParseError;
pub use sim::{Automaton, Hit, Seeding, Simulation};

/// Builds a simulation from rule notation, choosing the rule family from the notation.
///
/// `S/B/2/N` gives a [`rules::LifeLike`] rule, `S/B/C/N` with `C > 2` a [`rules::Generations`] rule.
pub fn from_notation(notation: &str, dims: Dims) -> Result<Box<dyn Automaton>, ParseError> {
    let t = rules::Totalistic::parse(notation)?;
    Ok(if t.states == 2 {
        Box::new(Simulation::new(rules::LifeLike::parse(notation)?, dims))
    } else {
        Box::new(Simulation::new(rules::Generations::parse(notation)?, dims))
    })
}

/// Builds a neural CA simulation from weights exported by `training/train_nca.py`.
pub fn from_nca_json(json: &str, dims: Dims) -> Result<Box<dyn Automaton>, ParseError> {
    let w = rules::NcaWeights::from_json(json)?;
    macro_rules! build {
        ($($c:literal),*) => {
            match w.channels {
                $($c => Box::new(Simulation::new(rules::NeuralCa::<$c>::new(w)?, dims)) as Box<dyn Automaton>,)*
                n => return Err(ParseError(format!("unsupported NCA channel count {n}; supported: {:?}", [$($c),*]))),
            }
        };
    }
    Ok(build!(8, 12, 16, 24, 32))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rules::{CountSet, Generations, LifeLike};

    #[test]
    fn count_sets_parse_and_print() {
        let s = CountSet::parse("5-7, 12,13", 26).unwrap();
        assert!(
            s.contains(5) && s.contains(6) && s.contains(7) && s.contains(12) && s.contains(13)
        );
        assert!(!s.contains(8));
        assert_eq!(s.to_string(), "5-7,12,13");
        assert!(CountSet::parse("27", 26).is_err());
        assert!(CountSet::parse("7", 6).is_err());
        assert!(CountSet::parse("5-3", 26).is_err());
        assert_eq!(CountSet::parse("", 26).unwrap(), CountSet::default());
    }

    #[test]
    fn notation_round_trips_and_picks_family() {
        let a = from_notation("4/4/5/M", Dims::cube(4)).unwrap();
        assert_eq!(
            (a.family(), a.notation().as_str()),
            ("generations", "4/4/5/M")
        );
        let b = from_notation(" 9-26 / 5-7,12-13,15 / 2 / m ", Dims::cube(4)).unwrap();
        assert_eq!(
            (b.family(), b.notation().as_str()),
            ("life-like", "9-26/5-7,12,13,15/2/M")
        );
        assert!(from_notation("4/4/M", Dims::cube(4)).is_err());
        assert!(from_notation("4/4/1/M", Dims::cube(4)).is_err());
        assert!(from_notation("4/4/5/X", Dims::cube(4)).is_err());
    }

    #[test]
    fn neighborhood_shapes() {
        assert_eq!(Neighborhood::Moore.size(), 26);
        assert_eq!(Neighborhood::VonNeumann.size(), 6);
        assert!(!Neighborhood::Moore
            .block_indices()
            .contains(&neighborhood::CENTER));
    }

    #[test]
    fn von_neumann_birth_grows_a_cross() {
        let mut sim = Simulation::new(LifeLike::parse("/1/2/N").unwrap(), Dims::cube(5));
        sim.grid_mut().set(2, 2, 2, true);
        sim.step();
        let g = sim.grid();
        assert!(!g.get(2, 2, 2));
        for (x, y, z) in [
            (1, 2, 2),
            (3, 2, 2),
            (2, 1, 2),
            (2, 3, 2),
            (2, 2, 1),
            (2, 2, 3),
        ] {
            assert!(g.get(x, y, z));
        }
        assert_eq!(sim.population(), 6);
    }

    #[test]
    fn grid_wraps_around() {
        let mut sim = Simulation::new(LifeLike::parse("/1/2/N").unwrap(), Dims::cube(5));
        sim.grid_mut().set(0, 0, 0, true);
        sim.step();
        assert!(sim.grid().get(4, 0, 0) && sim.grid().get(0, 4, 0) && sim.grid().get(0, 0, 4));
    }

    #[test]
    fn generations_decay_through_states() {
        let mut sim = Simulation::new(Generations::parse("//4/M").unwrap(), Dims::cube(3));
        sim.grid_mut().set(1, 1, 1, 1);
        let mut seen = vec![];
        for _ in 0..4 {
            sim.step();
            seen.push(sim.grid().get(1, 1, 1));
        }
        assert_eq!(seen, [2, 3, 0, 0]);
        let rule = sim.rule();
        assert_eq!(rule.shade(0), None);
        assert_eq!(rule.shade(1), Some(1.0));
        assert!(rule.shade(3).unwrap() > 0.0);
    }

    #[test]
    fn surface_skips_buried_cells() {
        let mut sim = Simulation::new(LifeLike::parse("/1/2/M").unwrap(), Dims::cube(5));
        for z in 1..4 {
            for y in 1..4 {
                for x in 1..4 {
                    sim.grid_mut().set(x, y, z, true);
                }
            }
        }
        let mut out = vec![];
        assert_eq!(sim.surface(&mut out), 26);
        assert_eq!(out.len(), 26 * 8);
        assert_eq!(sim.population(), 27);
    }

    #[test]
    fn seeding_is_deterministic_and_centered() {
        let mut a = from_notation("4/4/5/M", Dims::cube(20)).unwrap();
        let mut b = from_notation("4/4/5/M", Dims::cube(20)).unwrap();
        let s = Seeding {
            extent: 0.5,
            density: 0.5,
            seed: 7,
        };
        a.seed(s);
        b.seed(s);
        let (mut sa, mut sb) = (vec![], vec![]);
        a.surface(&mut sa);
        b.surface(&mut sb);
        assert_eq!(sa, sb);
        assert!(sa
            .chunks(8)
            .all(|c| (5.0..15.0).contains(&c[0]) && (5.0..15.0).contains(&c[2])));
        let n = a.population() as f32;
        assert!((400.0..600.0).contains(&n), "population {n}");
    }

    #[test]
    fn raycast_finds_first_filled_cell_and_face() {
        let mut sim = Simulation::new(LifeLike::parse("/1/2/M").unwrap(), Dims::cube(8));
        sim.grid_mut().set(4, 4, 4, true);
        sim.grid_mut().set(6, 4, 4, true);
        // From outside along +x through the row y=4, z=4.
        let hit = sim.raycast([-3.0, 4.5, 4.5], [1.0, 0.0, 0.0]).unwrap();
        assert_eq!(
            hit,
            Hit {
                cell: [4, 4, 4],
                normal: [-1, 0, 0]
            }
        );
        // From above, diagonally down onto the top face of (6, 4, 4).
        let hit = sim.raycast([6.5, 20.0, 4.5], [0.0, -1.0, 0.0]).unwrap();
        assert_eq!(
            hit,
            Hit {
                cell: [6, 4, 4],
                normal: [0, 1, 0]
            }
        );
        assert_eq!(sim.raycast([-3.0, 1.5, 1.5], [1.0, 0.0, 0.0]), None);
        assert_eq!(sim.raycast([-3.0, 4.5, 4.5], [-1.0, 0.0, 0.0]), None);
    }

    #[test]
    fn erase_and_fill_brush() {
        let mut sim = Simulation::new(Generations::parse("4/4/5/M").unwrap(), Dims::cube(9));
        assert_eq!(sim.fill([4.5, 4.5, 4.5], 1.0), 7);
        assert_eq!(sim.population(), 7);
        assert_eq!(sim.fill([4.5, 4.5, 4.5], 1.0), 0);
        assert_eq!(sim.erase([4.5, 4.5, 4.5], 0.5), 1);
        assert_eq!(sim.population(), 6);
    }

    /// One update with fire rate 1 must match the reference computed by
    /// `training/make_fixture.py` with the training code.
    #[test]
    fn neural_step_matches_training_code() {
        #[derive(serde::Deserialize)]
        struct Fixture {
            size: usize,
            weights: serde_json::Value,
            input: Vec<f32>,
            output: Vec<f32>,
        }
        let f: Fixture =
            serde_json::from_str(include_str!("../tests/fixtures/nca_step.json")).unwrap();
        let w = rules::NcaWeights::from_json(&f.weights.to_string()).unwrap();
        let mut sim = Simulation::new(rules::NeuralCa::<8>::new(w).unwrap(), Dims::cube(f.size));
        for (cell, chunk) in sim.grid_mut().cells_mut().iter_mut().zip(f.input.chunks(8)) {
            cell.copy_from_slice(chunk);
        }
        sim.step();
        let got: Vec<f32> = sim.grid().cells().iter().flatten().copied().collect();
        let worst = got
            .iter()
            .zip(&f.output)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(worst < 1e-4, "max difference {worst}");
        assert!(sim.population() > 0);
    }

    /// Same check against a NumPy re-implementation of the PyTorch notebook's model
    /// (`training/make_torch_fixture.py`): custom kernels, channel-major features, zero boundary.
    #[test]
    fn neural_step_matches_pytorch_notebook_layout() {
        #[derive(serde::Deserialize)]
        struct Fixture {
            size: usize,
            weights: serde_json::Value,
            input: Vec<f32>,
            output: Vec<f32>,
        }
        let f: Fixture =
            serde_json::from_str(include_str!("../tests/fixtures/nca_torch_step.json")).unwrap();
        let w = rules::NcaWeights::from_json(&f.weights.to_string()).unwrap();
        let mut sim = Simulation::new(rules::NeuralCa::<16>::new(w).unwrap(), Dims::cube(f.size));
        for (cell, chunk) in sim
            .grid_mut()
            .cells_mut()
            .iter_mut()
            .zip(f.input.chunks(16))
        {
            cell.copy_from_slice(chunk);
        }
        sim.step();
        let got: Vec<f32> = sim.grid().cells().iter().flatten().copied().collect();
        let worst = got
            .iter()
            .zip(&f.output)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(worst < 1e-4, "max difference {worst}");
        assert!(sim.population() > 0);
    }
}
