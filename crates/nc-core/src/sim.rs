//! Running a rule over a grid, and a type-erased interface for front ends.

use crate::grid::{Dims, Grid};
use crate::rng::Rng;
use crate::rule::{Rule, StepContext};

/// How to fill the grid when (re)seeding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Seeding {
    /// Edge length of the centered seeding cube, as a fraction of the grid (0..=1).
    pub extent: f32,
    /// Probability that a cell inside the seeding cube is seeded.
    pub density: f32,
    pub seed: u64,
}

impl Default for Seeding {
    fn default() -> Self {
        Self { extent: 0.25, density: 0.5, seed: 1 }
    }
}

/// A grid evolving under a concrete rule, double-buffered.
pub struct Simulation<R: Rule> {
    rule: R,
    front: Grid<R::Cell>,
    back: Grid<R::Cell>,
    ctx: StepContext,
    generation: u64,
}

impl<R: Rule> Simulation<R> {
    pub fn new(rule: R, dims: Dims) -> Self {
        Self {
            rule,
            front: Grid::new(dims),
            back: Grid::new(dims),
            ctx: StepContext::new(dims),
            generation: 0,
        }
    }

    pub fn rule(&self) -> &R {
        &self.rule
    }

    pub fn grid(&self) -> &Grid<R::Cell> {
        &self.front
    }

    pub fn grid_mut(&mut self) -> &mut Grid<R::Cell> {
        &mut self.front
    }

    pub fn step(&mut self) {
        self.rule.step(&self.front, &mut self.back, &self.ctx);
        std::mem::swap(&mut self.front, &mut self.back);
        self.generation += 1;
    }
}

/// Object-safe view of any [`Simulation`], so front ends can switch rule families at runtime.
pub trait Automaton {
    fn dims(&self) -> Dims;
    fn notation(&self) -> String;
    fn family(&self) -> &'static str;
    fn generation(&self) -> u64;
    fn step(&mut self);
    fn clear(&mut self);
    fn seed(&mut self, seeding: Seeding);
    /// Number of non-empty cells.
    fn population(&self) -> usize;
    /// Appends `[x, y, z, shade]` for every non-empty cell that has at least one empty
    /// face neighbor (fully buried cells are skipped since they can't be seen).
    /// Returns the number of cells written.
    fn surface(&self, out: &mut Vec<f32>) -> usize;
}

impl<R: Rule> Automaton for Simulation<R> {
    fn dims(&self) -> Dims {
        self.front.dims()
    }

    fn notation(&self) -> String {
        self.rule.notation()
    }

    fn family(&self) -> &'static str {
        self.rule.family()
    }

    fn generation(&self) -> u64 {
        self.generation
    }

    fn step(&mut self) {
        Simulation::step(self)
    }

    fn clear(&mut self) {
        self.front.fill(R::Cell::default());
        self.generation = 0;
    }

    fn seed(&mut self, s: Seeding) {
        self.clear();
        let d = self.dims();
        let mut rng = Rng::new(s.seed);
        let span = |n: usize| {
            let k = ((n as f32 * s.extent.clamp(0.0, 1.0)).round() as usize).clamp(1, n);
            let lo = (n - k) / 2;
            lo..lo + k
        };
        for z in span(d.z) {
            for y in span(d.y) {
                for x in span(d.x) {
                    if rng.next_f32() < s.density {
                        let c = self.rule.seed_cell(&mut rng);
                        self.front.set(x, y, z, c);
                    }
                }
            }
        }
    }

    fn population(&self) -> usize {
        self.front.cells().iter().filter(|&&c| self.rule.shade(c).is_some()).count()
    }

    fn surface(&self, out: &mut Vec<f32>) -> usize {
        let d = self.dims();
        let cells = self.front.cells();
        let filled = |x: usize, y: usize, z: usize| self.rule.shade(cells[d.index(x, y, z)]).is_some();
        let mut n = 0;
        for z in 0..d.z {
            for y in 0..d.y {
                for x in 0..d.x {
                    let Some(shade) = self.rule.shade(cells[d.index(x, y, z)]) else { continue };
                    // Cells on the grid boundary are always visible.
                    let buried = x > 0
                        && y > 0
                        && z > 0
                        && x + 1 < d.x
                        && y + 1 < d.y
                        && z + 1 < d.z
                        && filled(x - 1, y, z)
                        && filled(x + 1, y, z)
                        && filled(x, y - 1, z)
                        && filled(x, y + 1, z)
                        && filled(x, y, z - 1)
                        && filled(x, y, z + 1);
                    if !buried {
                        out.extend_from_slice(&[x as f32, y as f32, z as f32, shade]);
                        n += 1;
                    }
                }
            }
        }
        n
    }
}
