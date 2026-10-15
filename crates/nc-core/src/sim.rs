//! Running a rule over a grid, and a type-erased interface for front ends.

use crate::grid::{Dims, Grid};
use crate::neighborhood::Neighborhood;
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

/// The first filled cell a ray enters, and the face it entered through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hit {
    pub cell: [usize; 3],
    /// Outward normal of the face that was hit, e.g. `[0, 1, 0]` for the top face.
    /// All zeros if the ray started inside a filled cell.
    pub normal: [i32; 3],
}

/// A grid evolving under a concrete rule, double-buffered.
pub struct Simulation<R: Rule> {
    rule: R,
    front: Grid<R::Cell>,
    back: Grid<R::Cell>,
    ctx: StepContext,
    generation: u64,
    rng: Rng,
}

impl<R: Rule> Simulation<R> {
    pub fn new(rule: R, dims: Dims) -> Self {
        Self {
            rule,
            front: Grid::new(dims),
            back: Grid::new(dims),
            ctx: StepContext::new(dims),
            generation: 0,
            rng: Rng::new(0x5EED),
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
        self.ctx.generation = self.generation;
        self.rule.step(&self.front, &mut self.back, &self.ctx);
        std::mem::swap(&mut self.front, &mut self.back);
        self.generation += 1;
    }

    fn filled(&self, x: usize, y: usize, z: usize) -> bool {
        self.rule.shade(self.front.get(x, y, z)).is_some()
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
    /// Appends `[x, y, z, shade, r, g, b, has_color]` for every non-empty cell that has at
    /// least one empty face neighbor (fully buried cells are skipped since they can't be seen).
    /// Returns the number of cells written.
    fn surface(&self, out: &mut Vec<f32>) -> usize;
    /// Casts a ray in grid coordinates (cell `(x, y, z)` spans `x..x+1` etc.).
    fn raycast(&self, origin: [f32; 3], dir: [f32; 3]) -> Option<Hit>;
    /// Empties every cell within `radius` of `center`. Returns how many cells changed.
    fn erase(&mut self, center: [f32; 3], radius: f32) -> usize;
    /// Fills every empty cell within `radius` of `center`. Returns how many cells changed.
    fn fill(&mut self, center: [f32; 3], radius: f32) -> usize;
}

/// Cells whose centers lie within `radius` of `center`, clipped to the grid.
fn ball(d: Dims, center: [f32; 3], radius: f32) -> impl Iterator<Item = (usize, usize, usize)> {
    let r = radius.max(0.5);
    let range = move |c: f32, n: usize| {
        let lo = (c - r).floor().max(0.0) as usize;
        let hi = ((c + r).ceil().max(0.0) as usize).min(n);
        lo..hi
    };
    let (rx, ry, rz) = (range(center[0], d.x), range(center[1], d.y), range(center[2], d.z));
    rz.flat_map(move |z| {
        let rx = rx.clone();
        ry.clone().flat_map(move |y| rx.clone().map(move |x| (x, y, z)))
    })
    .filter(move |&(x, y, z)| {
        let p = [x as f32 + 0.5 - center[0], y as f32 + 0.5 - center[1], z as f32 + 0.5 - center[2]];
        p[0] * p[0] + p[1] * p[1] + p[2] * p[2] <= r * r
    })
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
        self.rng = Rng::new(s.seed ^ 0x9E37_79B9);
        self.rule.seed_grid(&mut self.front, s);
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
                    let cell = cells[d.index(x, y, z)];
                    let Some(shade) = self.rule.shade(cell) else { continue };
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
                        let (rgb, has) = match self.rule.color(cell) {
                            Some(c) => (c, 1.0),
                            None => ([0.0; 3], 0.0),
                        };
                        out.extend_from_slice(&[x as f32, y as f32, z as f32, shade, rgb[0], rgb[1], rgb[2], has]);
                        n += 1;
                    }
                }
            }
        }
        n
    }

    fn raycast(&self, origin: [f32; 3], dir: [f32; 3]) -> Option<Hit> {
        let d = self.dims();
        let size = [d.x as f32, d.y as f32, d.z as f32];
        // Clip the ray to the grid's bounding box (slab method).
        let (mut t0, mut t1) = (0.0f32, f32::INFINITY);
        let mut entry_axis = None;
        for a in 0..3 {
            if dir[a].abs() < 1e-12 {
                if origin[a] < 0.0 || origin[a] > size[a] {
                    return None;
                }
                continue;
            }
            let (mut near, mut far) = ((0.0 - origin[a]) / dir[a], (size[a] - origin[a]) / dir[a]);
            if near > far {
                std::mem::swap(&mut near, &mut far);
            }
            if near > t0 {
                t0 = near;
                entry_axis = Some(a);
            }
            t1 = t1.min(far);
        }
        if t0 > t1 {
            return None;
        }
        // Walk cells with Amanatides & Woo's traversal.
        let p = [0, 1, 2].map(|a| origin[a] + dir[a] * (t0 + 1e-4));
        let mut cell = [0, 1, 2].map(|a| (p[a].floor() as isize).clamp(0, size[a] as isize - 1));
        let step = [0, 1, 2].map(|a| if dir[a] > 0.0 { 1isize } else { -1 });
        let delta = [0, 1, 2].map(|a| if dir[a] != 0.0 { (1.0 / dir[a]).abs() } else { f32::INFINITY });
        let mut next = [0, 1, 2].map(|a| {
            if dir[a] == 0.0 {
                f32::INFINITY
            } else {
                let boundary = cell[a] as f32 + if dir[a] > 0.0 { 1.0 } else { 0.0 };
                (boundary - p[a]) / dir[a]
            }
        });
        let mut normal = [0i32; 3];
        if let Some(a) = entry_axis {
            normal[a] = -step[a] as i32;
        }
        loop {
            let [x, y, z] = cell.map(|c| c as usize);
            if self.filled(x, y, z) {
                return Some(Hit { cell: [x, y, z], normal });
            }
            let a = if next[0] < next[1] {
                if next[0] < next[2] { 0 } else { 2 }
            } else if next[1] < next[2] {
                1
            } else {
                2
            };
            cell[a] += step[a];
            if cell[a] < 0 || cell[a] >= size[a] as isize {
                return None;
            }
            next[a] += delta[a];
            normal = [0; 3];
            normal[a] = -step[a] as i32;
        }
    }

    fn erase(&mut self, center: [f32; 3], radius: f32) -> usize {
        let d = self.dims();
        let mut n = 0;
        for (x, y, z) in ball(d, center, radius) {
            if self.front.get(x, y, z) != R::Cell::default() {
                self.front.set(x, y, z, R::Cell::default());
                n += 1;
            }
        }
        n
    }

    fn fill(&mut self, center: [f32; 3], radius: f32) -> usize {
        let d = self.dims();
        let targets: Vec<_> = ball(d, center, radius).filter(|&(x, y, z)| !self.filled(x, y, z)).collect();
        // Decide every new cell from the grid as it was, so the brush result doesn't
        // depend on iteration order.
        let snapshot = self.front.clone();
        for &(x, y, z) in &targets {
            let n = self.ctx.neighbors(&snapshot, Neighborhood::Moore, x, y, z);
            let c = self.rule.added_cell(&n, &mut self.rng);
            self.front.set(x, y, z, c);
        }
        targets.len()
    }
}
