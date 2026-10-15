//! The extension point: every rule family implements [`Rule`].

use crate::grid::{Dims, Grid};
use crate::neighborhood::{Neighborhood, Neighbors, Wrap};
use crate::rng::Rng;
use crate::sim::Seeding;

/// A cellular automaton update rule.
///
/// A rule chooses its own cell type, so a binary rule can use `bool`, a decaying rule `u8`,
/// and a continuous or neural rule a float or a fixed array of channels.
///
/// Most rules only implement [`Rule::next`], which sees one cell and its 3x3x3 block.
/// Rules that need a different evaluation strategy (larger kernels, FFT convolution,
/// stochastic updates, GPU dispatch) override [`Rule::step`] instead.
pub trait Rule: Send + Sync + 'static {
    type Cell: Copy + Default + PartialEq + Send + Sync + 'static;

    /// Canonical notation for this rule, e.g. `4/4/5/M`.
    fn notation(&self) -> String;

    /// Short name of the rule family, e.g. `life-like`.
    fn family(&self) -> &'static str;

    fn neighborhood(&self) -> Neighborhood;

    /// The next state of `cell` given its neighbors.
    fn next(&self, cell: Self::Cell, neighbors: &Neighbors<'_, Self::Cell>) -> Self::Cell;

    /// A cell to place when randomly seeding the grid.
    fn seed_cell(&self, rng: &mut Rng) -> Self::Cell;

    /// How a cell should be drawn: `None` for empty, otherwise an intensity in `0.0..=1.0`.
    fn shade(&self, cell: Self::Cell) -> Option<f32>;

    /// The cell's own color in linear RGB, for rules whose cells carry one.
    fn color(&self, _cell: Self::Cell) -> Option<[f32; 3]> {
        None
    }

    /// How solid a cell is, for smooth surfaces: the surface is drawn where this crosses
    /// [`Rule::iso_level`]. The default is 1 for any drawn cell and 0 otherwise.
    fn density(&self, cell: Self::Cell) -> f32 {
        if self.shade(cell).is_some() {
            1.0
        } else {
            0.0
        }
    }

    /// The density at which the smooth surface is drawn.
    fn iso_level(&self) -> f32 {
        0.5
    }

    /// The cell placed when a user adds material at an empty spot.
    fn added_cell(&self, _neighbors: &Neighbors<'_, Self::Cell>, rng: &mut Rng) -> Self::Cell {
        self.seed_cell(rng)
    }

    /// Fills an empty grid with a starting pattern. The default scatters
    /// [`Rule::seed_cell`] through a centered cube; rules with a fixed seed override it.
    fn seed_grid(&self, grid: &mut Grid<Self::Cell>, s: Seeding) {
        let d = grid.dims();
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
                        grid.set(x, y, z, self.seed_cell(&mut rng));
                    }
                }
            }
        }
    }

    /// Appends a cell's state as floats, for rules that can hand their state to a GPU.
    fn cell_to_floats(&self, _cell: Self::Cell, _out: &mut Vec<f32>) {}

    /// The inverse of [`Rule::cell_to_floats`]; `None` if the rule has no float form.
    fn cell_from_floats(&self, _floats: &[f32]) -> Option<Self::Cell> {
        None
    }

    /// Everything a GPU implementation of this rule needs besides its weights, as JSON.
    /// `None` for rules without a GPU path.
    fn gpu_params(&self) -> Option<String> {
        None
    }

    /// For rules whose behavior can be rotated (neural CAs with gradient perception): the
    /// orientations the rule was trained for (`none`, `yaw` or `full`). `None` if it can't rotate.
    fn orientation_mode(&self) -> Option<String> {
        None
    }

    /// Rotates the rule's behavior (radians). Ignored by rules that can't rotate.
    fn set_orientation(&mut self, _yaw: f32, _pitch: f32, _roll: f32) {}

    /// Advances the whole grid by one generation, writing into `dst`.
    fn step(&self, src: &Grid<Self::Cell>, dst: &mut Grid<Self::Cell>, ctx: &StepContext) {
        let d = src.dims();
        let shape = self.neighborhood();
        let out = dst.cells_mut();
        let mut i = 0;
        for z in 0..d.z {
            for y in 0..d.y {
                for x in 0..d.x {
                    let n = ctx.wrap.neighbors(src, shape, x, y, z);
                    out[i] = self.next(src.cells()[i], &n);
                    i += 1;
                }
            }
        }
    }
}

/// Data shared across steps for a given grid size.
pub struct StepContext {
    pub(crate) wrap: Wrap,
    /// The generation being computed, for rules that need per-step randomness.
    pub generation: u64,
}

impl StepContext {
    pub fn new(dims: Dims) -> Self {
        Self {
            wrap: Wrap::new(dims),
            generation: 0,
        }
    }

    /// Neighbors of `(x, y, z)` in `grid`, for rules that override [`Rule::step`].
    pub fn neighbors<'a, C: Copy + Default>(
        &self,
        grid: &'a Grid<C>,
        shape: Neighborhood,
        x: usize,
        y: usize,
        z: usize,
    ) -> Neighbors<'a, C> {
        self.wrap.neighbors(grid, shape, x, y, z)
    }
}
