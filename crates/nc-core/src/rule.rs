//! The extension point: every rule family implements [`Rule`].

use crate::grid::Grid;
use crate::neighborhood::{Neighborhood, Neighbors, Wrap};
use crate::rng::Rng;

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

/// Precomputed data shared across steps for a given grid size.
pub struct StepContext {
    pub(crate) wrap: Wrap,
}

impl StepContext {
    pub fn new(dims: crate::Dims) -> Self {
        Self { wrap: Wrap::new(dims) }
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
