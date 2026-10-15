//! Neural cellular automata: each cell holds a vector of channels and a small learned
//! network maps what a cell perceives in its 3x3x3 block to an update.
//!
//! The architecture is the 3D version of "Growing Neural Cellular Automata"
//! (Mordvintsev et al., 2020) and matches `training/train_nca.py`:
//!
//! 1. Perception: for each channel, the identity and 3D Sobel gradients along x, y and z,
//!    giving `4 * C` features ordered kernel-major (`feature[k * C + c]`).
//! 2. Update: `delta = W2 · relu(W1 · features + b1) + b2`.
//! 3. Stochastic firing: each cell applies its update with probability `fire_rate`.
//! 4. Alive masking: a cell is kept only if some cell in its 3x3x3 block has
//!    alpha (channel 3) above `alive_threshold`, both before and after the update.
//!
//! Channels 0..3 are premultiplied RGB, channel 3 is alpha, the rest are hidden state.

use super::ParseError;
use crate::grid::Grid;
use crate::neighborhood::{Neighborhood, Neighbors};
use crate::rng::Rng;
use crate::rule::{Rule, StepContext};
use crate::sim::Seeding;
use serde::Deserialize;

pub const ALPHA: usize = 3;

/// Cells at or above this alpha are drawn and count as solid for raycasting.
pub const VISIBLE_ALPHA: f32 = 0.2;

/// Trained weights and settings, as exported by `training/train_nca.py`.
#[derive(Clone, Debug, Deserialize)]
pub struct NcaWeights {
    #[serde(default)]
    pub name: String,
    pub channels: usize,
    pub hidden: usize,
    pub fire_rate: f32,
    pub alive_threshold: f32,
    /// Grid edge the model was trained on.
    pub grid: Option<usize>,
    /// Seed cell `[z, y, x]` in the training grid.
    pub seed: Option<[usize; 3]>,
    /// `hidden x 4*channels`, row-major.
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    /// `channels x hidden`, row-major.
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

impl NcaWeights {
    pub fn from_json(json: &str) -> Result<Self, ParseError> {
        let w: Self = serde_json::from_str(json).map_err(|e| ParseError(format!("bad NCA weights: {e}")))?;
        let (c, h) = (w.channels, w.hidden);
        let checks = [("w1", w.w1.len(), h * 4 * c), ("b1", w.b1.len(), h), ("w2", w.w2.len(), c * h), ("b2", w.b2.len(), c)];
        for (name, got, want) in checks {
            if got != want {
                return Err(ParseError(format!("NCA weights: `{name}` has {got} values, expected {want}")));
            }
        }
        if c <= ALPHA {
            return Err(ParseError(format!("NCA needs at least {} channels, got {c}", ALPHA + 1)));
        }
        Ok(w)
    }
}

/// Identity and Sobel x/y/z kernels over the 3x3x3 block, in block order.
fn kernels() -> [[f32; 27]; 4] {
    let d = [-1.0, 0.0, 1.0];
    let s = [1.0, 2.0, 1.0];
    let mut k = [[0.0f32; 27]; 4];
    for (i, off) in crate::neighborhood::BLOCK_OFFSETS.iter().enumerate() {
        let [x, y, z] = off.map(|o| (o + 1) as usize);
        k[0][i] = if i == crate::neighborhood::CENTER { 1.0 } else { 0.0 };
        k[1][i] = d[x] * s[y] * s[z] / 32.0;
        k[2][i] = s[x] * d[y] * s[z] / 32.0;
        k[3][i] = s[x] * s[y] * d[z] / 32.0;
    }
    k
}

/// A neural CA with `C` channels per cell.
#[derive(Clone, Debug)]
pub struct NeuralCa<const C: usize> {
    pub weights: NcaWeights,
    kernels: [[f32; 27]; 4],
}

impl<const C: usize> NeuralCa<C>
where
    [f32; C]: Default,
{
    pub fn new(weights: NcaWeights) -> Result<Self, ParseError> {
        if weights.channels != C {
            return Err(ParseError(format!("weights have {} channels, expected {C}", weights.channels)));
        }
        Ok(Self { weights, kernels: kernels() })
    }

    /// The state a seed cell starts in: alpha and hidden channels at 1, color at 0.
    pub fn seed_state() -> [f32; C] {
        std::array::from_fn(|c| if c >= ALPHA { 1.0 } else { 0.0 })
    }

    fn update(&self, block: &[[f32; C]; 27], feats: &mut [f32], hidden: &mut [f32]) -> [f32; C] {
        let w = &self.weights;
        for (k, kernel) in self.kernels.iter().enumerate() {
            let f = &mut feats[k * C..(k + 1) * C];
            f.fill(0.0);
            for (j, &kw) in kernel.iter().enumerate() {
                if kw != 0.0 {
                    for c in 0..C {
                        f[c] += kw * block[j][c];
                    }
                }
            }
        }
        let n = 4 * C;
        for (h, out) in hidden.iter_mut().enumerate() {
            let row = &w.w1[h * n..(h + 1) * n];
            let v = w.b1[h] + row.iter().zip(feats.iter()).map(|(a, b)| a * b).sum::<f32>();
            *out = v.max(0.0);
        }
        let mut cell = block[crate::neighborhood::CENTER];
        for (c, v) in cell.iter_mut().enumerate() {
            let row = &w.w2[c * w.hidden..(c + 1) * w.hidden];
            *v += w.b2[c] + row.iter().zip(hidden.iter()).map(|(a, b)| a * b).sum::<f32>();
        }
        cell
    }

    /// Which cells have an alpha above the threshold somewhere in their 3x3x3 block.
    fn alive_mask(&self, grid: &Grid<[f32; C]>, ctx: &StepContext) -> Vec<bool> {
        let d = grid.dims();
        let thr = self.weights.alive_threshold;
        let mut own = vec![false; d.len()];
        for (o, c) in own.iter_mut().zip(grid.cells()) {
            *o = c[ALPHA] > thr;
        }
        let mut out = vec![false; d.len()];
        let mut i = 0;
        for z in 0..d.z {
            for y in 0..d.y {
                for x in 0..d.x {
                    let n = ctx.neighbors(grid, Neighborhood::Moore, x, y, z);
                    out[i] = own[i] || n.iter().any(|c| c[ALPHA] > thr);
                    i += 1;
                }
            }
        }
        out
    }
}

fn fires(generation: u64, i: usize, rate: f32) -> bool {
    let mut r = Rng::new(generation.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (i as u64));
    r.next_f32() < rate
}

impl<const C: usize> Rule for NeuralCa<C>
where
    [f32; C]: Default,
{
    type Cell = [f32; C];

    fn notation(&self) -> String {
        let name = if self.weights.name.is_empty() { "model" } else { &self.weights.name };
        format!("nca:{name}")
    }

    fn family(&self) -> &'static str {
        "neural"
    }

    fn neighborhood(&self) -> Neighborhood {
        Neighborhood::Moore
    }

    /// Deterministic (always-firing, unmasked) single-cell update. The real step is
    /// [`Rule::step`], which adds stochastic firing and alive masking.
    fn next(&self, _cell: Self::Cell, n: &Neighbors<'_, Self::Cell>) -> Self::Cell {
        let mut feats = vec![0.0; 4 * C];
        let mut hidden = vec![0.0; self.weights.hidden];
        self.update(&n.block(), &mut feats, &mut hidden)
    }

    fn seed_cell(&self, _rng: &mut Rng) -> Self::Cell {
        Self::seed_state()
    }

    fn shade(&self, cell: Self::Cell) -> Option<f32> {
        let a = cell[ALPHA];
        (a >= VISIBLE_ALPHA).then(|| a.min(1.0))
    }

    fn color(&self, cell: Self::Cell) -> Option<[f32; 3]> {
        let a = cell[ALPHA].max(VISIBLE_ALPHA);
        Some([0, 1, 2].map(|c| (cell[c] / a).clamp(0.0, 1.0)))
    }

    /// New material copies the average state of the live cells around it, so it
    /// blends into the structure instead of acting as a fresh seed.
    fn added_cell(&self, n: &Neighbors<'_, Self::Cell>, _rng: &mut Rng) -> Self::Cell {
        let mut sum = [0.0f32; C];
        let mut count = 0.0;
        for c in n.iter().filter(|c| c[ALPHA] >= VISIBLE_ALPHA) {
            for (s, v) in sum.iter_mut().zip(c) {
                *s += v;
            }
            count += 1.0;
        }
        if count == 0.0 {
            return Self::seed_state();
        }
        let mut cell = sum.map(|s| s / count);
        cell[ALPHA] = cell[ALPHA].max(1.0);
        cell
    }

    /// Places the single seed cell, centered the same way the model was trained.
    fn seed_grid(&self, grid: &mut Grid<Self::Cell>, _s: Seeding) {
        let d = grid.dims();
        let train = self.weights.grid.unwrap_or(d.x);
        let [sz, sy, sx] = self.weights.seed.unwrap_or([train / 2; 3]);
        let shift = |v: usize, n: usize| (v as isize + (n as isize - train as isize) / 2).clamp(0, n as isize - 1) as usize;
        grid.set(shift(sx, d.x), shift(sy, d.y), shift(sz, d.z), Self::seed_state());
    }

    fn step(&self, src: &Grid<Self::Cell>, dst: &mut Grid<Self::Cell>, ctx: &StepContext) {
        let d = src.dims();
        let pre = self.alive_mask(src, ctx);
        let mut feats = vec![0.0; 4 * C];
        let mut hidden = vec![0.0; self.weights.hidden];
        {
            let out = dst.cells_mut();
            let mut i = 0;
            for z in 0..d.z {
                for y in 0..d.y {
                    for x in 0..d.x {
                        out[i] = if !pre[i] {
                            [0.0; C]
                        } else if fires(ctx.generation, i, self.weights.fire_rate) {
                            let n = ctx.neighbors(src, Neighborhood::Moore, x, y, z);
                            self.update(&n.block(), &mut feats, &mut hidden)
                        } else {
                            src.cells()[i]
                        };
                        i += 1;
                    }
                }
            }
        }
        let post = self.alive_mask(dst, ctx);
        for ((cell, a), b) in dst.cells_mut().iter_mut().zip(&pre).zip(&post) {
            if !(*a && *b) {
                *cell = [0.0; C];
            }
        }
    }
}
