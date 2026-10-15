//! Neural cellular automata: each cell holds a vector of channels and a small learned
//! network maps what a cell perceives in its 3x3x3 block to an update.
//!
//! The architecture is the 3D version of "Growing Neural Cellular Automata"
//! (Mordvintsev et al., 2020):
//!
//! 1. Perception: every channel is convolved with `K` fixed 3x3x3 kernels (by default
//!    the identity and Sobel x/y/z), giving `K * C` features.
//! 2. Update: `delta = W2 · relu(W1 · features + b1) + b2`.
//! 3. Stochastic firing: each cell applies its update with probability `fire_rate`.
//! 4. Alive masking: a cell is kept only if some cell in its 3x3x3 block has
//!    alpha (channel 3) above `alive_threshold`, both before and after the update.
//!
//! Channels 0..3 are RGB, channel 3 is alpha, the rest are hidden state. The kernels,
//! feature order, boundary and seed are all read from the weights file, so models from
//! different training code (`training/train_nca.py`, PyTorch notebooks) run unchanged.
//! See `docs/nca-weights.md`.

use super::ParseError;
use crate::grid::{Dims, Grid};
use crate::neighborhood::{Neighborhood, Neighbors, BLOCK_OFFSETS, CENTER};
use crate::rng::Rng;
use crate::rule::{Rule, StepContext};
use crate::sim::Seeding;
use serde::Deserialize;

pub const ALPHA: usize = 3;

/// Cells at or above this alpha are drawn and count as solid for raycasting.
pub const VISIBLE_ALPHA: f32 = 0.2;

/// How perception features are laid out in the input of the first layer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureOrder {
    /// `feature[k * C + c]`: all channels for kernel 0, then kernel 1, ...
    #[default]
    KernelMajor,
    /// `feature[c * K + k]`: what a PyTorch grouped `Conv3d(C, K*C, groups=C)` produces.
    ChannelMajor,
}

/// What lies beyond the grid edge.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Boundary {
    /// The grid is a torus (`padding_mode="circular"`).
    #[default]
    Wrap,
    /// Empty cells (PyTorch's default `padding=1`).
    Zero,
}

/// Trained weights and settings. See `docs/nca-weights.md` for the file format.
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
    /// Seed cell `[z, y, x]` in the training grid; defaults to the center.
    pub seed: Option<[usize; 3]>,
    /// Full state of the seed cell; defaults to 1 for alpha and hidden channels, 0 for RGB.
    pub seed_state: Option<Vec<f32>>,
    /// Perception kernels, 27 values each indexed `[dz][dy][dx]`. Defaults to identity + Sobel.
    pub kernels: Option<Vec<Vec<f32>>>,
    #[serde(default)]
    pub feature_order: FeatureOrder,
    #[serde(default)]
    pub boundary: Boundary,
    /// Kernels holding the x, y and z gradients, so the model can be rotated. Defaults to
    /// `[1, 2, 3]` when `kernels` is absent (the default Sobel set).
    pub gradient_kernels: Option<[usize; 3]>,
    /// Sign of each gradient kernel relative to +x, +y, +z. Defaults to all positive.
    pub gradient_signs: Option<[f32; 3]>,
    /// Orientations the model was trained for: `none`, `yaw` or `full`. Informational.
    pub rotation: Option<String>,
    /// `hidden x (kernels * channels)`, row-major.
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    /// `channels x hidden`, row-major.
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

impl NcaWeights {
    pub fn from_json(json: &str) -> Result<Self, ParseError> {
        let w: Self =
            serde_json::from_str(json).map_err(|e| ParseError(format!("bad NCA weights: {e}")))?;
        let (c, h, k) = (w.channels, w.hidden, w.kernel_count());
        let checks = [
            ("w1", w.w1.len(), h * k * c),
            ("b1", w.b1.len(), h),
            ("w2", w.w2.len(), c * h),
            ("b2", w.b2.len(), c),
            ("seed_state", w.seed_state.as_ref().map_or(c, Vec::len), c),
        ];
        for (name, got, want) in checks {
            if got != want {
                return Err(ParseError(format!(
                    "NCA weights: `{name}` has {got} values, expected {want}"
                )));
            }
        }
        if let Some(bad) = w.kernels.iter().flatten().find(|k| k.len() != 27) {
            return Err(ParseError(format!(
                "NCA weights: kernels need 27 values each, got {}",
                bad.len()
            )));
        }
        if let Some(g) = w.gradient_kernels {
            if g.iter().any(|&i| i >= k) {
                return Err(ParseError(format!(
                    "NCA weights: gradient_kernels {g:?} out of range for {k} kernels"
                )));
            }
        }
        if c <= ALPHA {
            return Err(ParseError(format!(
                "NCA needs at least {} channels, got {c}",
                ALPHA + 1
            )));
        }
        Ok(w)
    }

    fn kernel_count(&self) -> usize {
        self.kernels.as_ref().map_or(4, Vec::len)
    }

    fn gradient_kernels(&self) -> Option<[usize; 3]> {
        self.gradient_kernels
            .or(self.kernels.is_none().then_some([1, 2, 3]))
    }
}

/// Rotation matrix (acting on x, y, z with y up) for yaw about y, then pitch about x,
/// then roll about z, applied as `R = Ry(yaw) * Rx(pitch) * Rz(roll)`.
pub fn rotation_matrix(yaw: f32, pitch: f32, roll: f32) -> [[f32; 3]; 3] {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let (sr, cr) = roll.sin_cos();
    let ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]];
    let rx = [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]];
    let rz = [[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]];
    let mul = |a: [[f32; 3]; 3], b: [[f32; 3]; 3]| {
        std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
    };
    mul(mul(ry, rx), rz)
}

/// Identity and Sobel x/y/z kernels (scaled by 1/32), in block order.
pub fn default_kernels() -> Vec<[f32; 27]> {
    let d = [-1.0, 0.0, 1.0];
    let s = [1.0, 2.0, 1.0];
    let mut k = vec![[0.0f32; 27]; 4];
    for (i, off) in BLOCK_OFFSETS.iter().enumerate() {
        let [x, y, z] = off.map(|o| (o + 1) as usize);
        k[0][i] = if i == CENTER { 1.0 } else { 0.0 };
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
    /// Nonzero taps of each kernel as `(block index, weight)`.
    taps: Vec<Vec<(usize, f32)>>,
    seed: [f32; C],
    /// Current orientation; `None` is the trained orientation.
    rotation: Option<[[f32; 3]; 3]>,
}

impl<const C: usize> NeuralCa<C>
where
    [f32; C]: Default,
{
    pub fn new(weights: NcaWeights) -> Result<Self, ParseError> {
        if weights.channels != C {
            return Err(ParseError(format!(
                "weights have {} channels, expected {C}",
                weights.channels
            )));
        }
        // Kernels in the file are indexed [dz][dy][dx], which is exactly block order.
        let kernels: Vec<[f32; 27]> = match &weights.kernels {
            Some(ks) => ks.iter().map(|k| std::array::from_fn(|i| k[i])).collect(),
            None => default_kernels(),
        };
        let taps = kernels
            .iter()
            .map(|k| {
                k.iter()
                    .enumerate()
                    .filter(|(_, w)| **w != 0.0)
                    .map(|(i, w)| (i, *w))
                    .collect()
            })
            .collect();
        let seed = match &weights.seed_state {
            Some(s) => std::array::from_fn(|c| s[c]),
            None => std::array::from_fn(|c| if c >= ALPHA { 1.0 } else { 0.0 }),
        };
        Ok(Self {
            weights,
            taps,
            seed,
            rotation: None,
        })
    }

    /// Whether this model's gradient kernels are known, so it can be rotated.
    pub fn rotatable(&self) -> bool {
        self.weights.gradient_kernels().is_some()
    }

    /// Grows and repairs the shape rotated by `r` (see [`rotation_matrix`]). Models only
    /// hold their shape well at orientations they were trained for (`weights.rotation`).
    pub fn set_rotation(&mut self, r: [[f32; 3]; 3]) {
        let identity =
            (0..3).all(|i| (0..3).all(|j| (r[i][j] - f32::from(u8::from(i == j))).abs() < 1e-6));
        self.rotation = (!identity && self.rotatable()).then_some(r);
    }

    fn feature(&self, k: usize, c: usize) -> usize {
        match self.weights.feature_order {
            FeatureOrder::KernelMajor => k * C + c,
            FeatureOrder::ChannelMajor => c * self.taps.len() + k,
        }
    }

    /// The state a seed cell starts in.
    pub fn seed_state(&self) -> [f32; C] {
        self.seed
    }

    /// The 3x3x3 block around `(x, y, z)`, honoring the boundary mode.
    fn block(
        &self,
        grid: &Grid<[f32; C]>,
        ctx: &StepContext,
        x: usize,
        y: usize,
        z: usize,
    ) -> [[f32; C]; 27] {
        let d = grid.dims();
        let interior = x > 0 && y > 0 && z > 0 && x + 1 < d.x && y + 1 < d.y && z + 1 < d.z;
        if interior || self.weights.boundary == Boundary::Wrap {
            return ctx.neighbors(grid, Neighborhood::Moore, x, y, z).block();
        }
        std::array::from_fn(|i| {
            let [dx, dy, dz] = BLOCK_OFFSETS[i].map(isize::from);
            let (nx, ny, nz) = (x as isize + dx, y as isize + dy, z as isize + dz);
            let inside = |v: isize, n: usize| v >= 0 && v < n as isize;
            if inside(nx, d.x) && inside(ny, d.y) && inside(nz, d.z) {
                grid.get(nx as usize, ny as usize, nz as usize)
            } else {
                [0.0; C]
            }
        })
    }

    fn update(&self, block: &[[f32; C]; 27], feats: &mut [f32], hidden: &mut [f32]) -> [f32; C] {
        let w = &self.weights;
        let kn = self.taps.len();
        feats.fill(0.0);
        for (k, taps) in self.taps.iter().enumerate() {
            for &(j, kw) in taps {
                for (c, v) in block[j].iter().enumerate() {
                    feats[self.feature(k, c)] += kw * v;
                }
            }
        }
        if let (Some(r), Some(g)) = (self.rotation, w.gradient_kernels()) {
            // Express each channel's gradient in the shape's own frame: R^T g.
            let sign = w.gradient_signs.unwrap_or([1.0; 3]);
            for c in 0..C {
                let idx = [0, 1, 2].map(|a| self.feature(g[a], c));
                let v = [0, 1, 2].map(|a| feats[idx[a]] * sign[a]);
                for a in 0..3 {
                    feats[idx[a]] = (r[0][a] * v[0] + r[1][a] * v[1] + r[2][a] * v[2]) * sign[a];
                }
            }
        }
        let n = kn * C;
        for (h, out) in hidden.iter_mut().enumerate() {
            let row = &w.w1[h * n..(h + 1) * n];
            let v = w.b1[h]
                + row
                    .iter()
                    .zip(feats.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>();
            *out = v.max(0.0);
        }
        let mut cell = block[CENTER];
        for (c, v) in cell.iter_mut().enumerate() {
            let row = &w.w2[c * w.hidden..(c + 1) * w.hidden];
            *v += w.b2[c]
                + row
                    .iter()
                    .zip(hidden.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>();
        }
        cell
    }

    /// Which cells have an alpha above the threshold somewhere in their 3x3x3 block.
    fn alive_mask(&self, grid: &Grid<[f32; C]>) -> Vec<bool> {
        let d = grid.dims();
        let thr = self.weights.alive_threshold;
        let own: Vec<bool> = grid.cells().iter().map(|c| c[ALPHA] > thr).collect();
        let mut out = vec![false; d.len()];
        let mut i = 0;
        for z in 0..d.z {
            for y in 0..d.y {
                for x in 0..d.x {
                    out[i] = self.block_any(&own, d, x, y, z);
                    i += 1;
                }
            }
        }
        out
    }

    fn block_any(&self, own: &[bool], d: Dims, x: usize, y: usize, z: usize) -> bool {
        if own[d.index(x, y, z)] {
            return true;
        }
        let wrap = self.weights.boundary == Boundary::Wrap;
        BLOCK_OFFSETS.iter().any(|o| {
            let [dx, dy, dz] = o.map(isize::from);
            let (nx, ny, nz) = (x as isize + dx, y as isize + dy, z as isize + dz);
            if wrap {
                let (a, b, c) = d.wrap(nx, ny, nz);
                own[d.index(a, b, c)]
            } else {
                let inside = |v: isize, n: usize| v >= 0 && v < n as isize;
                inside(nx, d.x)
                    && inside(ny, d.y)
                    && inside(nz, d.z)
                    && own[d.index(nx as usize, ny as usize, nz as usize)]
            }
        })
    }
}

/// A 32-bit integer hash ("lowbias32"), cheap on GPUs, so the CPU and WebGPU paths
/// make the same firing decisions.
pub fn hash32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

/// Whether cell `i` applies its update at `generation`.
pub fn fires(generation: u64, i: usize, rate: f32) -> bool {
    let h = hash32(i as u32 ^ hash32(generation as u32 ^ 0x9e37_79b9));
    ((h >> 8) as f32) / 16_777_216.0 < rate
}

impl<const C: usize> Rule for NeuralCa<C>
where
    [f32; C]: Default,
{
    type Cell = [f32; C];

    fn notation(&self) -> String {
        let name = if self.weights.name.is_empty() {
            "model"
        } else {
            &self.weights.name
        };
        format!("nca:{name}")
    }

    fn family(&self) -> &'static str {
        "neural"
    }

    fn neighborhood(&self) -> Neighborhood {
        Neighborhood::Moore
    }

    /// Deterministic (always-firing, unmasked, wrapping) single-cell update. The real
    /// step is [`Rule::step`], which adds stochastic firing, masking and the boundary mode.
    fn next(&self, _cell: Self::Cell, n: &Neighbors<'_, Self::Cell>) -> Self::Cell {
        let mut feats = vec![0.0; self.taps.len() * C];
        let mut hidden = vec![0.0; self.weights.hidden];
        self.update(&n.block(), &mut feats, &mut hidden)
    }

    fn seed_cell(&self, _rng: &mut Rng) -> Self::Cell {
        self.seed
    }

    fn shade(&self, cell: Self::Cell) -> Option<f32> {
        let a = cell[ALPHA];
        (a >= VISIBLE_ALPHA).then(|| a.min(1.0))
    }

    fn density(&self, cell: Self::Cell) -> f32 {
        cell[ALPHA].clamp(0.0, 1.0)
    }

    fn iso_level(&self) -> f32 {
        VISIBLE_ALPHA
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
            return self.seed;
        }
        let mut cell = sum.map(|s| s / count);
        cell[ALPHA] = cell[ALPHA].max(1.0);
        cell
    }

    /// Places the single seed cell where the model was trained to start, centered in the
    /// grid and turned with the current rotation.
    fn seed_grid(&self, grid: &mut Grid<Self::Cell>, _s: Seeding) {
        let d = grid.dims();
        let train = self.weights.grid.unwrap_or(d.x);
        let [sz, sy, sx] = self.weights.seed.unwrap_or([train / 2; 3]);
        let tc = (train as f32 - 1.0) / 2.0;
        let p = [sx as f32 - tc, sy as f32 - tc, sz as f32 - tc];
        let p = match self.rotation {
            Some(r) => std::array::from_fn(|i| (0..3).map(|j| r[i][j] * p[j]).sum::<f32>()),
            None => p,
        };
        let place =
            |v: f32, n: usize| ((v + (n as f32 - 1.0) / 2.0).round().max(0.0) as usize).min(n - 1);
        grid.set(
            place(p[0], d.x),
            place(p[1], d.y),
            place(p[2], d.z),
            self.seed,
        );
    }

    fn cell_to_floats(&self, cell: Self::Cell, out: &mut Vec<f32>) {
        out.extend_from_slice(&cell);
    }

    fn cell_from_floats(&self, f: &[f32]) -> Option<Self::Cell> {
        (f.len() == C).then(|| std::array::from_fn(|c| f[c]))
    }

    fn gpu_params(&self) -> Option<String> {
        let w = &self.weights;
        let kernels: Vec<Vec<f32>> = match &w.kernels {
            Some(k) => k.clone(),
            None => default_kernels().iter().map(|k| k.to_vec()).collect(),
        };
        Some(
            serde_json::json!({
                "channels": C,
                "hidden": w.hidden,
                "kernels": kernels,
                "channel_major": w.feature_order == FeatureOrder::ChannelMajor,
                "wrap": w.boundary == Boundary::Wrap,
                "fire_rate": w.fire_rate,
                "alive_threshold": w.alive_threshold,
                "rotation": self.rotation,
                "gradient_kernels": w.gradient_kernels(),
                "gradient_signs": w.gradient_signs.unwrap_or([1.0; 3]),
            })
            .to_string(),
        )
    }

    fn orientation_mode(&self) -> Option<String> {
        self.rotatable().then(|| {
            self.weights
                .rotation
                .clone()
                .unwrap_or_else(|| "none".into())
        })
    }

    fn set_orientation(&mut self, yaw: f32, pitch: f32, roll: f32) {
        self.set_rotation(rotation_matrix(yaw, pitch, roll));
    }

    fn step(&self, src: &Grid<Self::Cell>, dst: &mut Grid<Self::Cell>, ctx: &StepContext) {
        let d = src.dims();
        let pre = self.alive_mask(src);
        let mut feats = vec![0.0; self.taps.len() * C];
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
                            self.update(&self.block(src, ctx, x, y, z), &mut feats, &mut hidden)
                        } else {
                            src.cells()[i]
                        };
                        i += 1;
                    }
                }
            }
        }
        let post = self.alive_mask(dst);
        for ((cell, a), b) in dst.cells_mut().iter_mut().zip(&pre).zip(&post) {
            if !(*a && *b) {
                *cell = [0.0; C];
            }
        }
    }
}
