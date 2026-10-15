//! Smooth surfaces: marching cubes over a per-cell density field.

use crate::grid::Dims;
use crate::mc_tables::{EDGE_TABLE, TRI_TABLE};

/// Floats per output vertex: position, normal, then shade, r, g, b, has_color.
pub const MESH_STRIDE: usize = 11;

/// Number of per-cell attributes carried to the surface (shade, r, g, b, has_color).
const ATTRS: usize = 5;

const CORNERS: [[usize; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
];

const EDGES: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];

/// Empty cells added around the grid so the surface closes at the boundary.
const PAD: usize = 2;

/// A density field sampled at cell centers, plus attributes premultiplied by density so
/// that blurring blends colors instead of darkening them.
struct Field {
    n: [usize; 3],
    density: Vec<f32>,
    attrs: Vec<[f32; ATTRS]>,
}

impl Field {
    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.n[1] + y) * self.n[0] + x
    }

    /// One pass of a separable [1, 2, 1] / 4 blur along each axis.
    fn blur(&mut self) {
        for axis in 0..3 {
            let stride = [1, self.n[0], self.n[0] * self.n[1]][axis];
            let len = self.n[axis];
            let d = self.density.clone();
            let a = self.attrs.clone();
            for i in 0..d.len() {
                let pos = (i / stride) % len;
                if pos == 0 || pos + 1 == len {
                    continue; // the outermost padding stays empty
                }
                let (lo, hi) = (i - stride, i + stride);
                self.density[i] = 0.25 * d[lo] + 0.5 * d[i] + 0.25 * d[hi];
                self.attrs[i] =
                    std::array::from_fn(|k| 0.25 * a[lo][k] + 0.5 * a[i][k] + 0.25 * a[hi][k]);
            }
        }
    }

    /// Central-difference gradient of the density.
    fn gradient(&self, x: usize, y: usize, z: usize) -> [f32; 3] {
        let p = [x, y, z];
        std::array::from_fn(|axis| {
            let mut lo = p;
            let mut hi = p;
            lo[axis] = lo[axis].saturating_sub(1);
            hi[axis] = (hi[axis] + 1).min(self.n[axis] - 1);
            let span = (hi[axis] - lo[axis]).max(1) as f32;
            (self.density[self.index(hi[0], hi[1], hi[2])]
                - self.density[self.index(lo[0], lo[1], lo[2])])
                / span
        })
    }
}

/// Extracts the `iso` level set of a per-cell density with marching cubes and appends
/// triangles to `out` as [`MESH_STRIDE`] floats per vertex, in grid coordinates (cell
/// `(x, y, z)` spans `x..x+1`). `cell(i)` gives the density and `[shade, r, g, b, has_color]`
/// of the cell at linear index `i`. `smooth` blur passes soften blocky inputs such as
/// binary occupancy. Returns the number of triangles.
pub fn marching_cubes(
    dims: Dims,
    iso: f32,
    smooth: u32,
    cell: impl Fn(usize) -> (f32, [f32; ATTRS]),
    out: &mut Vec<f32>,
) -> usize {
    let n = [dims.x + 2 * PAD, dims.y + 2 * PAD, dims.z + 2 * PAD];
    let mut f = Field {
        n,
        density: vec![0.0; n[0] * n[1] * n[2]],
        attrs: vec![[0.0; ATTRS]; n[0] * n[1] * n[2]],
    };
    for z in 0..dims.z {
        for y in 0..dims.y {
            for x in 0..dims.x {
                let (d, a) = cell(dims.index(x, y, z));
                let i = f.index(x + PAD, y + PAD, z + PAD);
                f.density[i] = d;
                f.attrs[i] = a.map(|v| v * d);
            }
        }
    }
    for _ in 0..smooth {
        f.blur();
    }

    let mut tris = 0;
    let mut verts = [[0.0f32; MESH_STRIDE]; 12];
    for z in 0..n[2] - 1 {
        for y in 0..n[1] - 1 {
            for x in 0..n[0] - 1 {
                let idx: [usize; 8] = CORNERS.map(|c| f.index(x + c[0], y + c[1], z + c[2]));
                let mut case = 0;
                for (bit, &i) in idx.iter().enumerate() {
                    if f.density[i] < iso {
                        case |= 1 << bit;
                    }
                }
                let edges = EDGE_TABLE[case] as u16;
                if edges == 0 {
                    continue;
                }
                for (e, &[a, b]) in EDGES.iter().enumerate() {
                    if edges & (1 << e) == 0 {
                        continue;
                    }
                    let (da, db) = (f.density[idx[a]], f.density[idx[b]]);
                    let t = if (db - da).abs() > 1e-6 {
                        ((iso - da) / (db - da)).clamp(0.0, 1.0)
                    } else {
                        0.5
                    };
                    let (ca, cb) = (CORNERS[a], CORNERS[b]);
                    let ga = f.gradient(x + ca[0], y + ca[1], z + ca[2]);
                    let gb = f.gradient(x + cb[0], y + cb[1], z + cb[2]);
                    let v = &mut verts[e];
                    let mut len = 0.0;
                    for k in 0..3 {
                        let base = [x, y, z][k] as f32 - PAD as f32 + 0.5;
                        v[k] = base + ca[k] as f32 + t * (cb[k] as f32 - ca[k] as f32);
                        // Density falls off outward, so the outward normal is minus the gradient.
                        v[3 + k] = -(ga[k] + t * (gb[k] - ga[k]));
                        len += v[3 + k] * v[3 + k];
                    }
                    let len = len.sqrt().max(1e-6);
                    v[3..6].iter_mut().for_each(|c| *c /= len);
                    let d = (da + t * (db - da)).max(1e-6);
                    let (aa, ab) = (f.attrs[idx[a]], f.attrs[idx[b]]);
                    for k in 0..ATTRS {
                        v[6 + k] = (aa[k] + t * (ab[k] - aa[k])) / d;
                    }
                }
                for tri in TRI_TABLE[case].chunks(3) {
                    if tri[0] < 0 {
                        break;
                    }
                    for &e in tri {
                        out.extend_from_slice(&verts[e as usize]);
                    }
                    tris += 1;
                }
            }
        }
    }
    tris
}
