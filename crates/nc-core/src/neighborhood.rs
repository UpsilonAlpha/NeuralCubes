//! Neighborhood shapes and the per-cell view a rule sees.

use crate::grid::{Dims, Grid};

/// Which surrounding cells count as neighbors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Neighborhood {
    /// All 26 cells in the surrounding 3x3x3 cube.
    Moore,
    /// The 6 face-adjacent cells.
    VonNeumann,
}

/// Offsets of the full 3x3x3 block, in the order used by [`Neighbors`]:
/// `dx` fastest, then `dy`, then `dz`, each running -1, 0, 1. Index 13 is the center.
pub const BLOCK_OFFSETS: [[i8; 3]; 27] = {
    let mut out = [[0i8; 3]; 27];
    let mut i = 0;
    while i < 27 {
        out[i] = [
            (i % 3) as i8 - 1,
            ((i / 3) % 3) as i8 - 1,
            (i / 9) as i8 - 1,
        ];
        i += 1;
    }
    out
};

/// Index of `(dx, dy, dz)` within [`BLOCK_OFFSETS`].
#[inline]
pub const fn block_index(dx: i8, dy: i8, dz: i8) -> usize {
    ((dx + 1) + 3 * (dy + 1) + 9 * (dz + 1)) as usize
}

pub const CENTER: usize = 13;

const MOORE: [usize; 26] = {
    let mut out = [0usize; 26];
    let mut i = 0;
    let mut j = 0;
    while i < 27 {
        if i != CENTER {
            out[j] = i;
            j += 1;
        }
        i += 1;
    }
    out
};

const VON_NEUMANN: [usize; 6] = [
    block_index(-1, 0, 0),
    block_index(1, 0, 0),
    block_index(0, -1, 0),
    block_index(0, 1, 0),
    block_index(0, 0, -1),
    block_index(0, 0, 1),
];

impl Neighborhood {
    /// Positions within the 3x3x3 block (see [`BLOCK_OFFSETS`]) that belong to this neighborhood.
    pub fn block_indices(self) -> &'static [usize] {
        match self {
            Neighborhood::Moore => &MOORE,
            Neighborhood::VonNeumann => &VON_NEUMANN,
        }
    }

    pub fn size(self) -> usize {
        self.block_indices().len()
    }

    pub fn letter(self) -> char {
        match self {
            Neighborhood::Moore => 'M',
            Neighborhood::VonNeumann => 'N',
        }
    }
}

/// The 3x3x3 block around one cell, as handed to a rule.
///
/// Totalistic rules only need [`Neighbors::iter`]; rules that care about direction
/// (gradients, learned convolutions) can use [`Neighbors::at`] or [`Neighbors::block`].
pub struct Neighbors<'a, C> {
    cells: &'a [C],
    block: [usize; 27],
    shape: Neighborhood,
}

impl<'a, C: Copy> Neighbors<'a, C> {
    /// Cell at offset `(dx, dy, dz)`, each in -1..=1. `(0, 0, 0)` is the cell itself.
    #[inline]
    pub fn at(&self, dx: i8, dy: i8, dz: i8) -> C {
        self.cells[self.block[block_index(dx, dy, dz)]]
    }

    /// The neighbors in this rule's neighborhood shape (excluding the center).
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = C> + '_ {
        self.shape
            .block_indices()
            .iter()
            .map(move |&b| self.cells[self.block[b]])
    }

    /// Number of neighbors matching `pred`.
    #[inline]
    pub fn count(&self, mut pred: impl FnMut(C) -> bool) -> u32 {
        self.iter().filter(|&c| pred(c)).count() as u32
    }

    /// The full 3x3x3 block in [`BLOCK_OFFSETS`] order, regardless of neighborhood shape.
    pub fn block(&self) -> [C; 27] {
        std::array::from_fn(|i| self.cells[self.block[i]])
    }
}

/// Precomputed wrap-around coordinates, so building a [`Neighbors`] costs no division.
pub(crate) struct Wrap {
    dims: Dims,
    xs: Vec<[usize; 3]>,
    ys: Vec<[usize; 3]>,
    zs: Vec<[usize; 3]>,
}

impl Wrap {
    pub fn new(dims: Dims) -> Self {
        let axis = |n: usize| -> Vec<[usize; 3]> {
            (0..n).map(|i| [(i + n - 1) % n, i, (i + 1) % n]).collect()
        };
        Self {
            dims,
            xs: axis(dims.x),
            ys: axis(dims.y),
            zs: axis(dims.z),
        }
    }

    #[inline]
    pub fn neighbors<'a, C: Copy + Default>(
        &self,
        grid: &'a Grid<C>,
        shape: Neighborhood,
        x: usize,
        y: usize,
        z: usize,
    ) -> Neighbors<'a, C> {
        let (xs, ys, zs) = (self.xs[x], self.ys[y], self.zs[z]);
        let d = self.dims;
        let mut block = [0usize; 27];
        let mut i = 0;
        for zz in zs {
            for yy in ys {
                for xx in xs {
                    block[i] = d.index(xx, yy, zz);
                    i += 1;
                }
            }
        }
        Neighbors {
            cells: grid.cells(),
            block,
            shape,
        }
    }
}
