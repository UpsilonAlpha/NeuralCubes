//! Dense 3D grids with toroidal (wrap-around) neighbor lookup.

/// Size of a 3D grid along each axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl Dims {
    pub const fn new(x: usize, y: usize, z: usize) -> Self {
        Self { x, y, z }
    }

    pub const fn cube(n: usize) -> Self {
        Self::new(n, n, n)
    }

    pub const fn len(&self) -> usize {
        self.x * self.y * self.z
    }

    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Linear index of `(x, y, z)`. X varies fastest, then Y, then Z.
    #[inline]
    pub const fn index(&self, x: usize, y: usize, z: usize) -> usize {
        x + self.x * (y + self.y * z)
    }

    #[inline]
    pub const fn coords(&self, i: usize) -> (usize, usize, usize) {
        (i % self.x, (i / self.x) % self.y, i / (self.x * self.y))
    }

    /// Wraps a signed coordinate pair onto the torus.
    #[inline]
    pub fn wrap(&self, x: isize, y: isize, z: isize) -> (usize, usize, usize) {
        (
            x.rem_euclid(self.x as isize) as usize,
            y.rem_euclid(self.y as isize) as usize,
            z.rem_euclid(self.z as isize) as usize,
        )
    }
}

/// A dense 3D array of cells of any type.
#[derive(Clone, Debug, PartialEq)]
pub struct Grid<C> {
    dims: Dims,
    cells: Vec<C>,
}

impl<C: Copy + Default> Grid<C> {
    pub fn new(dims: Dims) -> Self {
        Self { dims, cells: vec![C::default(); dims.len()] }
    }

    pub fn dims(&self) -> Dims {
        self.dims
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> C {
        self.cells[self.dims.index(x, y, z)]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, c: C) {
        let i = self.dims.index(x, y, z);
        self.cells[i] = c;
    }

    /// Reads a cell with toroidal wrapping, so any signed coordinate is valid.
    #[inline]
    pub fn get_wrapped(&self, x: isize, y: isize, z: isize) -> C {
        let (x, y, z) = self.dims.wrap(x, y, z);
        self.get(x, y, z)
    }

    pub fn cells(&self) -> &[C] {
        &self.cells
    }

    pub fn cells_mut(&mut self) -> &mut [C] {
        &mut self.cells
    }

    pub fn fill(&mut self, c: C) {
        self.cells.fill(c);
    }
}
