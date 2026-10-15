//! WebAssembly bindings: a thin wrapper over [`nc_core::Automaton`] for the browser.

use nc_core::{Automaton, Dims, Seeding};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Universe {
    sim: Box<dyn Automaton>,
    surface: Vec<f32>,
}

#[wasm_bindgen]
impl Universe {
    /// Creates a cubic universe of edge `size` running the rule in `notation` (e.g. `4/4/5/M`).
    #[wasm_bindgen(constructor)]
    pub fn new(notation: &str, size: usize) -> Result<Universe, JsError> {
        let sim = nc_core::from_notation(notation, Dims::cube(size.clamp(4, 256)))?;
        Ok(Self {
            sim,
            surface: Vec::new(),
        })
    }

    /// Creates a neural CA universe from weights JSON exported by `training/train_nca.py`.
    pub fn from_model(json: &str, size: usize) -> Result<Universe, JsError> {
        let sim = nc_core::from_nca_json(json, Dims::cube(size.clamp(4, 256)))?;
        Ok(Self {
            sim,
            surface: Vec::new(),
        })
    }

    /// Checks rule notation without building anything; returns the canonical form.
    pub fn validate(notation: &str) -> Result<String, JsError> {
        Ok(nc_core::from_notation(notation, Dims::cube(1))?.notation())
    }

    pub fn size(&self) -> usize {
        self.sim.dims().x
    }

    pub fn notation(&self) -> String {
        self.sim.notation()
    }

    pub fn family(&self) -> String {
        self.sim.family().to_string()
    }

    pub fn generation(&self) -> f64 {
        self.sim.generation() as f64
    }

    pub fn population(&self) -> usize {
        self.sim.population()
    }

    pub fn step(&mut self, steps: u32) {
        for _ in 0..steps {
            self.sim.step();
        }
    }

    pub fn clear(&mut self) {
        self.sim.clear();
    }

    /// Reseeds the grid. Rule families with a fixed seed (neural CAs) ignore the arguments.
    pub fn seed(&mut self, extent: f32, density: f32, seed: u32) {
        self.sim.seed(Seeding {
            extent,
            density,
            seed: seed as u64,
        });
    }

    /// Packed `[x, y, z, shade, r, g, b, has_color]` for every visible cell.
    pub fn surface(&mut self) -> Vec<f32> {
        self.surface.clear();
        self.sim.surface(&mut self.surface);
        self.surface.clone()
    }

    /// Casts a ray in grid coordinates. Returns `[x, y, z, nx, ny, nz]` for the first filled
    /// cell and the face it was entered through, or `undefined` on a miss.
    pub fn raycast(
        &self,
        ox: f32,
        oy: f32,
        oz: f32,
        dx: f32,
        dy: f32,
        dz: f32,
    ) -> Option<Vec<i32>> {
        self.sim.raycast([ox, oy, oz], [dx, dy, dz]).map(|h| {
            let [x, y, z] = h.cell.map(|v| v as i32);
            vec![x, y, z, h.normal[0], h.normal[1], h.normal[2]]
        })
    }

    /// Empties cells within `radius` of the center of cell `(x, y, z)`.
    pub fn erase(&mut self, x: i32, y: i32, z: i32, radius: f32) -> usize {
        self.sim
            .erase([x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5], radius)
    }

    /// Fills empty cells within `radius` of the center of cell `(x, y, z)`.
    pub fn fill(&mut self, x: i32, y: i32, z: i32, radius: f32) -> usize {
        self.sim
            .fill([x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5], radius)
    }
}
