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
        let size = size.clamp(4, 256);
        let sim = nc_core::from_notation(notation, Dims::cube(size))?;
        Ok(Self { sim, surface: Vec::new() })
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

    /// Fills a centered cube (`extent` of the grid edge) with live cells at `density`.
    pub fn seed(&mut self, extent: f32, density: f32, seed: u32) {
        self.sim.seed(Seeding { extent, density, seed: seed as u64 });
    }

    /// Packed `[x, y, z, shade]` for every visible cell.
    pub fn surface(&mut self) -> Vec<f32> {
        self.surface.clear();
        self.sim.surface(&mut self.surface);
        self.surface.clone()
    }
}
