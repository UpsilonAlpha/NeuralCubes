# Neural Cubes

A versatile, extensible 3D cellular automata engine in Rust, with a browser front end
compiled to WebAssembly and deployed to GitHub Pages.

## Layout

```
crates/nc-core   simulation core: grids, neighborhoods, the Rule trait, built-in rule families
crates/nc-web    wasm-bindgen wrapper exposing a `Universe` to JavaScript
web/             static front end (WebGL2 instanced cubes, no JS dependencies)
web/models/      trained neural CA weights the viewer can load
training/        JAX scripts: voxelize a mesh, train a regenerating NCA, export weights
docs/            the neural CA weights format
.github/         CI: tests, wasm build, Pages deploy
```

## Run it locally

Requirements: Rust (stable), the `wasm32-unknown-unknown` target, and
[wasm-pack](https://rustwasm.github.io/wasm-pack/).

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

wasm-pack build crates/nc-web --release --target web --out-dir ../../web/pkg --no-pack
python3 -m http.server 8000 -d web      # any static server works
# open http://localhost:8000
```

If wasm-pack can't download `wasm-opt`, add `--no-opt` to the build command.

Tests and a rough benchmark:

```sh
cargo test --workspace
cargo run --release -p nc-core --example bench "4/4/5/M" 64
```

The page URL keeps the current rule and size in its hash, e.g.
`#rule=4/4/5/M&size=48` or `#model=bunny`, so a link reproduces a setup.

## Using the viewer

- **Play / Pause / Step** control the simulation (keys: Space, `.`). Speed is in steps per second.
- **Orbit / Add / Remove** pick what the cursor does (keys: `1`, `2`, `3`). Add and Remove
  cast a ray from the cursor into the grid, preview the brush, and fill or empty the cells
  it covers. Right-drag always orbits; scroll or pinch zooms.
- **Load neural CA weights** runs any model exported in the [weights format](docs/nca-weights.md).

The "Bunny" preset is a neural CA trained to grow the Stanford bunny from one cell and
to regrow it after damage: pause, carve a chunk out with Remove, and press Play.

## Rule families

A rule family is one kind of update rule; individual rules are parameter choices within it.
Each family is an implementation of `nc_core::Rule`:

```rust
pub trait Rule {
    type Cell: Copy + Default + PartialEq;          // bool, u8, f32, [f32; 16], ...
    fn notation(&self) -> String;
    fn family(&self) -> &'static str;
    fn neighborhood(&self) -> Neighborhood;
    fn next(&self, cell: Self::Cell, n: &Neighbors<Self::Cell>) -> Self::Cell;
    fn seed_cell(&self, rng: &mut Rng) -> Self::Cell;
    fn shade(&self, cell: Self::Cell) -> Option<f32>;  // None = empty, else 0..1 for drawing
    fn step(&self, src: &Grid<Self::Cell>, dst: &mut Grid<Self::Cell>, ctx: &StepContext) { /* calls next() per cell */ }
}
```

`Neighbors` gives both totalistic access (`count`, `iter`) and positional access
(`at(dx, dy, dz)`, `block()`), so direction-aware rules such as neural cellular automata
fit the same interface. Rules needing a different evaluation strategy override `step`.

Built in:

| Family        | Cell | Notation        | Example                   |
|---------------|------|-----------------|---------------------------|
| `life-like`   | bool | `S/B/2/N`       | `4-6/3/2/M` (Architecture) |
| `generations` | u8   | `S/B/C/N`, C>2  | `4/4/5/M` (445)            |
| `neural`      | [f32; C] | weights JSON | `nca:bunny`               |

`S` and `B` are neighbor counts for survival and birth (`5-7,12`), `C` the number of
states, and `N` the neighborhood: `M` (Moore, 26 cells) or `N` (von Neumann, 6). The
grid wraps around at its edges.

The neural family runs the 3D version of
[Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/): Sobel perception,
a two-layer update network, stochastic updates and alive masking. Rust does inference
only; training lives in `training/`.

Planned: continuous (Lenia-style) rules.

## Training a neural CA

```sh
pip install jax optax numpy
cd training
python voxelize.py mesh.json --size 24 --fit 21 --out targets/shape.npy   # {"positions", "cells"} mesh
python train_nca.py targets/shape.npy --out ../web/models/shape.json --iterations 1500
```

On a 4-core CPU the bundled bunny (24³) takes about 5 s per iteration. Pass `--remat`
for 32³ and larger grids to save memory; a GPU is much faster for those.

## Adding a rule family

1. Add a module under `crates/nc-core/src/rules/` with a struct implementing `Rule`.
2. Teach `nc_core::from_notation` (or a new constructor) how to build it.
3. The `Automaton` trait object, the wasm wrapper and the renderer need no changes.

## Deploying

Push to `main` on GitHub and enable Pages with **Settings → Pages → Source: GitHub Actions**.
The workflow runs the tests, builds the wasm package and publishes `web/`.
