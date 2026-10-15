# Neural Cubes

A versatile, extensible 3D cellular automata engine in Rust, with a browser front end
compiled to WebAssembly and deployed to GitHub Pages.

## Layout

```
crates/nc-core   simulation core: grids, neighborhoods, the Rule trait, built-in rule families
crates/nc-web    wasm-bindgen wrapper exposing a `Universe` to JavaScript
web/             static front end (WebGL2 rendering, WebGPU stepping, no JS dependencies)
web/models/      trained neural CA weights the viewer can load (listed in index.json)
training/        JAX scripts: build targets, train a regenerating NCA, export weights
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
- **View: Smooth** draws a marching-cubes surface instead of cubes: from the alpha
  channel for neural CAs (with colors blended between cells), and from blurred occupancy
  for the other families.
- **Orientation** (neural CAs only) turns the model's growth direction. Models trained with
  `--rotation yaw` hold their shape at any turn; `full` adds tilt and roll. Press Reseed to
  grow from scratch at the new angle, or change it live and watch the shape rebuild.
- **Run the model on the GPU** steps neural CAs with a WebGPU compute shader
  (`web/gpu.js`) when the browser supports it, and falls back to the Rust/wasm CPU path
  otherwise. Drawing, raycasting and edits always run against the CPU copy; GPU state is
  read back after each batch of steps and re-uploaded after an edit.
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

Any family gets the Smooth view for free: `Rule::density` and `Rule::iso_level` say what
the surface follows (by default, drawn cells at 0.5).

## Training a neural CA

Training is Python (JAX) and runs anywhere JAX does; the Rust engine only runs the result.
Train on a GPU: a 32³ shape runs at several seconds per iteration on a laptop CPU, so the
8000 iterations below are only practical on an NVIDIA GPU (one per shape).

### 1. Set up a GPU machine

```sh
python -m venv ~/venvs/nca && source ~/venvs/nca/bin/activate
pip install -r training/requirements.txt "jax[cuda12]"
python -c "import jax; print(jax.devices())"     # must list a CudaDevice, not CpuDevice
```

If it prints `CpuDevice`, the CUDA wheel didn't match the driver; see the
[JAX install guide](https://docs.jax.dev/en/latest/installation.html).

### 2. Pick the shapes

```sh
python training/targets.py --list
python training/targets.py tree --size 32 --preview       # prints a side view to check it
```

Presets: `bunny`, `tree`, `mushroom`, `cactus`, `torus`, `sphere`. Any mesh works too:
`python training/targets.py mesh model.obj --size 32 --out training/targets/model32.npy`,
then pass the `.npy` path instead of a preset name. New procedural presets are a function
in `training/targets.py` returning an inside mask and colors, added to `PRESETS`.

### 3. Train

One machine, every preset, spread across all its GPUs:

```sh
training/train_all.sh                                    # bunny tree mushroom cactus torus sphere
TARGETS="tree cactus" ROTATION=full training/train_all.sh
```

A SLURM cluster, one job per shape:

```sh
sbatch --array=0-5 training/cluster/train.sbatch
```

Both use the settings below and read `TARGETS`, `SIZE` (32), `ROTATION` (`yaw`),
`ITERATIONS` (8000) and `OUT` (`runs/`) from the environment. To run one shape by hand:

```sh
python training/train_nca.py tree --size 32 --rotation yaw --iterations 8000 \
  --batch 8 --damage 3 --hidden 128 --min-steps 64 --max-steps 96 \
  --checkpoint runs/tree.ckpt --out runs/tree.json --export-every 200
```

What the settings do, and why they're the defaults:

| Setting | Default | Why |
|---|---|---|
| `--size 32` | 32³ grid | Room for ears, branches and arms; about 24 voxels across the shape. |
| `--iterations 8000` | | The Distill paper's budget. Fewer leaves the shape unstable over long runs. |
| `--batch 8 --pool 256` | | Samples come from a pool and go back after each step, so the model trains on states it grew itself, including old ones. This is what teaches it to *stop* growing. |
| `--min-steps 64 --max-steps 96` | | Long rollouts per iteration; short ones grow the shape but don't hold it. |
| `--damage 3` | 3 of 8 | The best samples get spheres or slices cut out (`--damage-spheres`, `--damage-radius`, `--cut-prob`), so the model learns to regrow. |
| `--rotation yaw` | | Each sample grows at a random turn about the vertical axis, so one model works at any heading (the viewer's Orientation slider). `full` adds tilt and roll; it needs more iterations. `none` trains upright only. |
| `--hidden 128` | | Width of the update network. 64 trains faster; more helps detailed shapes. |
| `--remat` | off | Recomputes activations to save GPU memory. Add it if a run fails with out of memory. |

Runs checkpoint (weights, optimizer and pool), so rerunning the same command resumes, and
a preempted SLURM job requeues itself. Weights are written to `runs/<shape>.json` every 200
iterations, so a half-trained model can be opened in the viewer (Load neural CA weights)
while training goes on. Logs go to `logs/`.

### 4. Check that it holds its shape

A model can match the target at step 96 and still drift or swell over thousands of steps.
Measure it before shipping:

```sh
cargo run --release -p nc-core --example longrun runs/tree.json 3000
```

This grows the shape, carves a ball out halfway, and prints the cell count and the overlap
(IoU) with the step-150 shape over time. A good model stays near IoU 1 and recovers after
the cut; a count that keeps climbing means it needs more iterations or longer rollouts.

### 5. Ship it in the viewer

Copy the weights to `web/models/<shape>.json` and add a line to `web/models/index.json`:

```json
[
  { "model": "bunny", "label": "Bunny (neural CA, regenerates)" },
  { "model": "tree", "label": "Tree (neural CA)" }
]
```

The preset list, `#model=tree` links and `nca:tree` in the rule box then all work.

Models from other training code load too if exported in the
[weights format](docs/nca-weights.md); `training/export_torch_notebook.py` converts the
PyTorch notebook's models.

## Adding a rule family

1. Add a module under `crates/nc-core/src/rules/` with a struct implementing `Rule`.
2. Teach `nc_core::from_notation` (or a new constructor) how to build it.
3. The `Automaton` trait object, the wasm wrapper and the renderer need no changes.

## Deploying

Push to `main` on GitHub and enable Pages with **Settings → Pages → Source: GitHub Actions**.
The workflow runs the tests, builds the wasm package and publishes `web/`.
