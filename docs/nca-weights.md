# Neural CA weights format

`nc_core::rules::NeuralCa` (and the "Load neural CA weights" button in the web viewer)
reads a JSON file with this shape:

```json
{
  "format": "neural-cubes-nca/1",
  "name": "bunny",
  "channels": 16,
  "hidden": 64,
  "fire_rate": 0.5,
  "alive_threshold": 0.1,
  "grid": 24,
  "seed": [12, 11, 12],
  "w1": [...], "b1": [...],
  "w2": [...], "b2": [...]
}
```

| Field | Meaning |
|---|---|
| `channels` | State channels per cell, `C`. 8, 12, 16, 24 or 32. Channels 0..2 are premultiplied RGB, 3 is alpha, the rest hidden. |
| `hidden` | Width of the hidden layer, `H`. |
| `fire_rate` | Probability that a cell applies its update on a step. |
| `alive_threshold` | A cell survives only if some cell in its 3x3x3 block has alpha above this, before and after the update. |
| `grid` | Edge of the training grid (optional). The viewer opens at this size. |
| `seed` | Seed cell as `[z, y, x]` in the training grid (optional, defaults to the center). Seeds get alpha and hidden channels set to 1. |
| `seed_state` | Full `C`-vector for the seed cell (optional). Default: alpha and hidden 1, RGB 0. |
| `kernels` | Perception kernels, each 27 values indexed `[dz][dy][dx]` (optional, default identity + Sobel/32). |
| `feature_order` | `kernel_major` (default, `k*C + c`) or `channel_major` (`c*K + k`, PyTorch grouped conv). |
| `boundary` | `wrap` (default, torus) or `zero` (PyTorch `padding=1`). |
| `w1` | `H x K*C`, row-major (`K` = number of kernels). |
| `b1` | `H`. |
| `w2` | `C x H`, row-major. |
| `b2` | `C`. |

## The update

For each cell, with `x` the `C`-vector state and the grid wrapping at its edges:

1. **Perceive.** Convolve every channel with four 3x3x3 kernels: identity, and Sobel
   along x, y and z. Sobel x is `d[dx] * s[dy] * s[dz] / 32` with `d = [-1, 0, 1]`,
   `s = [1, 2, 1]` (likewise for y and z). Features are **kernel-major**:
   `feature[k * C + c]` for kernel `k` in (identity, sobel x, sobel y, sobel z).
2. **Update.** `delta = w2 · relu(w1 · features + b1) + b2`.
3. **Fire.** Apply `x += delta` with probability `fire_rate`.
4. **Mask.** Zero cells whose 3x3x3 block has no alpha above `alive_threshold`, checked
   on the state before and after the update.

Axes: the grid is stored `[z][y][x]` and y is up in the viewer. In a PyTorch
`(N, C, D, H, W)` tensor that means D = z, H = y, W = x.

## Exporting from PyTorch

For the "Minimalist Neural Cellular Automata" notebook, use
`training/export_torch_notebook.py`: paste it into a cell and call
`export_nca(results['final_model'], 'bunny.json', grid_shape=data['y'].shape[-3:])`.
It copies the kernels straight from the frozen perception conv and sets
`feature_order: channel_major`, `boundary: zero` and an all-ones seed, so the engine
reproduces the notebook's update. `crates/nc-core` tests this against a NumPy
re-implementation of the notebook's step (`training/make_torch_fixture.py`).

Tensor axes `(D, H, W)` map to the engine's `(z, y, x)`, with y up in the viewer.
