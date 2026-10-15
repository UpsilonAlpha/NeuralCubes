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
| `w1` | `H x 4C`, row-major. |
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

Most growing-NCA notebooks perceive with a grouped convolution, which orders features
**channel-major** (`feature[c * 4 + k]`). Reorder the columns of the first layer when
exporting:

```python
import json, numpy as np

C, H = 16, 64
w1 = model.w1.weight.detach().cpu().numpy().reshape(H, C, 4)   # channel-major
w1 = w1.transpose(0, 2, 1).reshape(H, 4 * C)                   # to kernel-major
w2 = model.w2.weight.detach().cpu().numpy().reshape(C, H)
b1 = model.w1.bias.detach().cpu().numpy()
b2 = (model.w2.bias.detach().cpu().numpy() if model.w2.bias is not None else np.zeros(C))

json.dump({
    "format": "neural-cubes-nca/1", "name": "my-model",
    "channels": C, "hidden": H, "fire_rate": 0.5, "alive_threshold": 0.1,
    "grid": 32, "seed": [16, 16, 16],
    "w1": w1.ravel().tolist(), "b1": b1.tolist(),
    "w2": w2.ravel().tolist(), "b2": b2.tolist(),
}, open("my-model.json", "w"))
```

Check that the Sobel kernels, their order, their scaling and the padding mode (this
engine wraps) match the notebook. If they differ, the model will still run but won't
grow the shape it learned. `training/make_fixture.py` shows how the Rust test checks a
single step against the training code; the same idea works for a PyTorch model.
