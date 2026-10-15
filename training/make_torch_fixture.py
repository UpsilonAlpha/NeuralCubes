"""Writes a fixture checking that nc-core runs models from the PyTorch NCA notebook.

This is an independent NumPy re-implementation of one step of the notebook's `CA` module
(grouped Conv3d perception with zero padding, the notebook's Sobel kernels, channel-major
features, max_pool3d alive masking), with every cell firing.
"""

import json
import sys

import numpy as np

C, H, S = 16, 20, 7
identity = np.zeros((3, 3, 3)); identity[1, 1, 1] = 1
dx = np.array([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
               [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
               [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]) / 8
dy = np.array([[[1, 2, 1], [2, 4, 2], [1, 2, 1]],
               [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
               [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]]) / 8
dz = np.array([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
               [[2, 4, 2], [0, 0, 0], [-2, -4, -2]],
               [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]) / 8
kernels = np.stack([identity, dx, dy, dz])           # (4, 3, 3, 3) over (D, H, W)
conv0 = np.tile(kernels, [C, 1, 1, 1])               # (4C, 3, 3, 3); output o reads channel o // 4

rng = np.random.default_rng(5)
w1 = rng.normal(0, 0.2, (H, 4 * C)).astype(np.float32)   # Conv3d(4C, H, 1).weight squeezed
b1 = rng.normal(0, 0.1, H).astype(np.float32)
w2 = rng.normal(0, 0.2, (C, H)).astype(np.float32)       # Conv3d(H, C, 1).weight squeezed
b2 = rng.normal(0, 0.1, C).astype(np.float32)

x = rng.uniform(0, 1, (C, S, S, S)).astype(np.float32)   # one sample, axes [C, D, H, W]
x[3] = np.where(rng.uniform(size=x[3].shape) < 0.2, x[3], 0.0)

def correlate_zero(a, k):  # torch Conv3d(padding=1): cross-correlation with zero padding
    p = np.pad(a, 1)
    out = np.zeros_like(a)
    for i in range(3):
        for j in range(3):
            for l in range(3):
                out += k[i, j, l] * p[i:i + S, j:j + S, l:l + S]
    return out

def alive(a):  # max_pool3d(3, stride=1, padding=1) > 0.1
    p = np.pad(a, 1, constant_values=-np.inf)
    m = np.full_like(a, -np.inf)
    for i in range(3):
        for j in range(3):
            for l in range(3):
                m = np.maximum(m, p[i:i + S, j:j + S, l:l + S])
    return m > 0.1

pre = alive(x[3])
feats = np.stack([correlate_zero(x[o // 4], conv0[o]) for o in range(4 * C)])
h = np.maximum(np.einsum("hf,fdyx->hdyx", w1, feats) + b1[:, None, None, None], 0)
y = x + np.einsum("ch,hdyx->cdyx", w2, h) + b2[:, None, None, None]
y = y * pre * alive(y[3])

doc = {
    "size": S,
    "weights": {
        "channels": C, "hidden": H, "fire_rate": 1.0, "alive_threshold": 0.1,
        "kernels": [k.ravel().tolist() for k in kernels],
        "feature_order": "channel_major", "boundary": "zero",
        "seed_state": [1.0] * C,
        "w1": w1.ravel().tolist(), "b1": b1.tolist(), "w2": w2.ravel().tolist(), "b2": b2.tolist(),
    },
    # Engine cells are stored [z][y][x] with channels last, i.e. tensor axes (D, H, W, C).
    "input": x.transpose(1, 2, 3, 0).ravel().tolist(),
    "output": y.transpose(1, 2, 3, 0).ravel().tolist(),
}
json.dump(doc, open(sys.argv[1], "w"))
print("alive cells after:", int((y[3] > 0.1).sum()))
