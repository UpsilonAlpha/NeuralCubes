"""Export a model from the "Minimalist Neural Cellular Automata" PyTorch notebook
(3D version) to the Neural Cubes weights format.

Paste this into a notebook cell after training, then download the JSON and load it with
"Load neural CA weights" in the viewer (or drop it in web/models/):

    export_nca(results['final_model'], 'bunny_grow.json', name='bunny-grow',
               grid_shape=data['y'].shape[-3:])
"""

import json

import numpy as np


def export_nca(model, path, name="model", grid_shape=(27, 27, 27), fire_rate=0.5, alive_threshold=0.1):
    """Writes `model` (the notebook's `CA` module) as Neural Cubes NCA weights JSON."""
    perceive, process, output = model.update[0], model.update[1], model.update[3]
    as_np = lambda t: t.detach().cpu().numpy().astype(np.float32)

    C = model.state_dim
    K = perceive.out_channels // C        # kernels per channel (4: identity, dx, dy, dz)
    H = process.out_channels
    kernels = as_np(perceive.weight)[:K, 0]   # the grouped conv tiles the same K kernels per channel
    assert np.allclose(as_np(perceive.weight)[:, 0], np.tile(kernels, [C, 1, 1, 1])), "per-channel kernels differ"

    D, Hh, W = (int(v) for v in grid_shape)
    doc = {
        "format": "neural-cubes-nca/1",
        "name": name,
        "channels": C,
        "hidden": H,
        "fire_rate": fire_rate,             # the notebook updates where rand > 0.5
        "alive_threshold": alive_threshold,  # max_pool3d(alpha) > 0.1
        "grid": max(D, Hh, W),
        "seed": [D // 2, Hh // 2, W // 2],   # init_state sets the center cell...
        "seed_state": [1.0] * C,             # ...to 1 in every channel
        "kernels": [k.ravel().tolist() for k in kernels],  # [d][h][w] order, as stored
        "feature_order": "channel_major",    # grouped Conv3d output is c * K + k
        "boundary": "zero",                  # Conv3d(padding=1) pads with zeros
        # Which kernels are the x, y, z gradients (tensor axes W, H, D), so the viewer can
        # rotate the model. The notebook's "dz" runs along H and "dy" along D, both flipped.
        "gradient_kernels": [1, 3, 2],
        "gradient_signs": [1.0, -1.0, -1.0],
        "w1": as_np(process.weight).reshape(H, K * C).ravel().tolist(),
        "b1": as_np(process.bias).tolist(),
        "w2": as_np(output.weight).reshape(C, H).ravel().tolist(),
        "b2": (as_np(output.bias) if output.bias is not None else np.zeros(C, np.float32)).tolist(),
    }
    with open(path, "w") as f:
        json.dump(doc, f)
    print(f"wrote {path}: {C} channels, {K} kernels, hidden {H}, grid {D}x{Hh}x{W}")
    try:  # offer a download when running in Colab
        from google.colab import files
        files.download(path)
    except ImportError:
        pass
