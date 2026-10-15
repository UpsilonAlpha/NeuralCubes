"""Writes the reference fixture for nc-core's `neural_step_matches_training_code` test."""

import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

import train_nca as t

size, channels, hidden = 6, 8, 12
rng = np.random.default_rng(3)
params = {
    "w1": jnp.asarray(rng.normal(0, 0.3, (hidden, 4 * channels)), jnp.float32),
    "b1": jnp.asarray(rng.normal(0, 0.1, hidden), jnp.float32),
    "w2": jnp.asarray(rng.normal(0, 0.3, (channels, hidden)), jnp.float32),
    "b2": jnp.asarray(rng.normal(0, 0.1, channels), jnp.float32),
}
x = rng.uniform(0, 1, (1, size, size, size, channels)).astype(np.float32)
x[..., t.ALPHA] = np.where(rng.uniform(size=x.shape[:-1]) < 0.15, x[..., t.ALPHA], 0.0)  # sparse life
y = t.step(params, jnp.asarray(x), jax.random.PRNGKey(0), 1.0, 0.1)
doc = {
    "size": size,
    "weights": {"channels": channels, "hidden": hidden, "fire_rate": 1.0, "alive_threshold": 0.1,
                **{k: np.asarray(v).ravel().tolist() for k, v in params.items()}},
    "input": x.ravel().tolist(),
    "output": np.asarray(y).ravel().tolist(),
}
json.dump(doc, open(sys.argv[1], "w"))
print("alive in:", int((x[..., 3] > 0.1).sum()), "nonzero out cells:", int((np.abs(np.asarray(y)).sum(-1) > 0).sum()))
