"""Train a 3D neural cellular automaton to grow and regenerate a voxel target.

Runs on CPU with JAX (no GPU needed for a 32^3 target, just patience):

    pip install jax optax numpy
    python train_nca.py targets/bunny32.npy --out ../web/models/bunny.json

The architecture follows "Growing Neural Cellular Automata" (Mordvintsev et al., 2020),
lifted to 3D: identity + 3D Sobel perception, a two-layer 1x1x1 update network,
stochastic updates, and alive masking on the alpha channel. Training uses a sample
pool and spherical damage so the result regenerates after parts are removed.

The exported JSON is what `nc_core::rules::NeuralCa` loads; see `../docs/nca-weights.md`.
"""

import argparse
import json
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

REMAT = False  # set by --remat
ALPHA = 3  # channels 0..2 are premultiplied RGB, 3 is alpha, the rest are hidden


def perception_kernels():
    d = np.array([-1.0, 0.0, 1.0])
    s = np.array([1.0, 2.0, 1.0])
    ident = np.zeros((3, 3, 3))
    ident[1, 1, 1] = 1.0
    # Indexed [dz, dy, dx], each offset -1..1 stored at 0..2.
    sx = np.einsum("z,y,x->zyx", s, s, d) / 32.0
    sy = np.einsum("z,y,x->zyx", s, d, s) / 32.0
    sz = np.einsum("z,y,x->zyx", d, s, s) / 32.0
    return np.stack([ident, sx, sy, sz])


KERNELS = perception_kernels()
OFFSETS = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


def neighbor(x, dz, dy, dx):
    """x[z+dz, y+dy, x+dx] with wrap-around; x is (B, Z, Y, X, C)."""
    return jnp.roll(x, shift=(-dz, -dy, -dx), axis=(1, 2, 3))


def perceive(x):
    feats = []
    for k in KERNELS:
        acc = 0.0
        for dz, dy, dx in OFFSETS:
            w = k[dz + 1, dy + 1, dx + 1]
            if w != 0.0:
                acc = acc + w * neighbor(x, dz, dy, dx)
        feats.append(acc)
    return jnp.concatenate(feats, -1)  # kernel-major: feature k*C + c


def alive(x, threshold):
    a = x[..., ALPHA:ALPHA + 1]
    m = a
    for off in OFFSETS:
        m = jnp.maximum(m, neighbor(a, *off))
    return m > threshold


def step(params, x, key, fire_rate, threshold):
    pre = alive(x, threshold)
    h = jax.nn.relu(perceive(x) @ params["w1"].T + params["b1"])
    dx = h @ params["w2"].T + params["b2"]
    fire = jax.random.uniform(key, x.shape[:-1] + (1,)) < fire_rate
    x = x + dx * fire
    post = alive(x, threshold)
    return x * (pre & post)


@partial(jax.jit, static_argnames=("steps", "fire_rate", "threshold"))
def rollout(params, x, key, steps, fire_rate, threshold):
    body = lambda carry, k: (step(params, carry, k, fire_rate, threshold), None)
    if REMAT:
        body = jax.checkpoint(body)  # trade compute for memory on large grids
    x, _ = jax.lax.scan(body, x, jax.random.split(key, steps))
    return x


def init_params(key, channels, hidden):
    k1, _ = jax.random.split(key)
    fan_in = 4 * channels
    return {
        "w1": jax.random.normal(k1, (hidden, fan_in)) * np.sqrt(2.0 / fan_in) * 0.5,
        "b1": jnp.zeros(hidden),
        "w2": jnp.zeros((channels, hidden)),  # start as the identity update
        "b2": jnp.zeros(channels),
    }


def seed_state(shape, channels, pos):
    x = np.zeros(shape + (channels,), np.float32)
    x[pos][ALPHA:] = 1.0
    return x


def sphere_damage(rng, x, n):
    """Zeros a random sphere in each of the n samples."""
    size = x.shape[1]
    zz, yy, xx = np.meshgrid(*[np.arange(size)] * 3, indexing="ij")
    for i in range(n):
        alive_idx = np.argwhere(x[i, ..., ALPHA] > 0.1)
        if len(alive_idx) == 0:
            continue
        c = alive_idx[rng.integers(len(alive_idx))]
        r = rng.uniform(0.15, 0.35) * size
        x[i][(zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2 < r * r] = 0.0
    return x


def export(params, cfg, path):
    w = {k: np.asarray(v, np.float32) for k, v in params.items()}
    doc = {
        "format": "neural-cubes-nca/1",
        "name": cfg["name"],
        "channels": cfg["channels"],
        "hidden": cfg["hidden"],
        "fire_rate": cfg["fire_rate"],
        "alive_threshold": cfg["threshold"],
        "grid": cfg["size"],
        "seed": [int(v) for v in cfg["seed"]],  # [z, y, x]
        "w1": [round(float(v), 6) for v in w["w1"].ravel()],
        "b1": [round(float(v), 6) for v in w["b1"].ravel()],
        "w2": [round(float(v), 6) for v in w["w2"].ravel()],
        "b2": [round(float(v), 6) for v in w["b2"].ravel()],
        "trained_steps": cfg["iteration"],
        "loss": cfg["loss"],
    }
    with open(path, "w") as f:
        json.dump(doc, f, separators=(",", ":"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="bunny")
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--iterations", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--pool", type=int, default=128)
    ap.add_argument("--min-steps", type=int, default=48)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--damage", type=int, default=2, help="samples damaged per batch")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--export-every", type=int, default=100)
    ap.add_argument("--resume", help="weights JSON to continue from")
    ap.add_argument("--remat", action="store_true", help="recompute activations to save memory (32^3 and up)")
    args = ap.parse_args()

    global REMAT
    REMAT = args.remat
    target = jnp.asarray(np.load(args.target))
    size = target.shape[0]
    filled = np.argwhere(np.asarray(target[..., ALPHA]) > 0.5)
    centroid = filled.mean(0)
    seed_pos = tuple(filled[np.argmin(((filled - centroid) ** 2).sum(1))])
    cfg = dict(name=args.name, channels=args.channels, hidden=args.hidden, fire_rate=0.5,
               threshold=0.1, size=size, seed=seed_pos, iteration=0, loss=None)

    params = init_params(jax.random.PRNGKey(0), args.channels, args.hidden)
    if args.resume:
        doc = json.load(open(args.resume))
        c, h = doc["channels"], doc["hidden"]
        params = {"w1": jnp.array(doc["w1"]).reshape(h, 4 * c), "b1": jnp.array(doc["b1"]),
                  "w2": jnp.array(doc["w2"]).reshape(c, h), "b2": jnp.array(doc["b2"])}
        cfg["iteration"] = doc.get("trained_steps", 0)

    schedule = optax.piecewise_constant_schedule(args.lr, {args.iterations // 2: 0.3, args.iterations * 3 // 4: 0.3})
    normalize = optax.GradientTransformation(
        lambda _: optax.EmptyState(),
        lambda g, s, p=None: (jax.tree.map(lambda x: x / (jnp.linalg.norm(x) + 1e-8), g), s),
    )
    opt = optax.chain(normalize, optax.adam(schedule))
    opt_state = opt.init(params)

    def per_sample_loss(x):
        return jnp.mean((x[..., :4] - target[None]) ** 2, axis=(1, 2, 3, 4))

    @partial(jax.jit, static_argnames=("steps",))
    def train_step(params, opt_state, x, key, steps):
        def loss_fn(p):
            y = rollout(p, x, key, steps, 0.5, 0.1)
            l = per_sample_loss(y)
            return l.mean(), (y, l)
        (loss, (y, l)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, y, l, loss

    seed = seed_state((size,) * 3, args.channels, seed_pos)
    pool = np.repeat(seed[None], args.pool, 0)
    rng = np.random.default_rng(0)
    key = jax.random.PRNGKey(1)
    t0, start = time.time(), cfg["iteration"]
    for it in range(cfg["iteration"], args.iterations):
        idx = rng.choice(args.pool, args.batch, replace=False)
        x = pool[idx].copy()
        l = np.asarray(per_sample_loss(jnp.asarray(x)))
        order = np.argsort(-l)  # worst first
        x, idx = x[order], idx[order]
        x[0] = seed
        if args.damage:
            x[-args.damage:] = sphere_damage(rng, x[-args.damage:], args.damage)
        key, sub = jax.random.split(key)
        # A few fixed rollout lengths, so JAX compiles a handful of graphs rather than one per length.
        steps = int(rng.choice(np.linspace(args.min_steps, args.max_steps, 3).round().astype(int)))
        params, opt_state, y, _, loss = train_step(params, opt_state, jnp.asarray(x), sub, steps)
        pool[idx] = np.asarray(y)
        cfg["iteration"], cfg["loss"] = it + 1, float(loss)
        if (it + 1) % 10 == 0:
            print(f"iter {it + 1:5d}  loss {float(loss):.5f}  {(time.time() - t0) / (it + 1 - start):.2f}s/it", flush=True)
        if (it + 1) % args.export_every == 0 or it + 1 == args.iterations:
            export(params, cfg, args.out)
    print("done")


if __name__ == "__main__":
    main()
