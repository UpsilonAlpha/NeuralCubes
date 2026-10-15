"""Train a 3D neural cellular automaton to grow, persist and regenerate a voxel target.

    pip install -r requirements.txt            # add jax[cuda12] on a GPU machine
    python train_nca.py tree --out ../web/models/tree.json
    python train_nca.py targets/model32.npy --out model.json --rotation yaw --damage 2

The target is a preset name from `targets.py` (built at --size) or an .npy file it wrote.

The model is "Growing Neural Cellular Automata" (Mordvintsev et al., 2020) in 3D:
identity + Sobel perception, a two-layer 1x1x1 update network, stochastic updates and
alive masking on alpha. Training follows the paper's regenerating setup: a sample pool,
the worst sample reset to the seed, and damage (ablation) applied to the best samples.

Rotation training (--rotation yaw|full) gives every pool sample a random orientation.
Its target is rotated to match, and its cells see their Sobel gradients rotated into the
sample's own frame, so one model learns to grow and repair the shape at any
orientation. The viewer exposes the same rotation as an orientation control.

Checkpoints (--checkpoint) hold weights, optimizer state and the pool, so a cluster job
that gets preempted resumes where it stopped. The exported JSON is what
`nc_core::rules::NeuralCa` loads; see ../docs/nca-weights.md.
"""

import argparse
import json
import os
import pickle
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.scipy.ndimage import map_coordinates

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


def perceive(x, rot=None):
    """Kernel-major features (feature k*C + c). `rot` is (B, 3, 3): gradients are
    expressed in each sample's own frame as R^T @ [gx, gy, gz]."""
    feats = []
    for k in KERNELS:
        acc = 0.0
        for dz, dy, dx in OFFSETS:
            w = k[dz + 1, dy + 1, dx + 1]
            if w != 0.0:
                acc = acc + w * neighbor(x, dz, dy, dx)
        feats.append(acc)
    if rot is not None:
        grad = jnp.stack(feats[1:], -2)  # (B, Z, Y, X, 3, C), axes x, y, z
        grad = jnp.einsum("bji,bzyxjc->bzyxic", rot, grad)  # R^T g
        feats = [feats[0]] + [grad[..., i, :] for i in range(3)]
    return jnp.concatenate(feats, -1)


def alive(x, threshold):
    a = x[..., ALPHA:ALPHA + 1]
    m = a
    for off in OFFSETS:
        m = jnp.maximum(m, neighbor(a, *off))
    return m > threshold


def step(params, x, key, fire_rate, threshold, rot=None):
    pre = alive(x, threshold)
    h = jax.nn.relu(perceive(x, rot) @ params["w1"].T + params["b1"])
    dx = h @ params["w2"].T + params["b2"]
    fire = jax.random.uniform(key, x.shape[:-1] + (1,)) < fire_rate
    x = x + dx * fire
    post = alive(x, threshold)
    return x * (pre & post)


@partial(jax.jit, static_argnames=("steps", "fire_rate", "threshold"))
def rollout(params, x, key, rot, steps, fire_rate, threshold):
    body = lambda carry, k: (step(params, carry, k, fire_rate, threshold, rot), None)
    if REMAT:
        body = jax.checkpoint(body)  # trade compute for memory on large grids
    x, _ = jax.lax.scan(body, x, jax.random.split(key, steps))
    return x


# ---------------------------------------------------------------------------
# Rotations


def random_rotation(rng, mode):
    """A rotation matrix acting on (x, y, z) vectors, y up."""
    if mode == "none":
        return np.eye(3, dtype=np.float32)
    if mode == "yaw":
        t = rng.uniform(0, 2 * np.pi)
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)
    q = rng.normal(size=4)  # uniform over SO(3) via a random unit quaternion
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], np.float32)


@jax.jit
def rotate_target(target, rot):
    """target[z, y, x, c] rotated by `rot` about the grid center (trilinear)."""
    n = target.shape[0]
    c = (n - 1) / 2
    idx = jnp.arange(n, dtype=jnp.float32) - c
    z, y, x = jnp.meshgrid(idx, idx, idx, indexing="ij")
    p = jnp.stack([x, y, z], -1) @ rot  # R^T p, the source point, as row vectors
    coords = [p[..., 2] + c, p[..., 1] + c, p[..., 0] + c]
    return jnp.stack([map_coordinates(target[..., ch], coords, order=1, mode="constant")
                      for ch in range(target.shape[-1])], -1)


def rotate_point(p, rot, n):
    """Grid cell [z, y, x] rotated about the center, rounded and clamped."""
    c = (n - 1) / 2
    v = rot @ (np.array([p[2], p[1], p[0]], np.float64) - c) + c
    return tuple(int(np.clip(round(v[i]), 0, n - 1)) for i in (2, 1, 0))


# ---------------------------------------------------------------------------
# Damage (ablation)


def damage(rng, x, args):
    """Ablates one sample in place: random spheres, or a planar cut removing one side."""
    n = x.shape[0]
    zz, yy, xx = np.meshgrid(*[np.arange(n)] * 3, indexing="ij")
    live = np.argwhere(x[..., ALPHA] > 0.1)
    if len(live) == 0:
        return x
    if rng.uniform() < args.cut_prob:
        normal = rng.normal(size=3)
        normal /= np.linalg.norm(normal)
        c = live[rng.integers(len(live))]
        side = (zz - c[0]) * normal[0] + (yy - c[1]) * normal[1] + (xx - c[2]) * normal[2]
        x[side > 0] = 0.0
        return x
    for _ in range(rng.integers(1, args.damage_spheres + 1)):
        c = live[rng.integers(len(live))]
        r = rng.uniform(*args.damage_radius) * n
        x[(zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2 < r * r] = 0.0
    return x


# ---------------------------------------------------------------------------
# Model I/O


def init_params(key, channels, hidden):
    fan_in = 4 * channels
    return {
        "w1": jax.random.normal(key, (hidden, fan_in)) * np.sqrt(2.0 / fan_in) * 0.5,
        "b1": jnp.zeros(hidden),
        "w2": jnp.zeros((channels, hidden)),  # start as the identity update
        "b2": jnp.zeros(channels),
    }


def params_from_json(doc):
    c, h = doc["channels"], doc["hidden"]
    return {"w1": jnp.array(doc["w1"]).reshape(h, 4 * c), "b1": jnp.array(doc["b1"]),
            "w2": jnp.array(doc["w2"]).reshape(c, h), "b2": jnp.array(doc["b2"])}


def seed_state(shape, channels, pos):
    x = np.zeros(shape + (channels,), np.float32)
    x[pos][ALPHA:] = 1.0
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
        "rotation": cfg["rotation"],  # orientations it was trained for: none, yaw or full
        "gradient_kernels": [1, 2, 3],  # Sobel x, y, z, so the engine can rotate them
        "w1": [round(float(v), 6) for v in w["w1"].ravel()],
        "b1": [round(float(v), 6) for v in w["b1"].ravel()],
        "w2": [round(float(v), 6) for v in w["w2"].ravel()],
        "b2": [round(float(v), 6) for v in w["b2"].ravel()],
        "trained_steps": cfg["iteration"],
        "loss": cfg["loss"],
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    os.replace(tmp, path)


def load_target(spec, size, fit):
    if spec.endswith(".npy"):
        return np.load(spec)
    import targets

    if spec not in targets.PRESETS:
        raise SystemExit(f"unknown target {spec!r}; presets: {', '.join(targets.PRESETS)}")
    return targets.to_rgba(*targets.PRESETS[spec][0](size, fit or size - 6))


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="preset name (see targets.py --list) or .npy target")
    ap.add_argument("--out", required=True, help="weights JSON to write")
    ap.add_argument("--name", help="model name shown in the viewer (default: target name)")
    ap.add_argument("--size", type=int, default=24, help="grid edge for preset targets")
    ap.add_argument("--fit", type=int, help="shape size in voxels for preset targets")
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--iterations", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--pool", type=int, default=256)
    ap.add_argument("--min-steps", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=56)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--rotation", choices=["none", "yaw", "full"], default="none",
                    help="train at random orientations: about the vertical axis, or any")
    ap.add_argument("--damage", type=int, default=2, help="samples ablated per batch (the best ones)")
    ap.add_argument("--damage-spheres", type=int, default=2, help="up to this many spheres per ablation")
    ap.add_argument("--damage-radius", type=float, nargs=2, default=[0.1, 0.3], metavar=("MIN", "MAX"),
                    help="sphere radius range as a fraction of the grid")
    ap.add_argument("--cut-prob", type=float, default=0.2, help="chance an ablation is a planar cut instead")
    ap.add_argument("--export-every", type=int, default=100)
    ap.add_argument("--checkpoint", help="checkpoint file; resumes from it if it exists")
    ap.add_argument("--resume", help="weights JSON to start from (weights only)")
    ap.add_argument("--remat", action="store_true", help="recompute activations to save memory (32^3 and up)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    global REMAT
    REMAT = args.remat
    if args.damage > args.batch - 1:
        ap.error("--damage must be less than --batch (one sample per batch is reset to the seed)")
    print("devices:", jax.devices(), flush=True)

    target_np = load_target(args.target, args.size, args.fit).astype(np.float32)
    target = jnp.asarray(target_np)
    size = target.shape[0]
    filled = np.argwhere(target_np[..., ALPHA] > 0.5)
    centroid = filled.mean(0)
    seed_pos = tuple(int(v) for v in filled[np.argmin(((filled - centroid) ** 2).sum(1))])
    name = args.name or os.path.splitext(os.path.basename(args.target))[0]
    cfg = dict(name=name, channels=args.channels, hidden=args.hidden, fire_rate=0.5, threshold=0.1,
               size=size, seed=seed_pos, rotation=args.rotation, iteration=0, loss=None)

    schedule = optax.piecewise_constant_schedule(args.lr, {args.iterations // 2: 0.3, args.iterations * 3 // 4: 0.3})
    normalize = optax.GradientTransformation(
        lambda _: optax.EmptyState(),
        lambda g, s, p=None: (jax.tree.map(lambda x: x / (jnp.linalg.norm(x) + 1e-8), g), s),
    )
    opt = optax.chain(normalize, optax.adam(schedule))

    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    params = init_params(jax.random.PRNGKey(args.seed + 1), args.channels, args.hidden)
    if args.resume:
        params = params_from_json(json.load(open(args.resume)))
    opt_state = opt.init(params)
    rots = np.stack([random_rotation(rng, args.rotation) for _ in range(args.pool)])

    def seed_for(rot):
        return seed_state((size,) * 3, args.channels, rotate_point(seed_pos, rot, size))

    pool = np.stack([seed_for(r) for r in rots])
    if args.checkpoint and os.path.exists(args.checkpoint):
        with open(args.checkpoint, "rb") as f:
            ck = pickle.load(f)
        params, opt_state, pool, rots = ck["params"], ck["opt_state"], ck["pool"], ck["rots"]
        cfg["iteration"], rng = ck["iteration"], ck["rng"]
        print(f"resumed from {args.checkpoint} at iteration {cfg['iteration']}", flush=True)

    def per_sample_loss(x, tgt):
        return jnp.mean((x[..., :4] - tgt) ** 2, axis=(1, 2, 3, 4))

    @partial(jax.jit, static_argnames=("steps",))
    def train_step(params, opt_state, x, tgt, rot, key, steps):
        def loss_fn(p):
            y = rollout(p, x, key, rot, steps, 0.5, 0.1)
            l = per_sample_loss(y, tgt)
            return l.mean(), (y, l)
        (loss, (y, l)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, y, l, loss

    rotated = args.rotation != "none"
    lengths = np.linspace(args.min_steps, args.max_steps, 3).round().astype(int)
    t0, start = time.time(), cfg["iteration"]
    for it in range(cfg["iteration"], args.iterations):
        idx = rng.choice(args.pool, args.batch, replace=False)
        x, rot = pool[idx].copy(), rots[idx]
        if rotated:
            tgt = jnp.stack([rotate_target(target, jnp.asarray(r)) for r in rot])
        else:
            tgt = jnp.broadcast_to(target, (args.batch,) + target.shape)
        order = np.argsort(-np.asarray(per_sample_loss(jnp.asarray(x), tgt)))  # worst first
        x, idx, rot, tgt = x[order], idx[order], rot[order], tgt[order]
        x[0] = seed_for(rot[0])  # the worst sample restarts from the seed
        for i in range(1, args.damage + 1):
            damage(rng, x[-i], args)  # the best samples get ablated
        key, sub = jax.random.split(key)
        # A few fixed rollout lengths, so JAX compiles a handful of graphs rather than one per length.
        steps = int(rng.choice(lengths))
        rot_j = jnp.asarray(rot) if rotated else None
        params, opt_state, y, _, loss = train_step(params, opt_state, jnp.asarray(x), tgt, rot_j, sub, steps)
        pool[idx] = np.asarray(y)
        cfg["iteration"], cfg["loss"] = it + 1, float(loss)
        if (it + 1) % 10 == 0:
            print(f"iter {it + 1:5d}  loss {float(loss):.5f}  {(time.time() - t0) / (it + 1 - start):.2f}s/it", flush=True)
        if (it + 1) % args.export_every == 0 or it + 1 == args.iterations:
            export(params, cfg, args.out)
            if args.checkpoint:
                tmp = args.checkpoint + ".tmp"
                with open(tmp, "wb") as f:
                    pickle.dump({"params": params, "opt_state": opt_state, "pool": pool, "rots": rots,
                                 "iteration": it + 1, "rng": rng}, f)
                os.replace(tmp, args.checkpoint)
    print("done")


if __name__ == "__main__":
    main()
