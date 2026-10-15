"""Build RGBA voxel targets for NCA training: built-in presets or any triangle mesh.

    python targets.py --list
    python targets.py tree --size 32 --out targets/tree32.npy
    python targets.py mesh path/to/model.obj --size 32 --out targets/model32.npy
    python targets.py bunny --size 24 --fit 21 --out targets/bunny24.npy --preview

Output shape is (size, size, size, 4), indexed [z, y, x, channel] with y up, RGB
premultiplied by alpha, all in 0..1. This is the layout `train_nca.py` expects.
"""

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Meshes


def winding_number(points, tris, chunk=1024):
    """Generalized winding number of each point w.r.t. a triangle soup.

    Robust to small holes (the Stanford bunny's base is open), unlike ray parity.
    """
    out = np.zeros(len(points))
    for s in range(0, len(points), chunk):
        p = points[s:s + chunk, None, None, :]
        v = tris[None] - p  # (P, T, 3, 3)
        a, b, c = v[..., 0, :], v[..., 1, :], v[..., 2, :]
        la, lb, lc = (np.linalg.norm(x, axis=-1) for x in (a, b, c))
        det = np.einsum("ptk,ptk->pt", a, np.cross(b, c))
        div = (la * lb * lc + np.einsum("ptk,ptk->pt", a, b) * lc
               + np.einsum("ptk,ptk->pt", b, c) * la + np.einsum("ptk,ptk->pt", c, a) * lb)
        out[s:s + chunk] = np.arctan2(det, div).sum(-1) / (2 * np.pi)
    return out


def load_mesh(path):
    """Vertices and triangle indices from JSON ({"positions", "cells"}) or any format trimesh reads."""
    if path.endswith(".json"):
        m = json.load(open(path))
        return np.asarray(m["positions"], np.float64), np.asarray(m["cells"], np.int64)
    import trimesh  # pip install trimesh

    mesh = trimesh.load(path, force="mesh")
    return np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64)


def mesh_occupancy(path, size, fit):
    """Boolean [z, y, x] grid: the mesh scaled so its longest side spans `fit` voxels, centered."""
    verts, faces = load_mesh(path)
    tris = verts[faces]
    lo, hi = verts.min(0), verts.max(0)
    scale = fit / (hi - lo).max()
    idx = np.arange(size) + 0.5 - size / 2
    z, y, x = np.meshgrid(idx, idx, idx, indexing="ij")
    pts = np.stack([x, y, z], -1).reshape(-1, 3) / scale + (lo + hi) / 2
    return (winding_number(pts, tris) > 0.5).reshape(size, size, size)


def height_gradient(inside, lo=(0.80, 0.45, 0.25), hi=(0.98, 0.88, 0.70)):
    """Colors filled voxels with a gradient from the lowest to the highest filled row."""
    n = inside.shape[1]
    ys = np.nonzero(inside.any(axis=(0, 2)))[0]
    t = np.clip((np.arange(n) - ys.min()) / max(ys.max() - ys.min(), 1), 0, 1)
    lo, hi = np.array(lo), np.array(hi)
    rgb = lo + (hi - lo) * t[:, None]
    return np.broadcast_to(rgb[None, :, None, :], inside.shape + (3,))


# ---------------------------------------------------------------------------
# Procedural presets. Each takes centered coordinates in voxels (y up) and returns
# (inside mask, rgb array).


def _coords(size):
    idx = np.arange(size) + 0.5 - size / 2
    z, y, x = np.meshgrid(idx, idx, idx, indexing="ij")
    return x, y, z


def _paint(shape, *parts):
    """Later parts paint over earlier ones."""
    inside = np.zeros(shape, bool)
    rgb = np.zeros(shape + (3,))
    for mask, color in parts:
        inside |= mask
        rgb[mask] = color
    return inside, rgb


def tree(size, fit):
    x, y, z = _coords(size)
    s = fit / 26
    base = -13 * s
    r_xz = np.hypot(x, z)
    trunk = (r_xz < 1.6 * s) & (y > base) & (y < base + 12 * s)
    canopy = np.zeros_like(trunk)
    for cy, r in ((base + 12 * s, 8.5 * s), (base + 17 * s, 6.5 * s), (base + 21.5 * s, 4.5 * s)):
        canopy |= np.hypot(r_xz, (y - cy) * 1.4) < r
    return _paint(x.shape, (trunk, (0.45, 0.28, 0.15)), (canopy, (0.25, 0.62, 0.30)))


def mushroom(size, fit):
    x, y, z = _coords(size)
    s = fit / 26
    base = -13 * s
    r_xz = np.hypot(x, z)
    stem = (r_xz < 3.2 * s) & (y > base) & (y < base + 14 * s)
    cap = (np.hypot(r_xz, (y - (base + 13 * s)) * 1.5) < 12 * s) & (y > base + 12 * s)
    gills = cap & (y < base + 14 * s)
    return _paint(x.shape, (stem, (0.93, 0.89, 0.80)), (cap, (0.80, 0.18, 0.15)), (gills, (0.90, 0.80, 0.70)))


def cactus(size, fit):
    x, y, z = _coords(size)
    s = fit / 26
    base = -13 * s
    def column(cx, cz, r, y0, y1):
        m = (np.hypot(x - cx, z - cz) < r) & (y > y0) & (y < y1)
        return m | (np.hypot(np.hypot(x - cx, z - cz), y - y1) < r)  # rounded top
    body = column(0, 0, 3.4 * s, base, base + 22 * s)
    arm_l = column(-7 * s, 0, 2.2 * s, base + 9 * s, base + 17 * s) | (
        (np.abs(y - (base + 9 * s)) < 2.2 * s) & (np.abs(z) < 2.2 * s) & (x > -7 * s) & (x < 0))
    arm_r = column(7 * s, 0, 2.2 * s, base + 12 * s, base + 19 * s) | (
        (np.abs(y - (base + 12 * s)) < 2.2 * s) & (np.abs(z) < 2.2 * s) & (x < 7 * s) & (x > 0))
    pot = (np.maximum(np.abs(x), np.abs(z)) < 5 * s) & (y > base - 1) & (y < base + 4 * s)
    return _paint(x.shape, (body | arm_l | arm_r, (0.30, 0.58, 0.28)), (pot, (0.72, 0.38, 0.22)))


def torus(size, fit):
    x, y, z = _coords(size)
    s = fit / 26
    inside = np.hypot(np.hypot(x, z) - 8.5 * s, y) < 4.2 * s
    angle = (np.arctan2(z, x) / (2 * np.pi)) % 1.0
    rgb = np.stack([0.5 + 0.45 * np.cos(2 * np.pi * (angle + k / 3)) for k in range(3)], -1)
    return inside, rgb


def sphere(size, fit):
    x, y, z = _coords(size)
    s = fit / 26
    inside = np.sqrt(x**2 + y**2 + z**2) < 12 * s
    # Bands of latitude and a stripe of longitude, so the viewer shows which way is up and
    # how the shape is turned.
    lat = np.where(y > 4 * s, 1, np.where(y < -4 * s, 2, 0))
    rgb = np.stack([np.choose(lat, [0.95, 0.30, 0.95]), np.choose(lat, [0.85, 0.55, 0.45]),
                    np.choose(lat, [0.30, 0.85, 0.20])], -1)
    stripe = (np.abs(z) < 2.5 * s) & (x > 0)
    rgb[stripe] = (0.85, 0.20, 0.25)
    return inside, rgb


def bunny(size, fit):
    inside = mesh_occupancy(os.path.join(HERE, "meshes", "bunny.json"), size, fit)
    return inside, height_gradient(inside)


PRESETS = {
    "bunny": (bunny, "Stanford bunny (public domain mesh), warm height gradient"),
    "tree": (tree, "brown trunk with a stacked green canopy"),
    "mushroom": (mushroom, "white stem, red cap, pale gills"),
    "cactus": (cactus, "column with two arms in a clay pot"),
    "torus": (torus, "ring with a hue that wraps around it"),
    "sphere": (sphere, "ball with latitude bands and one red stripe, so turns are visible"),
}


def to_rgba(inside, rgb):
    alpha = inside.astype(np.float32)[..., None]
    return np.concatenate([np.asarray(rgb, np.float32) * alpha, alpha], -1)


def preview(target):
    """Prints a side view (max over z) of the target, y up."""
    a = target[..., 3].max(0)
    for y in range(a.shape[0] - 1, -1, -1):
        print("".join("#" if a[y, x] > 0.5 else "." for x in range(a.shape[1])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", nargs="?", help="a preset name, or `mesh` followed by a mesh path")
    ap.add_argument("mesh_path", nargs="?")
    ap.add_argument("--size", type=int, default=32, help="grid edge in voxels")
    ap.add_argument("--fit", type=int, help="voxels along the shape's longest side (default: size - 6)")
    ap.add_argument("--out")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--preview", action="store_true", help="print a side view")
    args = ap.parse_args()

    if args.list or not args.preset:
        for name, (_, desc) in PRESETS.items():
            print(f"{name:10s} {desc}")
        print(f"{'mesh':10s} any OBJ/PLY/STL/GLB (via trimesh) or {{positions, cells}} JSON")
        return
    fit = args.fit or args.size - 6
    if args.preset == "mesh":
        if not args.mesh_path:
            ap.error("mesh needs a path")
        inside = mesh_occupancy(args.mesh_path, args.size, fit)
        target = to_rgba(inside, height_gradient(inside))
    elif args.preset in PRESETS:
        target = to_rgba(*PRESETS[args.preset][0](args.size, fit))
    else:
        ap.error(f"unknown preset {args.preset!r}; see --list")
    out = args.out or os.path.join("targets", f"{args.preset}{args.size}.npy")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.save(out, target)
    print(f"{out}: {int((target[..., 3] > 0.5).sum())} filled voxels in a {args.size}^3 grid")
    p = np.argwhere(target[..., 3] > 0.5) - (args.size - 1) / 2
    print(f"reaches {np.abs(p).max() + 0.5:.1f} cells from the center upright, "
          f"{max(np.hypot(p[:, 0], p[:, 2]).max(), np.abs(p[:, 1]).max()) + 0.5:.1f} at any yaw, "
          f"{np.sqrt((p ** 2).sum(1)).max() + 0.5:.1f} at any orientation (half the grid is {args.size / 2:g}); "
          "train_nca.py needs --margin cells to spare")
    if args.preview:
        preview(target)


if __name__ == "__main__":
    main()
