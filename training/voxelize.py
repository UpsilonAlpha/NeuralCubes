"""Voxelize a triangle mesh into an RGBA target grid for NCA training.

Inside/outside uses the generalized winding number, so meshes with small holes
(like the Stanford bunny's open base) still voxelize cleanly.

    python voxelize.py mesh.json --size 32 --fit 26 --out targets/bunny.npy

`mesh.json` holds {"positions": [[x, y, z], ...], "cells": [[i, j, k], ...]}
(the format of the `bunny` npm package). Output shape is (size, size, size, 4),
indexed [z, y, x, channel], with premultiplied RGB and alpha in 0..1.
"""

import argparse
import json

import numpy as np


def winding_number(points, tris, chunk=2048):
    """Generalized winding number of each point w.r.t. a triangle soup."""
    out = np.zeros(len(points))
    for s in range(0, len(points), chunk):
        p = points[s:s + chunk, None, None, :]  # (P, 1, 1, 3)
        v = tris[None] - p  # (P, T, 3, 3)
        a, b, c = v[..., 0, :], v[..., 1, :], v[..., 2, :]
        la, lb, lc = (np.linalg.norm(x, axis=-1) for x in (a, b, c))
        det = np.einsum("ptk,ptk->pt", a, np.cross(b, c))
        div = (la * lb * lc + np.einsum("ptk,ptk->pt", a, b) * lc
               + np.einsum("ptk,ptk->pt", b, c) * la + np.einsum("ptk,ptk->pt", c, a) * lb)
        out[s:s + chunk] = np.arctan2(det, div).sum(-1) / (2 * np.pi)
    return out


def height_color(t):
    """Warm gradient from the base (t=0) to the top (t=1)."""
    lo, hi = np.array([0.80, 0.45, 0.25]), np.array([0.98, 0.88, 0.70])
    return lo + (hi - lo) * t[..., None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--fit", type=int, default=26, help="voxels along the mesh's longest side")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mesh = json.load(open(args.mesh))
    verts = np.asarray(mesh["positions"], dtype=np.float64)
    tris = verts[np.asarray(mesh["cells"])]

    lo, hi = verts.min(0), verts.max(0)
    scale = args.fit / (hi - lo).max()
    center = (lo + hi) / 2
    n = args.size
    idx = np.arange(n) + 0.5 - n / 2
    z, y, x = np.meshgrid(idx, idx, idx, indexing="ij")
    pts = np.stack([x, y, z], -1).reshape(-1, 3) / scale + center

    inside = (winding_number(pts, tris) > 0.5).reshape(n, n, n)
    ys = np.nonzero(inside.any(axis=(0, 2)))[0]
    t = np.clip((np.arange(n) - ys.min()) / max(ys.max() - ys.min(), 1), 0, 1)
    rgb = np.broadcast_to(height_color(t)[None, :, None, :], (n, n, n, 3))
    alpha = inside.astype(np.float32)[..., None]
    target = np.concatenate([rgb * alpha, alpha], -1).astype(np.float32)
    np.save(args.out, target)
    print(f"{args.out}: {int(inside.sum())} filled voxels in a {n}^3 grid")


if __name__ == "__main__":
    main()
