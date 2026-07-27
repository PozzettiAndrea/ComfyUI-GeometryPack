# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Shared helpers for the Normals to Mesh backends.

A heightfield has only 2 gradient DOF (gx=-nx/nz, gy=+ny/nz), and a unit normal
makes nz redundant. Normals are decoded from RGB, re-normalised (robust to
non-unit / predicted normals), forced front-facing (single-valued heightfield),
then converted to gradients and integrated to a height field -- either by
mask-aware sparse Poisson (graph Laplacian over mask pixels, Neumann BCs at the
mask boundary, exact scale) or full-frame FFT least-squares (Frankot-Chellappa).
"""

import logging

import numpy as np
import trimesh as trimesh_module

log = logging.getLogger("geometrypack")


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.array(x)


# --------------------------------------------------------------------------- #
# input parsing                                                               #
# --------------------------------------------------------------------------- #
def parse_normals_input(normals):
    """ComfyUI IMAGE -> (H,W,3) float array."""
    rgb = _to_numpy(normals)
    if rgb.ndim == 4:                       # (B,H,W,C) -> first
        rgb = rgb[0]
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"'normals' must be an RGB IMAGE (H,W,3); got shape {np.shape(normals)}")
    return rgb


def parse_mask_input(mask, shape):
    """ComfyUI MASK/IMAGE -> (H,W) float mask, nearest-resized to `shape`."""
    marr = _to_numpy(mask).astype(np.float32)
    if marr.ndim == 3:                      # (B,H,W) -> first
        marr = marr[0]
    elif marr.ndim == 4:
        marr = marr[0, :, :, 0]
    if marr.max() > 1.0:
        marr = marr / 255.0
    if marr.shape != shape:                 # nearest-resize mask to the normal map
        ys = np.linspace(0, marr.shape[0] - 1, shape[0]).round().astype(int)
        xs = np.linspace(0, marr.shape[1] - 1, shape[1]).round().astype(int)
        marr = marr[np.ix_(ys, xs)]
    return marr


# --------------------------------------------------------------------------- #
# normal decoding                                                             #
# --------------------------------------------------------------------------- #
def decode_normals(rgb, normal_z, flip_y):
    """rgb: (H,W,3) in [0,1] (a standard normal map). Returns nx, ny, nz in [-1,1],
    each (H,W), normalised, with nz forced positive (heightfield)."""
    rgb = rgb.astype(np.float64)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    nx = rgb[:, :, 0] * 2.0 - 1.0
    ny = rgb[:, :, 1] * 2.0 - 1.0
    nz = rgb[:, :, 2] * 2.0 - 1.0
    if flip_y == "true":                       # OpenGL <-> DirectX green channel
        ny = -ny
    if normal_z == "recompute":
        nz = np.sqrt(np.clip(1.0 - nx * nx - ny * ny, 0.0, 1.0))
    # normalise the full vector (robust to non-unit / predicted normals)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    norm = np.where(norm < 1e-8, 1.0, norm)
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    nz = np.abs(nz)                            # single-valued heightfield: front-facing
    return nx, ny, nz


def normals_to_gradients(nx, ny, nz, mask, max_slope=20.0):
    """max_slope clamps |gradient| per axis. A grazing normal (nz ~ 0) otherwise
    yields a gradient of up to 1e6 -- the integrator faithfully turns that into a
    single-pixel spike shot towards infinity. A heightfield cannot represent
    near-vertical slopes anyway; 20 ~ 87 degrees."""
    safe_nz = np.where(nz < 1e-6, 1e-6, nz)
    gx = (-nx / safe_nz) * mask                # dz/d(col)
    gy = (ny / safe_nz) * mask                 # dz/d(row); row 0 = top, y flipped up
    if max_slope is not None and max_slope > 0:
        gx = np.clip(gx, -max_slope, max_slope)
        gy = np.clip(gy, -max_slope, max_slope)
    return gx, gy


# --------------------------------------------------------------------------- #
# mask-aware Poisson system                                                   #
# --------------------------------------------------------------------------- #
def build_poisson_system(mask):
    """Graph Laplacian (COO) + index map over mask pixels. Stencil clipping at the
    mask boundary = natural Neumann BC. Returns the COO (ii,jj,vv) of the symmetric
    Laplacian BEFORE the Dirichlet pin, plus edge index arrays for the RHS."""
    rows, cols = mask.shape
    m = mask > 0.5
    px = np.argwhere(m)
    N = len(px)
    r_arr, c_arr = px[:, 0], px[:, 1]
    idx_map = np.full((rows, cols), -1, dtype=np.int64)
    idx_map[r_arr, c_arr] = np.arange(N, dtype=np.int64)

    in_h = c_arr + 1 < cols
    k_cand_h = np.where(in_h)[0]
    k_l = k_cand_h[m[r_arr[k_cand_h], c_arr[k_cand_h] + 1]]
    k_r = idx_map[r_arr[k_l], c_arr[k_l] + 1]

    in_v = r_arr + 1 < rows
    k_cand_v = np.where(in_v)[0]
    k_u = k_cand_v[m[r_arr[k_cand_v] + 1, c_arr[k_cand_v]]]
    k_d = idx_map[r_arr[k_u] + 1, c_arr[k_u]]

    i_h = np.concatenate([k_l, k_l, k_r, k_r])
    j_h = np.concatenate([k_l, k_r, k_l, k_r])
    v_h = np.concatenate([np.ones(len(k_l)), -np.ones(len(k_l)),
                          -np.ones(len(k_l)), np.ones(len(k_l))])
    i_v = np.concatenate([k_u, k_u, k_d, k_d])
    j_v = np.concatenate([k_u, k_d, k_u, k_d])
    v_v = np.concatenate([np.ones(len(k_u)), -np.ones(len(k_u)),
                          -np.ones(len(k_u)), np.ones(len(k_u))])

    ii = np.concatenate([i_h, i_v])
    jj = np.concatenate([j_h, j_v])
    vv = np.concatenate([v_h, v_v])
    return N, r_arr, c_arr, idx_map, k_l, k_r, k_u, k_d, ii, jj, vv


def poisson_rhs(gx, gy, r_arr, c_arr, k_l, k_r, k_u, k_d, N):
    """Divergence RHS; edge gradients averaged at both endpoints (2nd-order)."""
    p_h = 0.5 * (gx[r_arr[k_l], c_arr[k_l]] + gx[r_arr[k_l], c_arr[k_l] + 1]).astype(np.float64)
    q_v = 0.5 * (gy[r_arr[k_u], c_arr[k_u]] + gy[r_arr[k_u] + 1, c_arr[k_u]]).astype(np.float64)
    rhs = np.zeros(N, dtype=np.float64)
    np.add.at(rhs, k_l, -p_h)
    np.add.at(rhs, k_r, p_h)
    np.add.at(rhs, k_u, -q_v)
    np.add.at(rhs, k_d, q_v)
    return rhs


def pin_dirichlet(ii, jj, vv, rhs):
    """Symmetric Dirichlet pin at node 0 to fix the constant null space."""
    keep = (ii != 0) & (jj != 0)
    ii_f = np.concatenate([ii[keep], [0]])
    jj_f = np.concatenate([jj[keep], [0]])
    vv_f = np.concatenate([vv[keep], [1.0]])
    rhs = rhs.copy()
    rhs[0] = 0.0
    return ii_f, jj_f, vv_f, rhs


# --------------------------------------------------------------------------- #
# mesh + outputs (shared by every backend)                                    #
# --------------------------------------------------------------------------- #
def build_surface_outputs(z_flat, mask, r_arr, c_arr, idx_map, N, height_scale):
    """From per-mask-pixel heights, build the surface mesh + normalised height image.
    Returns (mesh, height_image_np(H,W), lo, hi, n_faces)."""
    H, W = mask.shape
    m = mask > 0.5

    h = np.zeros((H, W), dtype=np.float32)
    h[r_arr, c_arr] = z_flat

    s = 2.0 / max(H, W)                         # isotropic xy/z scale
    x = (c_arr - W / 2.0) * s
    y = (H / 2.0 - r_arr) * s                   # y up
    zc = z_flat * s * float(height_scale)
    verts = np.stack([x, y, zc], axis=1).astype(np.float64)

    self_i = np.arange(N)
    rc1 = np.where(c_arr + 1 < W, idx_map[r_arr, np.clip(c_arr + 1, 0, W - 1)], -1)
    dn = np.where(r_arr + 1 < H, idx_map[np.clip(r_arr + 1, 0, H - 1), c_arr], -1)
    drc = np.where((r_arr + 1 < H) & (c_arr + 1 < W),
                   idx_map[np.clip(r_arr + 1, 0, H - 1), np.clip(c_arr + 1, 0, W - 1)], -1)
    k1 = (rc1 >= 0) & (dn >= 0)
    t1 = np.stack([self_i[k1], dn[k1], rc1[k1]], axis=1)
    k2 = (rc1 >= 0) & (drc >= 0) & (dn >= 0)
    t2 = np.stack([rc1[k2], dn[k2], drc[k2]], axis=1)
    faces = np.vstack([t1, t2]).astype(np.int64) if (k1.any() or k2.any()) else np.zeros((0, 3), np.int64)

    mesh = trimesh_module.Trimesh(vertices=verts, faces=faces, process=False)
    try:
        mesh.fix_normals()
    except Exception:
        pass

    # normalised height map (for preview), background 0
    hn = h.copy()
    hi, lo = float(h[m].max()), float(h[m].min())
    if hi > lo:
        hn = (h - lo) / (hi - lo)
    hn = hn * m
    return mesh, hn, lo, hi, len(faces)


def to_height_image(hn):
    """(H,W) float -> ComfyUI IMAGE tensor (1,H,W,3)."""
    import torch
    return torch.from_numpy(np.repeat(hn[:, :, None], 3, axis=2)[None].astype(np.float32))


def build_info(W, H, nv, nf, mask, sinfo, normal_z, flip_y, height_scale, lo, hi):
    return (
        f"Normals to Mesh\n\n"
        f"input: {W}x{H} | mask pixels: {nv:,} ({100*(mask>0.5).mean():.1f}%)\n"
        f"solver: {sinfo}\n"
        f"normal_z: {normal_z} | flip_y: {flip_y} | height_scale: {height_scale}\n"
        f"mesh: {nv:,} verts, {nf:,} faces\n"
        f"integrated height range (pre-scale): [{lo:.4g}, {hi:.4g}]\n"
        f"\nOutputs: surface (TRIMESH), height_map (IMAGE), info"
    )
