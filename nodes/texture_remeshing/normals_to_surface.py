# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""RGB Normals + Mask to Surface.

Reconstruct a 3D height surface from an RGB normal map restricted to a mask, by
mask-aware sparse Poisson normal integration (the same gradient-domain family as
Bilateral Normal Integration used by Lotus's mesh demos). The integration is done
ONLY over masked pixels -- a graph-Laplacian Poisson system on the mask domain, so
no false gradients are injected outside the silhouette (the failure mode of naive
FFT / Frankot-Chellappa integration, which zero-pads).

Two solvers:
  * cpu_superlu : exact sparse direct solve (scipy SuperLU). Robust, deterministic.
  * gpu_cg      : Jacobi-preconditioned conjugate gradient on the GPU (torch). Much
                  faster on large masks; the system is SPD so CG converges.

Normals: a heightfield has only 2 gradient DOF (gx=-nx/nz, gy=+-ny/nz), and a unit
normal makes nz redundant. We read all three channels, normalise the vector (robust
to non-unit / predicted normals), force nz>0 (single-valued heightfield), then derive
the gradients. normal_z='recompute' ignores B and uses nz=sqrt(1-nx^2-ny^2) instead.

Adapted from the CAD-AF surface_image_nodes integrator. Output PER the mask: a TRIMESH
height surface, a normalised height IMAGE, and an info string.
"""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.array(x)


# --------------------------------------------------------------------------- #
# normal decoding                                                             #
# --------------------------------------------------------------------------- #
def _decode_normals(rgb, normal_z, flip_y):
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


def _normals_to_gradients(nx, ny, nz, mask):
    safe_nz = np.where(nz < 1e-6, 1e-6, nz)
    gx = (-nx / safe_nz) * mask                # dz/d(col)
    gy = (ny / safe_nz) * mask                 # dz/d(row); row 0 = top, y flipped up
    return gx, gy


# --------------------------------------------------------------------------- #
# mask-aware Poisson system (ported from CAD-AF surface_image_nodes)          #
# --------------------------------------------------------------------------- #
def _build_poisson_system(mask):
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


def _poisson_rhs(gx, gy, r_arr, c_arr, k_l, k_r, k_u, k_d, N):
    """Divergence RHS; edge gradients averaged at both endpoints (2nd-order)."""
    p_h = 0.5 * (gx[r_arr[k_l], c_arr[k_l]] + gx[r_arr[k_l], c_arr[k_l] + 1]).astype(np.float64)
    q_v = 0.5 * (gy[r_arr[k_u], c_arr[k_u]] + gy[r_arr[k_u] + 1, c_arr[k_u]]).astype(np.float64)
    rhs = np.zeros(N, dtype=np.float64)
    np.add.at(rhs, k_l, -p_h)
    np.add.at(rhs, k_r, p_h)
    np.add.at(rhs, k_u, -q_v)
    np.add.at(rhs, k_d, q_v)
    return rhs


def _pin_dirichlet(ii, jj, vv, rhs):
    """Symmetric Dirichlet pin at node 0 to fix the constant null space."""
    keep = (ii != 0) & (jj != 0)
    ii_f = np.concatenate([ii[keep], [0]])
    jj_f = np.concatenate([jj[keep], [0]])
    vv_f = np.concatenate([vv[keep], [1.0]])
    rhs = rhs.copy()
    rhs[0] = 0.0
    return ii_f, jj_f, vv_f, rhs


def _solve_cpu(ii, jj, vv, rhs, N):
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    ii_f, jj_f, vv_f, rhs = _pin_dirichlet(ii, jj, vv, rhs)
    A = coo_matrix((vv_f, (ii_f.astype(np.int32), jj_f.astype(np.int32))), shape=(N, N)).tocsr()
    z = spsolve(A, rhs)
    return (z - z.min()).astype(np.float32), {"solver": "cpu_superlu"}


def _solve_gpu_cg(ii, jj, vv, rhs, N, iters, tol):
    """Jacobi-preconditioned CG on the SPD pinned Laplacian. torch, CUDA if available."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ii_f, jj_f, vv_f, rhs = _pin_dirichlet(ii, jj, vv, rhs)
    # Jacobi preconditioner = 1 / diagonal
    diag = np.zeros(N, dtype=np.float64)
    dmask = ii_f == jj_f
    np.add.at(diag, ii_f[dmask].astype(np.int64), vv_f[dmask])
    diag[diag == 0.0] = 1.0

    idx = torch.tensor(np.stack([ii_f, jj_f]), dtype=torch.long, device=dev)
    val = torch.tensor(vv_f, dtype=torch.float32, device=dev)
    A = torch.sparse_coo_tensor(idx, val, (N, N)).coalesce()
    b = torch.tensor(rhs, dtype=torch.float32, device=dev)
    Minv = torch.tensor(1.0 / diag, dtype=torch.float32, device=dev)

    def spmv(v):
        return torch.sparse.mm(A, v.unsqueeze(1)).squeeze(1)

    x = torch.zeros(N, dtype=torch.float32, device=dev)
    r = b - spmv(x)
    z = Minv * r
    p = z.clone()
    rz = torch.dot(r, z)
    bnorm = torch.linalg.norm(b) + 1e-20
    it_done = 0
    for it in range(int(iters)):
        Ap = spmv(p)
        alpha = rz / (torch.dot(p, Ap) + 1e-20)
        x = x + alpha * p
        r = r - alpha * Ap
        it_done = it + 1
        if torch.linalg.norm(r) / bnorm < tol:
            break
        z = Minv * r
        rz_new = torch.dot(r, z)
        p = z + (rz_new / (rz + 1e-20)) * p
        rz = rz_new
    z_np = x.detach().cpu().numpy().astype(np.float32)
    z_np -= z_np.min()
    return z_np, {"solver": "gpu_cg", "device": str(dev), "iters": it_done,
                  "residual": float((torch.linalg.norm(r) / bnorm).item())}


# --------------------------------------------------------------------------- #
# core                                                                        #
# --------------------------------------------------------------------------- #
def _normals_to_surface(rgb, mask, *, solver, height_scale, normal_z, flip_y,
                        cg_iters, cg_tol):
    H, W = mask.shape
    m = mask > 0.5
    if not m.any():
        raise ValueError("Mask is empty -- nothing to integrate.")

    nx, ny, nz = _decode_normals(rgb, normal_z, flip_y)
    gx, gy = _normals_to_gradients(nx, ny, nz, m.astype(np.float64))

    N, r_arr, c_arr, idx_map, k_l, k_r, k_u, k_d, ii, jj, vv = _build_poisson_system(mask)
    rhs = _poisson_rhs(gx, gy, r_arr, c_arr, k_l, k_r, k_u, k_d, N)
    if solver == "gpu_cg":
        z_flat, sinfo = _solve_gpu_cg(ii, jj, vv, rhs, N, cg_iters, cg_tol)
    else:
        z_flat, sinfo = _solve_cpu(ii, jj, vv, rhs, N)

    h = np.zeros((H, W), dtype=np.float32)
    h[r_arr, c_arr] = z_flat

    # ---- build the surface mesh over the mask (vectorised, idx_map indexing) ----
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
    return mesh, hn, sinfo, (N, len(faces), lo, hi)


class NormalsToSurfaceNode(io.ComfyNode):
    """Reconstruct a height surface from an RGB normal map + mask (Poisson integration)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackNormalsToSurface",
            display_name="RGB Normals + Mask to Surface",
            category="geompack/texture_remeshing",
            is_output_node=True,
            description=(
                "Reconstruct a 3D height surface from an RGB NORMAL map + a MASK, via "
                "mask-aware sparse Poisson normal integration (the gradient-domain family "
                "used by Bilateral Normal Integration). Integration runs ONLY over masked "
                "pixels (graph-Laplacian Poisson on the mask domain) so no false gradients "
                "leak outside the silhouette -- unlike naive FFT integration. Outputs a "
                "TRIMESH surface, a normalised height IMAGE, and info.\n\n"
                "Solvers: cpu_superlu (exact sparse direct, robust) or gpu_cg (Jacobi-PCG on "
                "the GPU, much faster for big masks). A heightfield has only 2 gradient DOF "
                "(gx=-nx/nz, gy=ny/nz), so the Z/B channel is redundant -- we read it only to "
                "normalise the (possibly non-unit, e.g. predicted) normal and force a "
                "front-facing heightfield. normal_z='recompute' ignores B entirely."
            ),
            inputs=[
                io.Image.Input("normals",
                    tooltip="RGB normal map: R=nx, G=ny, B=nz encoded in [0,1] (=> [-1,1]). "
                            "Predicted (e.g. Lotus) normals work -- they're re-normalised."),
                io.Mask.Input("mask",
                    tooltip="Which pixels to reconstruct. Integration + mesh cover only "
                            "mask>0.5; everything else is excluded (silhouette = surface boundary)."),
                io.Combo.Input("solver", options=["cpu_superlu", "gpu_cg"], default="cpu_superlu",
                    tooltip="cpu_superlu: exact sparse direct solve (robust, deterministic). "
                            "gpu_cg: Jacobi-preconditioned conjugate gradient on the GPU "
                            "(torch) -- much faster on large masks; SPD system so it converges."),
                io.Float.Input("height_scale", default=1.0, min=0.001, max=100.0, step=0.01,
                    display_mode="number",
                    tooltip="Z multiplier. 1.0 keeps Z metrically proportional to X/Y (the "
                            "integration is already scale-correct on the mask domain); raise/"
                            "lower to exaggerate or flatten relief."),
                io.Combo.Input("normal_z", options=["use", "recompute"], default="use",
                    tooltip="use: read B and normalise the full (nx,ny,nz) vector (robust to "
                            "non-unit/predicted normals). recompute: ignore B, set "
                            "nz=sqrt(1-nx^2-ny^2) (assumes a clean unit normal map)."),
                io.Combo.Input("flip_y", options=["false", "true"], default="false",
                    tooltip="Flip the green (ny) channel: OpenGL vs DirectX normal-map "
                            "convention. If the surface comes out inverted top-to-bottom, "
                            "toggle this."),
                io.Int.Input("cg_iters", default=2000, min=10, max=100000, step=10,
                    tooltip="[gpu_cg] Max conjugate-gradient iterations. Big masks need more; "
                            "stops early at cg_tol."),
                io.Float.Input("cg_tol", default=1e-5, min=1e-8, max=1e-2, step=1e-6,
                    display_mode="number",
                    tooltip="[gpu_cg] Relative residual tolerance for early stop. Lower = more "
                            "accurate, slower."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="surface"),
                io.Image.Output(display_name="height_map"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, normals, mask, solver="cpu_superlu", height_scale=1.0,
                normal_z="use", flip_y="false", cg_iters=2000, cg_tol=1e-5):
        import torch

        rgb = _to_numpy(normals)
        if rgb.ndim == 4:                       # (B,H,W,C) -> first
            rgb = rgb[0]
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError(f"'normals' must be an RGB IMAGE (H,W,3); got shape {np.shape(normals)}")

        marr = _to_numpy(mask).astype(np.float32)
        if marr.ndim == 3:                      # (B,H,W) -> first
            marr = marr[0]
        elif marr.ndim == 4:
            marr = marr[0, :, :, 0]
        if marr.max() > 1.0:
            marr = marr / 255.0
        if marr.shape != rgb.shape[:2]:         # nearest-resize mask to the normal map
            ys = np.linspace(0, marr.shape[0] - 1, rgb.shape[0]).round().astype(int)
            xs = np.linspace(0, marr.shape[1] - 1, rgb.shape[1]).round().astype(int)
            marr = marr[np.ix_(ys, xs)]

        mesh, hn, sinfo, (nv, nf, lo, hi) = _normals_to_surface(
            rgb, marr, solver=solver, height_scale=height_scale, normal_z=normal_z,
            flip_y=flip_y, cg_iters=int(cg_iters), cg_tol=float(cg_tol))

        H, W = marr.shape
        height_img = torch.from_numpy(np.repeat(hn[:, :, None], 3, axis=2)[None].astype(np.float32))

        info = (
            f"RGB Normals + Mask to Surface\n\n"
            f"input: {W}x{H} | mask pixels: {nv:,} ({100*(marr>0.5).mean():.1f}%)\n"
            f"solver: {sinfo}\n"
            f"normal_z: {normal_z} | flip_y: {flip_y} | height_scale: {height_scale}\n"
            f"mesh: {nv:,} verts, {nf:,} faces\n"
            f"integrated height range (pre-scale): [{lo:.4g}, {hi:.4g}]\n"
            f"\nOutputs: surface (TRIMESH), height_map (IMAGE), info"
        )
        log.info("NormalsToSurface: %s, %d verts %d faces", sinfo, nv, nf)
        return io.NodeOutput(mesh, height_img, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackNormalsToSurface": NormalsToSurfaceNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackNormalsToSurface": "RGB Normals + Mask to Surface"}
