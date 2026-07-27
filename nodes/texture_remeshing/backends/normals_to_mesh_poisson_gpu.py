# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Normals to Mesh - Poisson GPU backend. Same mask-aware sparse Poisson system as
the CPU backend, solved with Jacobi-preconditioned conjugate gradient on the GPU
(torch; falls back to CPU tensors without CUDA). Much faster on large masks; the
pinned Laplacian is SPD so CG converges."""

import logging

import numpy as np
from comfy_api.latest import io

from ._normals_to_mesh_common import (
    parse_normals_input, parse_mask_input, decode_normals, normals_to_gradients,
    build_poisson_system, poisson_rhs, pin_dirichlet,
    build_surface_outputs, to_height_image, build_info,
)

log = logging.getLogger("geometrypack")


def solve_gpu_cg(ii, jj, vv, rhs, N, iters, tol):
    """Jacobi-preconditioned CG on the SPD pinned Laplacian. torch, CUDA if available."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ii_f, jj_f, vv_f, rhs = pin_dirichlet(ii, jj, vv, rhs)
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
    return z_np, {"solver": "poisson_gpu (Jacobi-PCG)", "device": str(dev), "iters": it_done,
                  "residual": float((torch.linalg.norm(r) / bnorm).item())}


class NormalsToMeshPoissonGPUNode(io.ComfyNode):
    """Poisson GPU (Jacobi-PCG) backend for Normals to Mesh."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackNormalsToMesh_PoissonGPU",
            display_name="Normals to Mesh Poisson GPU (backend)",
            category="geompack/texture_remeshing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Image.Input("normals"),
                io.Mask.Input("mask", optional=True),
                io.Float.Input("height_scale", default=1.0, min=0.001, max=100.0, step=0.01,
                               display_mode="number", optional=True),
                io.Combo.Input("normal_z", options=["recompute", "use"], default="recompute", optional=True),
                io.Combo.Input("flip_y", options=["false", "true"], default="false", optional=True),
                io.Float.Input("max_slope_deg", default=87.0, min=1.0, max=89.9, step=0.5,
                               display_mode="number", optional=True),
                io.Int.Input("cg_iters", default=2000, min=10, max=100000, step=10, optional=True),
                io.Float.Input("cg_tol", default=1e-5, min=1e-8, max=1e-2, step=1e-6,
                               display_mode="number", optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="surface"),
                io.Image.Output(display_name="height_map"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, normals=None, mask=None, height_scale=1.0,
                normal_z="recompute", flip_y="false", cg_iters=2000, cg_tol=1e-5, max_slope_deg=87.0):
        rgb = parse_normals_input(normals)
        if mask is not None:
            marr = parse_mask_input(mask, rgb.shape[:2])
        else:                       # no mask = reconstruct the whole frame
            marr = np.ones(rgb.shape[:2], dtype=np.float32)
        m = marr > 0.5
        if not m.any():
            raise ValueError("Mask is empty -- nothing to integrate.")

        nx, ny, nz = decode_normals(rgb, normal_z, flip_y)
        gx, gy = normals_to_gradients(nx, ny, nz, m.astype(np.float64), max_slope=float(np.tan(np.radians(float(max_slope_deg)))))

        N, r_arr, c_arr, idx_map, k_l, k_r, k_u, k_d, ii, jj, vv = build_poisson_system(marr)
        rhs = poisson_rhs(gx, gy, r_arr, c_arr, k_l, k_r, k_u, k_d, N)
        z_flat, sinfo = solve_gpu_cg(ii, jj, vv, rhs, N, int(cg_iters), float(cg_tol))

        mesh, hn, lo, hi, nf = build_surface_outputs(z_flat, marr, r_arr, c_arr, idx_map, N, height_scale)
        H, W = marr.shape
        info = build_info(W, H, N, nf, marr, sinfo, normal_z, flip_y, height_scale, lo, hi)
        log.info("NormalsToMesh[poisson_gpu]: %s, %d verts %d faces", sinfo, N, nf)
        return io.NodeOutput(mesh, to_height_image(hn), info, ui={"text": [info]})


# Backend disabled for now -- solver + node code above kept intact. To re-enable,
# uncomment these mappings plus the poisson_gpu lines in normals_to_mesh.py
# (BACKEND_MAP + DynamicCombo option) and texture_remeshing/__init__.py.
# NODE_CLASS_MAPPINGS = {"GeomPackNormalsToMesh_PoissonGPU": NormalsToMeshPoissonGPUNode}
# NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackNormalsToMesh_PoissonGPU": "Normals to Mesh Poisson GPU (backend)"}
