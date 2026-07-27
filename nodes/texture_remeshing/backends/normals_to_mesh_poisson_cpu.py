# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Normals to Mesh - Poisson CPU backend. Mask-aware sparse Poisson normal
integration (graph Laplacian over mask pixels, Neumann BCs) solved exactly with
scipy's SuperLU direct solver. Robust and deterministic."""

import logging

import numpy as np
from comfy_api.latest import io

from ._normals_to_mesh_common import (
    parse_normals_input, parse_mask_input, decode_normals, normals_to_gradients,
    build_poisson_system, poisson_rhs, pin_dirichlet,
    build_surface_outputs, to_height_image, build_info,
)

log = logging.getLogger("geometrypack")


def solve_cpu(ii, jj, vv, rhs, N):
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    ii_f, jj_f, vv_f, rhs = pin_dirichlet(ii, jj, vv, rhs)
    A = coo_matrix((vv_f, (ii_f.astype(np.int32), jj_f.astype(np.int32))), shape=(N, N)).tocsr()
    z = spsolve(A, rhs)
    return (z - z.min()).astype(np.float32), {"solver": "poisson_cpu (SuperLU)"}


class NormalsToMeshPoissonCPUNode(io.ComfyNode):
    """Poisson CPU (SuperLU) backend for Normals to Mesh."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackNormalsToMesh_PoissonCPU",
            display_name="Normals to Mesh Poisson CPU (backend)",
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
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="surface"),
                io.Image.Output(display_name="height_map"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, normals=None, mask=None, height_scale=1.0,
                normal_z="recompute", flip_y="false", max_slope_deg=87.0):
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
        z_flat, sinfo = solve_cpu(ii, jj, vv, rhs, N)

        mesh, hn, lo, hi, nf = build_surface_outputs(z_flat, marr, r_arr, c_arr, idx_map, N, height_scale)
        H, W = marr.shape
        info = build_info(W, H, N, nf, marr, sinfo, normal_z, flip_y, height_scale, lo, hi)
        log.info("NormalsToMesh[poisson_cpu]: %s, %d verts %d faces", sinfo, N, nf)
        return io.NodeOutput(mesh, to_height_image(hn), info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackNormalsToMesh_PoissonCPU": NormalsToMeshPoissonCPUNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackNormalsToMesh_PoissonCPU": "Normals to Mesh Poisson CPU (backend)"}
