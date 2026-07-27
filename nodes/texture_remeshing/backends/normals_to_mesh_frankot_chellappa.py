# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Normals to Mesh - Frankot-Chellappa backend. Full-frame least-squares normal
integration in the frequency domain (FFT, periodic BCs). O(N log N) -- much faster
than a sparse solve on large images, but it always integrates the WHOLE frame:
a mask (if given) only crops the output mesh afterwards, it does not confine the
integration domain. Gradients outside a silhouette therefore still influence the
result near the boundary -- for masked objects prefer the Poisson backends, which
integrate strictly on the mask domain."""

import logging

import numpy as np
from comfy_api.latest import io

from ._normals_to_mesh_common import (
    parse_normals_input, parse_mask_input, decode_normals, normals_to_gradients,
    build_surface_outputs, to_height_image, build_info,
)

log = logging.getLogger("geometrypack")


def frankot_chellappa(gx, gy):
    """Reconstruct height field from gradient fields via least-squares FFT integration."""
    rows, cols = gx.shape

    wx = 2.0 * np.pi * np.fft.fftfreq(cols)
    wy = 2.0 * np.pi * np.fft.fftfreq(rows)
    WX, WY = np.meshgrid(wx, wy)

    denom = WX ** 2 + WY ** 2
    denom[0, 0] = 1.0  # avoid divide-by-zero at DC

    Z = (-1j * WX * np.fft.fft2(gx) + -1j * WY * np.fft.fft2(gy)) / denom
    Z[0, 0] = 0.0  # zero DC = arbitrary global offset

    return np.real(np.fft.ifft2(Z)).astype(np.float32)


class NormalsToMeshFrankotChellappaNode(io.ComfyNode):
    """Frankot-Chellappa (full-frame FFT) backend for Normals to Mesh."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackNormalsToMesh_FrankotChellappa",
            display_name="Normals to Mesh Frankot-Chellappa (backend)",
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
        H, W = rgb.shape[:2]
        if mask is not None:
            marr = parse_mask_input(mask, (H, W))
        else:
            marr = np.ones((H, W), dtype=np.float32)
        m = marr > 0.5
        if not m.any():
            raise ValueError("Mask is empty -- nothing to reconstruct.")

        nx, ny, nz = decode_normals(rgb, normal_z, flip_y)
        # Full-frame gradients: FC integrates the whole frame regardless of mask.
        gx, gy = normals_to_gradients(nx, ny, nz, np.ones((H, W), dtype=np.float64), max_slope=float(np.tan(np.radians(float(max_slope_deg)))))

        h = frankot_chellappa(gx, gy)

        # enumerate mask pixels (mesh domain) and shift heights so min-in-mask == 0
        px = np.argwhere(m)
        N = len(px)
        r_arr, c_arr = px[:, 0], px[:, 1]
        idx_map = np.full((H, W), -1, dtype=np.int64)
        idx_map[r_arr, c_arr] = np.arange(N, dtype=np.int64)
        z_flat = h[r_arr, c_arr]
        z_flat = (z_flat - z_flat.min()).astype(np.float32)

        sinfo = {"solver": "frankot_chellappa (full-frame FFT)"}
        mesh, hn, lo, hi, nf = build_surface_outputs(z_flat, marr, r_arr, c_arr, idx_map, N, height_scale)
        info = build_info(W, H, N, nf, marr, sinfo, normal_z, flip_y, height_scale, lo, hi)
        log.info("NormalsToMesh[frankot_chellappa]: %d verts %d faces", N, nf)
        return io.NodeOutput(mesh, to_height_image(hn), info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackNormalsToMesh_FrankotChellappa": NormalsToMeshFrankotChellappaNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackNormalsToMesh_FrankotChellappa": "Normals to Mesh Frankot-Chellappa (backend)"}
