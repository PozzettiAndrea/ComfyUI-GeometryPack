# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Geogram ABF++ (Angle-Based Flattening) UV unwrapping backend node.

Unlike libigl_lscm (Least Squares Conformal Maps, a convex least-squares-conformal
energy), ABF directly optimizes triangle ANGLES via a nonlinear solve to minimize
angle distortion -- a genuinely different objective from LSCM's energy, and
generally the higher-quality (if more expensive) parameterizer in Geogram's own
chart-flattening toolkit. Only the ABF parameterizer is exposed here; Geogram's
LSCM/spectral-LSCM modes are intentionally skipped as redundant with the existing
libigl_lscm backend, and its packer choice is fixed to XAtlas packing (matching
the quality of the existing dedicated xatlas backend's packing).
"""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class UVGeogramABFNode(io.ComfyNode):
    """Geogram ABF++ (Angle-Based Flattening) UV unwrapping backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackUV_GeogramABF",
            display_name="UV Geogram ABF (backend)",
            category="geompack/uv",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("hard_angles_threshold", default=45.0, min=1.0, max=179.0, step=1.0, tooltip=(
                    "Dihedral angle threshold, in degrees, used to decide where CHART "
                    "boundaries go before flattening even starts. An edge between two faces "
                    "whose dihedral (fold) angle exceeds this threshold is treated as a hard "
                    "feature and forced onto a chart seam -- so genuinely sharp edges (box "
                    "corners, hard mechanical edges) become chart boundaries rather than being "
                    "forced to flatten smoothly across them, which is where poor parameterizers "
                    "produce visible stretching/distortion. "
                    "HOW TO PICK A VALUE: the default, 45 degrees, treats moderately sharp "
                    "creases as chart boundaries -- a reasonable default for typical hard-surface "
                    "meshes. LOWER this (e.g. toward 20-30) if you want MORE chart boundaries / "
                    "smaller charts on a mesh with lots of gentle curvature you still want split "
                    "cleanly (trades more seams for less per-chart distortion). RAISE this (toward "
                    "90+) if you want FEWER, LARGER charts and only the most extreme creases "
                    "(e.g. near-90-degree box edges) to force a seam -- fewer charts means fewer "
                    "visible seam lines in a texture, but each chart has to stretch to cover more "
                    "curvature, so distortion within each chart goes up. On a mostly-smooth "
                    "organic mesh (a character, a rock), a high threshold with few/no hard edges "
                    "will produce few charts; on a mechanical/CAD-like mesh with lots of genuine "
                    "sharp edges, expect this to matter a lot more.")),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="unwrapped_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, hard_angles_threshold=45.0):
        import pygeogram

        log.info("Backend: geogram_abf")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))

        V = np.ascontiguousarray(trimesh.vertices, dtype=np.float64)
        F = np.ascontiguousarray(trimesh.faces, dtype=np.int32)

        # uvs: (M, 3, 2) -- UV per face corner, not per vertex. Geogram's own ABF++
        # (best-quality) parameterizer, packed via the XAtlas packer.
        uvs = pygeogram.make_atlas(
            V, F,
            hard_angles_threshold=float(hard_angles_threshold),
            parameterizer=pygeogram.PARAM_ABF,
            packer=pygeogram.PACK_XATLAS,
        )

        # Convert per-corner UVs to a valid per-vertex mesh: a vertex only needs to
        # be duplicated where it touches a chart seam (i.e. the SAME 3D position
        # ends up with DIFFERENT UVs on different faces). Dedup on (vertex_index, uv)
        # so shared, same-chart corners collapse back onto one output vertex --
        # same intent as xatlas.parametrize()'s vmapping, just done by hand since
        # make_atlas returns the more "raw" per-corner form.
        corner_vidx = F.reshape(-1).astype(np.float64)
        corner_uv = uvs.reshape(-1, 2)
        key = np.concatenate([corner_vidx[:, None], corner_uv], axis=1)
        uniq_keys, inverse = np.unique(key, axis=0, return_inverse=True)

        new_vertices = V[uniq_keys[:, 0].astype(np.int64)]
        new_uvs = uniq_keys[:, 1:3]
        new_faces = inverse.reshape(-1, 3).astype(np.int64)

        unwrapped = trimesh_module.Trimesh(vertices=new_vertices, faces=new_faces, process=False)

        from trimesh.visual import TextureVisuals
        unwrapped.visual = TextureVisuals(uv=new_uvs)

        unwrapped.metadata = trimesh.metadata.copy() if trimesh.metadata else {}
        unwrapped.metadata['uv_unwrap'] = {
            'algorithm': 'geogram_abf',
            'hard_angles_threshold': hard_angles_threshold,
            'original_vertices': len(trimesh.vertices),
            'unwrapped_vertices': len(new_vertices),
            'vertex_duplication_ratio': len(new_vertices) / max(1, len(trimesh.vertices)),
        }

        log.info("Output: %d vertices, %d faces", len(unwrapped.vertices), len(unwrapped.faces))

        info = f"""UV Unwrap Results (Geogram ABF++):

Algorithm: Angle-Based Flattening++ (nonlinear angle-distortion minimization)
Packing: XAtlas
Hard angle threshold: {hard_angles_threshold} deg

Input:  {len(trimesh.vertices):,} vertices, {len(trimesh.faces):,} faces
Output: {len(unwrapped.vertices):,} vertices, {len(unwrapped.faces):,} faces
Vertex duplication (chart seams): {len(unwrapped.vertices) / max(1, len(trimesh.vertices)):.2f}x

Distinct objective from libigl LSCM: minimizes ANGLE distortion via a nonlinear
solve rather than LSCM's convex least-squares-conformal energy.
"""
        return io.NodeOutput(unwrapped, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackUV_GeogramABF": UVGeogramABFNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackUV_GeogramABF": "UV Geogram ABF (backend)"}
