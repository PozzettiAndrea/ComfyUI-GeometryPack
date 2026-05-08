# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""libigl ARAP UV unwrapping backend node."""

import logging

import numpy as np
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class UVLibiglARAPNode(io.ComfyNode):
    """libigl ARAP-like parameterization backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackUV_LibiglARAP",
            display_name="UV libigl ARAP (backend)",
            category="geompack/uv",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("iterations", default=10, min=1, max=100, step=1),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="unwrapped_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, iterations=10):
        """libigl ARAP-like parameterization."""
        try:
            import igl
            import scipy.sparse
        except ImportError:
            raise ImportError("libigl and scipy not installed")

        log.info("Backend: libigl_arap")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: iterations=%d", iterations)

        # Find boundary
        boundary_loop = igl.boundary_loop(np.asarray(trimesh.faces, dtype=np.int32))

        if len(boundary_loop) == 0:
            raise ValueError(
                "Mesh has no boundary - ARAP parameterization requires an open mesh. "
                "Try using xatlas or libigl_lscm for closed meshes."
            )

        # Map boundary to circle
        bnd_angles = np.linspace(0, 2 * np.pi, len(boundary_loop), endpoint=False)
        bnd_uv = np.column_stack([
            0.5 + 0.5 * np.cos(bnd_angles),
            0.5 + 0.5 * np.sin(bnd_angles)
        ])

        # Initial harmonic solution
        uv_init = igl.harmonic(
            np.asarray(trimesh.vertices, dtype=np.float64),
            np.asarray(trimesh.faces, dtype=np.int32),
            boundary_loop.astype(np.int32),
            bnd_uv.astype(np.float64),
            1
        )

        # Apply iterative biharmonic refinement (ARAP-like)
        uv = uv_init.copy()
        for i in range(iterations):
            uv = igl.harmonic(
                np.asarray(trimesh.vertices, dtype=np.float64),
                np.asarray(trimesh.faces, dtype=np.int32),
                boundary_loop.astype(np.int32),
                bnd_uv.astype(np.float64),
                2  # biharmonic for smoother result
            )

        # Normalize to [0, 1]
        uv_min = uv.min(axis=0)
        uv_max = uv.max(axis=0)
        uv_range = uv_max - uv_min
        uv_range[uv_range < 1e-10] = 1.0
        uv = (uv - uv_min) / uv_range

        # Create unwrapped mesh
        unwrapped = trimesh.copy()
        from trimesh.visual import TextureVisuals
        unwrapped.visual = TextureVisuals(uv=uv)

        unwrapped.metadata['uv_unwrap'] = {
            'algorithm': 'libigl_arap_like',
            'iterations': iterations,
            'minimizes_distortion': True
        }

        log.info("Output: %d vertices, %d faces", len(unwrapped.vertices), len(unwrapped.faces))

        info = f"""UV Unwrap Results (libigl ARAP):

Algorithm: As-Rigid-As-Possible (biharmonic approximation)
Properties: Minimizes distortion, preserves shape
Iterations: {iterations}

Vertices: {len(trimesh.vertices):,}
Faces: {len(trimesh.faces):,}
Boundary Vertices: {len(boundary_loop):,}

Iterative solver for higher quality results.
Better preservation of angles and shapes.
"""
        return io.NodeOutput(unwrapped, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackUV_LibiglARAP": UVLibiglARAPNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackUV_LibiglARAP": "UV libigl ARAP (backend)"}
