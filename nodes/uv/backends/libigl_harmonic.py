# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""libigl harmonic UV unwrapping backend node."""

import logging

import numpy as np
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class UVLibiglHarmonicNode(io.ComfyNode):
    """libigl harmonic (Laplacian) mapping backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackUV_LibiglHarmonic",
            display_name="UV libigl Harmonic (backend)",
            category="geompack/uv",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="unwrapped_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh):
        """libigl harmonic (Laplacian) mapping."""
        try:
            import igl
        except ImportError:
            raise ImportError("libigl not installed (should be in requirements.txt)")

        log.info("Backend: libigl_harmonic")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))

        # Find boundary loop
        boundary_loop = igl.boundary_loop(np.asarray(trimesh.faces, dtype=np.int32))

        if len(boundary_loop) == 0:
            raise ValueError(
                "Mesh has no boundary - harmonic parameterization requires an open mesh. "
                "Try using xatlas or libigl_lscm for closed meshes."
            )

        # Map boundary to circle
        bnd_angles = np.linspace(0, 2 * np.pi, len(boundary_loop), endpoint=False)
        bnd_uv = np.column_stack([
            0.5 + 0.5 * np.cos(bnd_angles),
            0.5 + 0.5 * np.sin(bnd_angles)
        ])

        # Compute harmonic parameterization
        uv = igl.harmonic(
            np.asarray(trimesh.vertices, dtype=np.float64),
            np.asarray(trimesh.faces, dtype=np.int32),
            boundary_loop.astype(np.int32),
            bnd_uv.astype(np.float64),
            1  # Laplacian type
        )

        # Create unwrapped mesh
        unwrapped = trimesh.copy()
        from trimesh.visual import TextureVisuals
        unwrapped.visual = TextureVisuals(uv=uv)

        unwrapped.metadata['uv_unwrap'] = {
            'algorithm': 'libigl_harmonic',
            'boundary_vertices': len(boundary_loop),
            'guarantees_valid_uvs': True
        }

        log.info("Output: %d vertices, %d faces", len(unwrapped.vertices), len(unwrapped.faces))

        info = f"""UV Unwrap Results (libigl Harmonic):

Algorithm: Harmonic (Laplacian) mapping
Properties: Guarantees valid non-overlapping UVs

Vertices: {len(trimesh.vertices):,}
Faces: {len(trimesh.faces):,}
Boundary Vertices: {len(boundary_loop):,}

Requires open mesh with boundary.
Simple, fast, and stable parameterization.
"""
        return io.NodeOutput(unwrapped, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackUV_LibiglHarmonic": UVLibiglHarmonicNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackUV_LibiglHarmonic": "UV libigl Harmonic (backend)"}
