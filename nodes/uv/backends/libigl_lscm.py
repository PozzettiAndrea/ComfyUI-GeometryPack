# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""libigl LSCM UV unwrapping backend node."""

import logging

import numpy as np
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class UVLibiglLSCMNode(io.ComfyNode):
    """libigl LSCM conformal mapping backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackUV_LibiglLSCM",
            display_name="UV libigl LSCM (backend)",
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
        """libigl LSCM conformal mapping."""
        try:
            import igl
        except ImportError:
            raise ImportError("libigl not installed (should be in requirements.txt)")

        log.info("Backend: libigl_lscm")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))

        # Fix 2 vertices for unique solution
        v_fixed = np.array([0, len(trimesh.vertices)-1], dtype=np.int32)
        uv_fixed = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)

        # Compute LSCM
        uv_result = igl.lscm(
            np.asarray(trimesh.vertices, dtype=np.float64),
            np.asarray(trimesh.faces, dtype=np.int32),
            v_fixed,
            uv_fixed
        )
        uv = uv_result[0] if isinstance(uv_result, tuple) else uv_result

        # Normalize to [0, 1]
        uv_min = uv.min(axis=0)
        uv_max = uv.max(axis=0)
        uv_range = uv_max - uv_min
        uv_range[uv_range < 1e-10] = 1.0
        uv_normalized = (uv - uv_min) / uv_range

        # Create unwrapped mesh (copy)
        unwrapped = trimesh.copy()
        from trimesh.visual import TextureVisuals
        unwrapped.visual = TextureVisuals(uv=uv_normalized)

        unwrapped.metadata['uv_unwrap'] = {
            'algorithm': 'libigl_lscm',
            'conformal': True,
            'angle_preserving': True,
            'fixed_vertices': v_fixed.tolist()
        }

        log.info("Output: %d vertices, %d faces", len(unwrapped.vertices), len(unwrapped.faces))

        info = f"""UV Unwrap Results (libigl LSCM):

Algorithm: Least Squares Conformal Maps
Properties: Angle-preserving, conformal mapping

Vertices: {len(trimesh.vertices):,}
Faces: {len(trimesh.faces):,}

No vertex duplication - preserves original topology.
Minimizes angle distortion for organic shapes.
"""
        return io.NodeOutput(unwrapped, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackUV_LibiglLSCM": UVLibiglLSCMNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackUV_LibiglLSCM": "UV libigl LSCM (backend)"}
