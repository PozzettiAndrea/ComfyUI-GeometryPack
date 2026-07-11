# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Recompute mesh normals with custom settings.
"""

import logging

import numpy as np
import trimesh
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class ComputeNormalsNode(io.ComfyNode):
    """
    Recompute mesh normals with custom settings.

    Recalculates face and vertex normals. Useful after mesh manipulation,
    importing from formats without normals, or when normals seem incorrect.
    """


    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackComputeNormals",
            display_name="Compute Normals",
            category="geompack/repair",
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Combo.Input("smooth_vertex_normals", options=["true", "false"], default="true"),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh_with_normals"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, smooth_vertex_normals="true"):
        """
        Recompute mesh normals.

        Args:
            trimesh: Input trimesh.Trimesh object
            smooth_vertex_normals: Whether to smooth vertex normals

        Returns:
            tuple: (mesh_with_normals,)
        """
        log.info("Processing mesh with %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))

        # 'trimesh' here is the input-mesh argument (it shadows the top-level
        # `import trimesh`), so grab the module under another name for the helpers.
        import trimesh as tm

        result_mesh = trimesh.copy()
        result_mesh._cache.clear()
        face_normals = np.asarray(result_mesh.face_normals, dtype=np.float64)

        if smooth_vertex_normals == "false":
            # Faceted: per-vertex mean of adjacent face normals -- vectorized, no
            # Python loop. Prefer trimesh's helper; fall back to a numpy scatter.
            faces = np.asarray(result_mesh.faces)
            try:
                vertex_normals = np.asarray(
                    tm.geometry.mean_vertex_normals(
                        len(result_mesh.vertices), faces, face_normals),
                    dtype=np.float64)
            except Exception:
                vn = np.zeros((len(result_mesh.vertices), 3), dtype=np.float64)
                np.add.at(vn, faces.ravel(), np.repeat(face_normals, 3, axis=0))
                norms = np.linalg.norm(vn, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                vertex_normals = vn / norms
            result_mesh.metadata['normals_smoothed'] = False
            log.info("Computed faceted (mean) vertex normals")
        else:
            # Smooth: trimesh's angle-weighted vertex normals (also vectorized).
            vertex_normals = np.asarray(result_mesh.vertex_normals, dtype=np.float64)
            result_mesh.metadata['normals_smoothed'] = True
            log.info("Computed smooth vertex normals")

        # Store normals as vertex attributes for visualization: separate scalar
        # components (existing consumers expect these) plus a combined (n,3) field.
        result_mesh.vertex_attributes['normal_x'] = vertex_normals[:, 0]
        result_mesh.vertex_attributes['normal_y'] = vertex_normals[:, 1]
        result_mesh.vertex_attributes['normal_z'] = vertex_normals[:, 2]
        result_mesh.vertex_attributes['normal_magnitude'] = np.linalg.norm(vertex_normals, axis=1)
        result_mesh.vertex_attributes['normals_xyz'] = vertex_normals

        return io.NodeOutput(result_mesh)


NODE_CLASS_MAPPINGS = {
    "GeomPackComputeNormals": ComputeNormalsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackComputeNormals": "Compute Normals",
}
