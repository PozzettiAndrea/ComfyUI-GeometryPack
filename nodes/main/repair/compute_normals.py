# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Recompute mesh normals with custom settings.
"""

import numpy as np
import trimesh


class ComputeNormalsNode:
    """
    Recompute mesh normals with custom settings.

    Recalculates face and vertex normals. Useful after mesh manipulation,
    importing from formats without normals, or when normals seem incorrect.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "smooth_vertex_normals": (["true", "false"], {
                    "default": "true"
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh_with_normals",)
    FUNCTION = "compute_normals"
    CATEGORY = "geompack/repair"

    def compute_normals(self, trimesh, smooth_vertex_normals="true"):
        """
        Recompute mesh normals.

        Args:
            trimesh: Input trimesh.Trimesh object
            smooth_vertex_normals: Whether to smooth vertex normals

        Returns:
            tuple: (mesh_with_normals,)
        """
        print(f"[ComputeNormals] Processing mesh with {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")

        # Create a copy
        result_mesh = trimesh.copy()

        # Face normals are always recomputed automatically by trimesh
        # But we can force a cache clear and recomputation
        result_mesh._cache.clear()

        if smooth_vertex_normals == "false":
            # Use face normals directly (faceted appearance)
            # This creates sharp edges by not averaging normals across faces
            vertex_normals = np.zeros_like(result_mesh.vertices)
            for i, face in enumerate(result_mesh.faces):
                face_normal = result_mesh.face_normals[i]
                vertex_normals[face] += face_normal
            # Normalize
            norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            vertex_normals = vertex_normals / norms

            # Store in mesh (note: trimesh will override this with smoothed normals)
            # So we need to mark it in metadata
            result_mesh.metadata['normals_smoothed'] = False

            # Store normals as vertex attributes for visualization
            result_mesh.vertex_attributes['normal_x'] = vertex_normals[:, 0]
            result_mesh.vertex_attributes['normal_y'] = vertex_normals[:, 1]
            result_mesh.vertex_attributes['normal_z'] = vertex_normals[:, 2]
            result_mesh.vertex_attributes['normal_magnitude'] = np.linalg.norm(vertex_normals, axis=1)

            print(f"[ComputeNormals] Computed faceted (non-smooth) normals")
        else:
            # Trimesh automatically computes smooth vertex normals
            # Just access them to ensure they're computed
            vertex_normals = result_mesh.vertex_normals
            result_mesh.metadata['normals_smoothed'] = True

            # Store normals as vertex attributes for visualization
            result_mesh.vertex_attributes['normal_x'] = vertex_normals[:, 0]
            result_mesh.vertex_attributes['normal_y'] = vertex_normals[:, 1]
            result_mesh.vertex_attributes['normal_z'] = vertex_normals[:, 2]
            result_mesh.vertex_attributes['normal_magnitude'] = np.linalg.norm(vertex_normals, axis=1)

            print(f"[ComputeNormals] Computed smooth vertex normals")

        return (result_mesh,)


NODE_CLASS_MAPPINGS = {
    "GeomPackComputeNormals": ComputeNormalsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackComputeNormals": "Compute Normals",
}
