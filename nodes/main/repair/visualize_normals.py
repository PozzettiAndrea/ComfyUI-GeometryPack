# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Create normal field visualization for VTK viewer.
"""

import numpy as np
import trimesh


class VisualizNormalFieldNode:
    """
    Create normal field visualization for VTK viewer.

    Adds vertex normals as scalar fields (X, Y, Z components) that can be
    visualized with color mapping in the VTK viewer. Useful for debugging
    normal orientation issues or understanding surface geometry.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("mesh_with_fields", "info")
    FUNCTION = "visualize_normals"
    CATEGORY = "geompack/repair"

    def visualize_normals(self, trimesh):
        """
        Add normal components as vertex scalar fields.

        Args:
            trimesh: Input trimesh.Trimesh object

        Returns:
            tuple: (mesh_with_normal_fields, info_string)
        """
        print(f"[VisualizeNormals] Processing mesh with {len(trimesh.vertices)} vertices")

        # Create a copy
        result_mesh = trimesh.copy()

        # Get vertex normals
        normals = result_mesh.vertex_normals

        # Add each component as a scalar field
        result_mesh.vertex_attributes['normal_x'] = normals[:, 0].astype(np.float32)
        result_mesh.vertex_attributes['normal_y'] = normals[:, 1].astype(np.float32)
        result_mesh.vertex_attributes['normal_z'] = normals[:, 2].astype(np.float32)

        # Also add normal magnitude (should be ~1.0 for unit normals)
        normal_magnitude = np.linalg.norm(normals, axis=1).astype(np.float32)
        result_mesh.vertex_attributes['normal_magnitude'] = normal_magnitude

        info = f"""Normal Field Visualization:

Added Scalar Fields:
  • normal_x: X component of vertex normals ({normals[:, 0].min():.3f} to {normals[:, 0].max():.3f})
  • normal_y: Y component of vertex normals ({normals[:, 1].min():.3f} to {normals[:, 1].max():.3f})
  • normal_z: Z component of vertex normals ({normals[:, 2].min():.3f} to {normals[:, 2].max():.3f})
  • normal_magnitude: Length of normal vectors (avg: {normal_magnitude.mean():.6f})

Use VTK viewer with 'Preview Mesh (VTK with Fields)' to visualize
these scalar fields with color mapping!

Expected Values:
  • Components: -1.0 to 1.0
  • Magnitude: ~1.0 (unit normals)
"""

        print(f"[VisualizeNormals] Added 4 scalar fields to mesh")

        return (result_mesh, info)


NODE_CLASS_MAPPINGS = {
    "GeomPackVisualizeNormalField": VisualizNormalFieldNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackVisualizeNormalField": "Visualize Normal Field",
}
