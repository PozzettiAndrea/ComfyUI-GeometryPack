# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Create normal field visualization for VTK viewer.
"""

import logging

import numpy as np
import trimesh
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class VisualizNormalFieldNode(io.ComfyNode):
    """
    Create normal field visualization for VTK viewer.

    Adds vertex normals as scalar fields (X, Y, Z components) that can be
    visualized with color mapping in the VTK viewer. Useful for debugging
    normal orientation issues or understanding surface geometry.
    """


    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackVisualizeNormalField",
            display_name="Visualize Normal Field",
            category="geompack/repair",
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh_with_fields"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh):
        """
        Add normal components as vertex scalar fields.

        Args:
            trimesh: Input trimesh.Trimesh object

        Returns:
            tuple: (mesh_with_normal_fields, info_string)
        """
        log.info("Processing mesh with %d vertices", len(trimesh.vertices))

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

        log.info("Added 4 scalar fields to mesh")

        return io.NodeOutput(result_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {
    "GeomPackVisualizeNormalField": VisualizNormalFieldNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackVisualizeNormalField": "Visualize Normal Field",
}
