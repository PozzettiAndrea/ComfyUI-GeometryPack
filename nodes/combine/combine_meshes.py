# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Combine Meshes Node - Concatenate multiple meshes into one
"""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class CombineMeshesNode(io.ComfyNode):
    """
    Combine Meshes - Concatenate multiple meshes into one.

    Simply concatenates vertices and faces without performing boolean operations.
    The result contains all geometry from input meshes as separate components.
    Useful for grouping objects or preparing batch operations.
    """


    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackCombineMeshes",
            display_name="Combine Meshes",
            category="geompack/combine",
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("mesh_a"),
                io.Custom("TRIMESH").Input("mesh_b", optional=True),
                io.Custom("TRIMESH").Input("mesh_c", optional=True),
                io.Custom("TRIMESH").Input("mesh_d", optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="combined_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, mesh_a, mesh_b=None, mesh_c=None, mesh_d=None):
        """
        Combine multiple meshes into one.

        Args:
            mesh_a: First mesh (required)
            mesh_b, mesh_c, mesh_d: Optional additional meshes

        Returns:
            tuple: (combined_mesh, info_string)
        """
        meshes = [mesh_a]
        if mesh_b is not None:
            meshes.append(mesh_b)
        if mesh_c is not None:
            meshes.append(mesh_c)
        if mesh_d is not None:
            meshes.append(mesh_d)

        log.info("Combining %d meshes", len(meshes))

        # Track input stats
        input_stats = []
        total_vertices = 0
        total_faces = 0

        for i, mesh in enumerate(meshes):
            input_stats.append({
                'index': i + 1,
                'vertices': len(mesh.vertices),
                'faces': len(mesh.faces)
            })
            total_vertices += len(mesh.vertices)
            total_faces += len(mesh.faces)
            log.info("Mesh %d: %d vertices, %d faces", i+1, len(mesh.vertices), len(mesh.faces))

        # Concatenate meshes
        if len(meshes) == 1:
            result = mesh_a.copy()
        else:
            result = trimesh_module.util.concatenate(meshes)

        # Preserve metadata from first mesh
        result.metadata = mesh_a.metadata.copy()
        result.metadata['combined'] = {
            'num_meshes': len(meshes),
            'input_stats': input_stats,
            'total_vertices': len(result.vertices),
            'total_faces': len(result.faces)
        }

        # Build info string
        mesh_lines = []
        for stat in input_stats:
            mesh_lines.append(f"  Mesh {stat['index']}: {stat['vertices']:,} vertices, {stat['faces']:,} faces")

        info = f"""Combine Meshes Results:

Number of Meshes Combined: {len(meshes)}

Input Meshes:
{chr(10).join(mesh_lines)}

Combined Result:
  Total Vertices: {len(result.vertices):,}
  Total Faces: {len(result.faces):,}
  Connected Components: {len(trimesh_module.graph.connected_components(result.face_adjacency)[1])}

Note: Meshes are concatenated without boolean operations.
Components remain separate within the combined mesh.
"""

        log.info("Result: %d vertices, %d faces", len(result.vertices), len(result.faces))
        return io.NodeOutput(result, info, ui={"text": [info]})


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackCombineMeshes": CombineMeshesNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackCombineMeshes": "Combine Meshes",
}
