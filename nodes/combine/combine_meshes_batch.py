# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Combine Meshes from Batch Node - Concatenate a batch/list of meshes into one
"""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class CombineMeshesBatchNode(io.ComfyNode):
    """
    Combine Meshes from Batch - Concatenate a list of meshes into one.

    Takes a batch of meshes (list) and combines them into a single mesh.
    Simply concatenates vertices and faces without performing boolean operations.
    The result contains all geometry from input meshes as separate components.
    Useful for combining batch outputs from processing pipelines.
    """

    INPUT_IS_LIST = True


    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackCombineMeshesBatch",
            display_name="Combine Meshes (Batch)",
            category="geompack/combine",
            description='Combine a batch of meshes into a single mesh.',
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("meshes", tooltip="List of meshes to combine"),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="combined_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, meshes):
        """
        Combine a list of meshes into one.

        Args:
            meshes: List of trimesh objects

        Returns:
            tuple: (combined_mesh, info_string)
        """
        # Handle case where meshes is wrapped in another list due to ComfyUI batching
        if len(meshes) == 1 and isinstance(meshes[0], list):
            meshes = meshes[0]

        # Filter out None values
        meshes = [m for m in meshes if m is not None]

        if not meshes:
            raise ValueError("No valid meshes provided to combine")

        log.info("Combining %d meshes", len(meshes))

        # Track input stats
        input_stats = []
        total_vertices = 0
        total_faces = 0

        for i, mesh in enumerate(meshes):
            vertex_count = len(mesh.vertices) if hasattr(mesh, 'vertices') else 0
            face_count = len(mesh.faces) if hasattr(mesh, 'faces') else 0
            input_stats.append({
                'index': i + 1,
                'vertices': vertex_count,
                'faces': face_count
            })
            total_vertices += vertex_count
            total_faces += face_count
            log.info("Mesh %d: %d vertices, %d faces", i+1, vertex_count, face_count)

        # Concatenate meshes
        if len(meshes) == 1:
            result = meshes[0].copy()
        else:
            result = trimesh_module.util.concatenate(meshes)

        # Preserve metadata from first mesh
        if hasattr(meshes[0], 'metadata'):
            result.metadata = meshes[0].metadata.copy()
        else:
            result.metadata = {}

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

        # Calculate connected components
        try:
            num_components = len(trimesh_module.graph.connected_components(result.face_adjacency)[1])
        except Exception:
            num_components = len(meshes)  # Fallback estimate

        info = f"""Combine Meshes (Batch) Results:

Number of Meshes Combined: {len(meshes)}

Input Meshes:
{chr(10).join(mesh_lines)}

Combined Result:
  Total Vertices: {len(result.vertices):,}
  Total Faces: {len(result.faces):,}
  Connected Components: {num_components}

Note: Meshes are concatenated without boolean operations.
Components remain separate within the combined mesh.
"""

        log.info("Result: %d vertices, %d faces", len(result.vertices), len(result.faces))
        return io.NodeOutput(result, info, ui={"text": [info]})


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackCombineMeshesBatch": CombineMeshesBatchNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackCombineMeshesBatch": "Combine Meshes (Batch)",
}
