# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Extract Largest Connected Component Node.

Uses trimesh's graph.connected_components() to identify disconnected regions
and returns only the largest one (by face count).

Supports batch processing: input a list of meshes, get a list of results.
"""

import logging
import os

import numpy as np
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class ExtractLargestComponentNode(io.ComfyNode):
    """
    Extract the largest connected component from a mesh.

    Removes all smaller disconnected parts, keeping only the component
    with the most faces.

    Supports batch processing: input a list of meshes, get a list of results.
    """

    INPUT_IS_LIST = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackExtractLargestComponent",
            display_name="Extract Largest Component",
            category="geompack/analysis",
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="trimesh", is_output_list=True),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh):
        import trimesh as trimesh_module

        meshes = trimesh if isinstance(trimesh, list) else [trimesh]

        result_meshes = []
        summary_lines = []

        for mesh in meshes:
            components = trimesh_module.graph.connected_components(
                mesh.face_adjacency,
                nodes=np.arange(len(mesh.faces))
            )

            num_components = len(components)
            mesh_name = mesh.metadata.get('file_name', 'mesh') if hasattr(mesh, 'metadata') else 'mesh'
            mesh_name_short = os.path.splitext(mesh_name)[0]

            if num_components <= 1:
                # Already a single component, return as-is
                summary_lines.append(f"{mesh_name_short}: 1 component, no filtering needed ({len(mesh.faces):,} faces)")
                log.info("%s: single component, passing through", mesh_name_short)
                result_meshes.append(mesh.copy())
                continue

            # Find largest component by face count
            largest_idx = max(range(num_components), key=lambda i: len(components[i]))
            largest_face_indices = components[largest_idx]

            # Extract submesh
            largest_faces = mesh.faces[largest_face_indices]
            unique_vertex_indices = np.unique(largest_faces.flatten())

            # Build vertex index remapping
            vertex_remap = np.full(len(mesh.vertices), -1, dtype=np.int64)
            vertex_remap[unique_vertex_indices] = np.arange(len(unique_vertex_indices))

            new_vertices = mesh.vertices[unique_vertex_indices]
            new_faces = vertex_remap[largest_faces]

            result_mesh = trimesh_module.Trimesh(
                vertices=new_vertices,
                faces=new_faces,
                process=False,
            )

            # Copy metadata
            if hasattr(mesh, 'metadata') and mesh.metadata:
                result_mesh.metadata.update(mesh.metadata)

            # Copy face attributes (remapped to new face indices)
            if hasattr(mesh, 'face_attributes'):
                for attr_name, attr_values in mesh.face_attributes.items():
                    if isinstance(attr_values, np.ndarray) and len(attr_values) == len(mesh.faces):
                        result_mesh.face_attributes[attr_name] = attr_values[largest_face_indices]

            # Copy vertex attributes (remapped to new vertex indices)
            if hasattr(mesh, 'vertex_attributes'):
                for attr_name, attr_values in mesh.vertex_attributes.items():
                    if isinstance(attr_values, np.ndarray) and len(attr_values) == len(mesh.vertices):
                        result_mesh.vertex_attributes[attr_name] = attr_values[unique_vertex_indices]

            total_faces = len(mesh.faces)
            kept_faces = len(largest_face_indices)
            removed_faces = total_faces - kept_faces

            summary = (
                f"{mesh_name_short}: kept largest of {num_components} components — "
                f"{kept_faces:,} faces kept, {removed_faces:,} removed "
                f"({kept_faces / total_faces * 100:.1f}% of original)"
            )
            summary_lines.append(summary)
            log.info("%s: %d components, kept largest (%d faces, removed %d)",
                     mesh_name_short, num_components, kept_faces, removed_faces)

            result_meshes.append(result_mesh)

        info = "\n\n".join(summary_lines)
        return io.NodeOutput(result_meshes, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {
    "GeomPackExtractLargestComponent": ExtractLargestComponentNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackExtractLargestComponent": "Extract Largest Component",
}
