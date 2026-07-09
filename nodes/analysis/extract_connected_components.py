# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Extract Connected Components Node.

Uses trimesh's graph.connected_components() to identify disconnected regions,
ranks them by a chosen metric (triangle count, total area, or bounding-box
diagonal), and returns a selected slice of that ranking.

Supports batch processing: input a list of meshes, get a list of results.
"""

import logging
import os

import numpy as np
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class ExtractConnectedComponentsNode(io.ComfyNode):
    """
    Extract connected components from a mesh, ranked by size.

    Splits a mesh into its disconnected components, sorts them by the chosen
    metric, and keeps a selected range of that ranking. Defaults (sort by
    triangle count, highest first, start_index=0, max_components=1) reproduce
    the old "Extract Largest Component" behavior.

    Supports batch processing: input a list of meshes, get a list of results.
    """

    INPUT_IS_LIST = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackExtractConnectedComponents",
            display_name="Extract Connected Components",
            category="geompack/analysis",
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Combo.Input("sort_by", options=["num_triangles", "total_area", "bbox_diagonal"], default="num_triangles",
                                tooltip="Metric used to rank components."),
                io.Combo.Input("order", options=["highest_first", "lowest_first"], default="highest_first",
                                tooltip="Ranking direction."),
                io.Int.Input("start_index", default=0, min=0, max=100000,
                             tooltip="Skip the first N components in the ranking."),
                io.Int.Input("max_components", default=1, min=-1, max=100000,
                             tooltip="Keep up to N components after start_index (-1 = unlimited)."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="trimesh", is_output_list=True),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, sort_by, order, start_index, max_components):
        import trimesh as trimesh_module

        meshes = trimesh if isinstance(trimesh, list) else [trimesh]
        sort_by_val = sort_by[0] if isinstance(sort_by, list) else sort_by
        order_val = order[0] if isinstance(order, list) else order
        start_index_val = start_index[0] if isinstance(start_index, list) else start_index
        max_components_val = max_components[0] if isinstance(max_components, list) else max_components

        reverse = (order_val == "highest_first")

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

            def _metric(face_indices, mesh=mesh, sort_by_val=sort_by_val):
                if sort_by_val == "total_area":
                    return float(mesh.area_faces[face_indices].sum())
                if sort_by_val == "bbox_diagonal":
                    verts = mesh.vertices[np.unique(mesh.faces[face_indices])]
                    if len(verts) == 0:
                        return 0.0
                    return float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
                return len(face_indices)  # num_triangles

            ranked = sorted(range(num_components), key=lambda i: _metric(components[i]), reverse=reverse)

            if start_index_val >= num_components:
                summary_lines.append(
                    f"{mesh_name_short}: start_index ({start_index_val}) >= number of components "
                    f"({num_components}), skipping"
                )
                log.warning("%s: start_index %d >= %d components, skipping",
                            mesh_name_short, start_index_val, num_components)
                continue

            if max_components_val < 0:
                selected = ranked[start_index_val:]
            else:
                selected = ranked[start_index_val:start_index_val + max_components_val]

            total_faces = len(mesh.faces)

            for rank_pos, comp_idx in zip(range(start_index_val, start_index_val + len(selected)), selected):
                face_indices = components[comp_idx]

                faces = mesh.faces[face_indices]
                unique_vertex_indices = np.unique(faces.flatten())

                # Build vertex index remapping
                vertex_remap = np.full(len(mesh.vertices), -1, dtype=np.int64)
                vertex_remap[unique_vertex_indices] = np.arange(len(unique_vertex_indices))

                new_vertices = mesh.vertices[unique_vertex_indices]
                new_faces = vertex_remap[faces]

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
                            result_mesh.face_attributes[attr_name] = attr_values[face_indices]

                # Copy vertex attributes (remapped to new vertex indices)
                if hasattr(mesh, 'vertex_attributes'):
                    for attr_name, attr_values in mesh.vertex_attributes.items():
                        if isinstance(attr_values, np.ndarray) and len(attr_values) == len(mesh.vertices):
                            result_mesh.vertex_attributes[attr_name] = attr_values[unique_vertex_indices]

                kept_faces = len(face_indices)
                summary_lines.append(
                    f"{mesh_name_short}: component rank {rank_pos} of {num_components} "
                    f"({sort_by_val}, {order_val}) — {kept_faces:,}/{total_faces:,} faces "
                    f"({kept_faces / total_faces * 100:.1f}%)"
                )
                log.info("%s: rank %d/%d by %s (%s), %d faces",
                         mesh_name_short, rank_pos, num_components, sort_by_val, order_val, kept_faces)

                result_meshes.append(result_mesh)

        info = "\n\n".join(summary_lines) if summary_lines else "No components extracted."
        return io.NodeOutput(result_meshes, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {
    "GeomPackExtractConnectedComponents": ExtractConnectedComponentsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackExtractConnectedComponents": "Extract Connected Components",
}
