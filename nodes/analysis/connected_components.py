# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Connected Components Node - Label disconnected mesh parts with part_id field.

Uses trimesh's graph.connected_components() to identify disconnected regions
and assigns a unique part_id to each face based on which component it belongs to.

Supports batch processing: input a list of meshes, get a list of results.
"""

import logging
import os

import numpy as np
from comfy_api.latest import io

log = logging.getLogger("geometrypack")

class ConnectedComponentsNode(io.ComfyNode):
    """
    Label disconnected mesh components with a part_id face attribute.

    Each disconnected region of the mesh gets a unique integer ID (0, 1, 2, ...).
    The part_id is stored as a face attribute that can be visualized with the
    field-based mesh preview nodes.

    Supports batch processing: input a list of meshes, get a list of results.
    """

    INPUT_IS_LIST = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackConnectedComponents",
            display_name="Connected Components",
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
        """
        Label each face with its connected component ID.

        Args:
            trimesh: Input trimesh object(s)

        Returns:
            dict with "result" tuple and "ui" data for display
        """
        import trimesh as trimesh_module

        # Handle batch input
        meshes = trimesh if isinstance(trimesh, list) else [trimesh]

        result_meshes = []
        summary_lines = []
        ui_components = []  # For dynamic UI display

        for mesh in meshes:
            # Get connected components using face adjacency
            # Returns list of arrays, each containing face indices for one component
            components = trimesh_module.graph.connected_components(
                mesh.face_adjacency,
                nodes=np.arange(len(mesh.faces))
            )

            num_components = len(components)

            # Create part_id array for all faces
            part_ids = np.zeros(len(mesh.faces), dtype=np.int32)

            # Collect detailed component info
            component_details = []
            for component_id, face_indices in enumerate(components):
                part_ids[face_indices] = component_id

                # Get vertices for this component
                component_faces = mesh.faces[face_indices]
                component_vertex_indices = np.unique(component_faces.flatten())
                num_vertices = len(component_vertex_indices)
                num_faces = len(face_indices)

                component_details.append({
                    "id": component_id,
                    "faces": num_faces,
                    "vertices": num_vertices,
                    "face_indices": face_indices.tolist() if num_faces < 10 else None
                })

            # Sort by face count descending
            component_details.sort(key=lambda x: x["faces"], reverse=True)

            # Get mesh name for summary
            mesh_name = mesh.metadata.get('file_name', 'mesh') if hasattr(mesh, 'metadata') else 'mesh'
            mesh_name_short = os.path.splitext(mesh_name)[0]

            # Build detailed summary string
            detail_lines = [f"{mesh_name_short}: {num_components} component(s)"]
            for comp in component_details:
                detail_lines.append(f"  #{comp['id']}: {comp['faces']:,} faces, {comp['vertices']:,} vertices")

            summary_lines.append("\n".join(detail_lines))

            # Store for UI
            ui_components.append({
                "mesh_name": mesh_name_short,
                "num_components": num_components,
                "total_faces": len(mesh.faces),
                "total_vertices": len(mesh.vertices),
                "components": component_details
            })

            # Print to console
            log.info("%s: %d component(s)", mesh_name_short, num_components)
            for comp in component_details[:10]:  # Limit console output
                log.info("  #%d: %s faces, %s vertices", comp['id'], f"{comp['faces']:,}", f"{comp['vertices']:,}")
            if len(component_details) > 10:
                log.info("  ... and %d more components", len(component_details) - 10)

            # Store as face attribute
            result_mesh = mesh.copy()
            result_mesh.face_attributes['part_id'] = part_ids

            # Also store in metadata for compatibility
            if not hasattr(result_mesh, 'metadata'):
                result_mesh.metadata = {}
            result_mesh.metadata['part_ids'] = part_ids
            result_mesh.metadata['num_components'] = num_components
            result_mesh.metadata['component_details'] = component_details

            result_meshes.append(result_mesh)

        # Create summary string
        summary = "\n\n".join(summary_lines)

        log.info("Processed %d mesh(es)", len(meshes))

        # Return both outputs and UI data
        return io.NodeOutput(result_meshes, summary, ui={ "text": [summary], "component_data": ui_components })

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeomPackConnectedComponents": ConnectedComponentsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackConnectedComponents": "Connected Components",
}
