# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Connected Components Node - Label disconnected mesh parts with part_id field.

Uses trimesh's graph.connected_components() to identify disconnected regions
and assigns a unique part_id to each face based on which component it belongs to.

Supports batch processing: input a list of meshes, get a list of results.
"""

import os
import numpy as np


class ConnectedComponentsNode:
    """
    Label disconnected mesh components with a part_id face attribute.

    Each disconnected region of the mesh gets a unique integer ID (0, 1, 2, ...).
    The part_id is stored as a face attribute that can be visualized with the
    field-based mesh preview nodes.

    Supports batch processing: input a list of meshes, get a list of results.
    """

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("trimesh", "component_summary")
    OUTPUT_IS_LIST = (True, False)  # TRIMESH is list, STRING is single summary
    OUTPUT_NODE = True  # Enable UI output for dynamic display
    FUNCTION = "label_components"
    CATEGORY = "geompack/analysis"

    def label_components(self, trimesh):
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
            print(f"[ConnectedComponents] {mesh_name_short}: {num_components} component(s)")
            for comp in component_details[:10]:  # Limit console output
                print(f"[ConnectedComponents]   #{comp['id']}: {comp['faces']:,} faces, {comp['vertices']:,} vertices")
            if len(component_details) > 10:
                print(f"[ConnectedComponents]   ... and {len(component_details) - 10} more components")

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

        print(f"[ConnectedComponents] Processed {len(meshes)} mesh(es)")

        # Return both outputs and UI data
        return {
            "result": (result_meshes, summary),
            "ui": {
                "component_data": ui_components
            }
        }


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeomPackConnectedComponents": ConnectedComponentsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackConnectedComponents": "Connected Components",
}
