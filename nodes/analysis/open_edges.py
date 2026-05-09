# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Open Edges Node - Detect faces with open/boundary edges.

Finds edges that belong to only one face (not shared) and marks the faces
that contain these edges. Useful for detecting holes and mesh boundaries.

Supports batch processing: input a list of meshes, get a list of results.
"""

import logging
import os

import numpy as np
from comfy_api.latest import io

log = logging.getLogger("geometrypack")

class OpenEdgesNode(io.ComfyNode):
    """
    Detect and label faces with open/boundary edges.

    A boundary edge is an edge that belongs to only one face (not shared).
    This node finds all such edges and marks the faces containing them.

    Supports batch processing: input a list of meshes, get a list of results.
    """

    INPUT_IS_LIST = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackOpenEdges",
            display_name="Open Edges",
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
        Find faces with open/boundary edges.

        Args:
            trimesh: Input trimesh object(s)

        Returns:
            dict with "result" tuple and "ui" data for display
        """
        import trimesh as trimesh_module
        from trimesh.grouping import group_rows

        # Handle batch input
        meshes = trimesh if isinstance(trimesh, list) else [trimesh]

        result_meshes = []
        summary_lines = []
        ui_data = []  # For dynamic UI display

        for mesh in meshes:
            # Get edges
            edges_sorted = mesh.edges_sorted

            # Find boundary edges (edges that appear only once)
            boundary_edge_indices = group_rows(edges_sorted, require_count=1)
            boundary_edges = edges_sorted[boundary_edge_indices]

            num_boundary_edges = len(boundary_edges)

            # Find faces that contain boundary edges
            # Build a set of boundary edge tuples for fast lookup
            boundary_edge_set = set(map(tuple, boundary_edges))

            # Check each face for boundary edges
            boundary_faces = []
            face_edge_info = []  # Store which edges are open for each face

            for face_idx, face in enumerate(mesh.faces):
                # Get the 3 edges of this face
                edges = [
                    tuple(sorted([face[0], face[1]])),
                    tuple(sorted([face[1], face[2]])),
                    tuple(sorted([face[2], face[0]])),
                ]

                # Check which edges are boundary edges
                open_edges = [e for e in edges if e in boundary_edge_set]

                if open_edges:
                    boundary_faces.append(face_idx)
                    face_edge_info.append({
                        "face_id": face_idx,
                        "num_open_edges": len(open_edges),
                        "open_edges": open_edges,
                        "vertices": face.tolist()
                    })

            num_boundary_faces = len(boundary_faces)

            # Get boundary vertices
            boundary_vertices = np.unique(boundary_edges.flatten()) if num_boundary_edges > 0 else np.array([])
            num_boundary_vertices = len(boundary_vertices)

            # Get mesh name
            mesh_name = mesh.metadata.get('file_name', 'mesh') if hasattr(mesh, 'metadata') else 'mesh'
            mesh_name_short = os.path.splitext(mesh_name)[0]

            # Build summary string
            detail_lines = [f"{mesh_name_short}: {num_boundary_edges} open edge(s), {num_boundary_faces} face(s) with open edges"]
            detail_lines.append(f"  {num_boundary_vertices} boundary vertices")

            # Show first few faces with open edges
            for info in face_edge_info[:10]:
                detail_lines.append(f"  Face {info['face_id']}: {info['num_open_edges']} open edge(s)")

            if len(face_edge_info) > 10:
                detail_lines.append(f"  ... and {len(face_edge_info) - 10} more faces")

            summary_lines.append("\n".join(detail_lines))

            # Prepare UI data
            # For small numbers of faces, include face indices
            ui_faces = []
            for info in face_edge_info:
                face_data = {
                    "id": info["face_id"],
                    "open_edges": info["num_open_edges"],
                    "vertices": info["vertices"]
                }
                # Include edge details for faces with few edges
                if info["num_open_edges"] <= 3:
                    face_data["edge_details"] = [list(e) for e in info["open_edges"]]
                ui_faces.append(face_data)

            ui_data.append({
                "mesh_name": mesh_name_short,
                "num_open_edges": num_boundary_edges,
                "num_boundary_faces": num_boundary_faces,
                "num_boundary_vertices": num_boundary_vertices,
                "total_faces": len(mesh.faces),
                "total_vertices": len(mesh.vertices),
                "faces": ui_faces,
                "boundary_vertex_ids": boundary_vertices.tolist() if len(boundary_vertices) < 100 else None
            })

            # Print to console
            log.info("%s: %d open edges, %d faces with open edges", mesh_name_short, num_boundary_edges, num_boundary_faces)
            for info in face_edge_info[:5]:
                log.info("  Face %d: %d open edge(s)", info['face_id'], info['num_open_edges'])
            if len(face_edge_info) > 5:
                log.info("  ... and %d more faces", len(face_edge_info) - 5)

            # Create face attribute for visualization
            # 0 = interior face, 1+ = number of open edges on that face
            open_edge_count = np.zeros(len(mesh.faces), dtype=np.int32)
            for info in face_edge_info:
                open_edge_count[info["face_id"]] = info["num_open_edges"]

            # Store as face attribute
            result_mesh = mesh.copy()
            result_mesh.face_attributes['open_edge_count'] = open_edge_count

            # Also store boundary info in metadata
            if not hasattr(result_mesh, 'metadata'):
                result_mesh.metadata = {}
            result_mesh.metadata['num_open_edges'] = num_boundary_edges
            result_mesh.metadata['num_boundary_faces'] = num_boundary_faces
            result_mesh.metadata['boundary_face_ids'] = boundary_faces

            result_meshes.append(result_mesh)

        # Create summary string
        summary = "\n\n".join(summary_lines)

        log.info("Processed %d mesh(es)", len(meshes))

        return io.NodeOutput(result_meshes, summary, ui={ "text": [summary], "open_edges_data": ui_data })

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeomPackOpenEdges": OpenEdgesNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackOpenEdges": "Open Edges",
}
