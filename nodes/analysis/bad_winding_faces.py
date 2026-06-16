# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Bad Winding Faces Node - Detect faces whose winding is inconsistent with neighbours.

Two faces sharing a manifold edge are winding-consistent only when that edge is
traversed in OPPOSITE directions by the two faces. Where it is traversed the SAME
way, the two faces disagree on which side is "out" -- a winding seam. This node marks
every face touching such a seam, writing a per-face field 'bad_winding' (count of
inconsistent edges on that face; 0 = fine) for visualization / thresholding.

Supports batch processing: input a list of meshes, get a list of results.
"""

import logging
import os

import numpy as np
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class BadWindingFacesNode(io.ComfyNode):
    """
    Label faces with inconsistent winding relative to their neighbours.

    Adds a per-face field 'bad_winding' = number of shared (manifold) edges on the
    face whose neighbour disagrees on winding direction. 0 means the face agrees with
    all its neighbours. Boundary and non-manifold edges are ignored here (use Open
    Edges for those).

    Supports batch processing: input a list of meshes, get a list of results.
    """

    INPUT_IS_LIST = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackBadWindingFaces",
            display_name="Bad Winding Faces",
            category="geompack/analysis",
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.String.Input("field_name", default="bad_winding", tooltip="Name of the per-face field to write (0 = consistent, >0 = number of inconsistent shared edges)."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="trimesh", is_output_list=True),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, field_name="bad_winding"):
        from trimesh.grouping import group_rows

        # INPUT_IS_LIST: list-typed inputs arrive as lists; scalars too.
        meshes = trimesh if isinstance(trimesh, list) else [trimesh]
        field_name = field_name[0] if isinstance(field_name, list) else field_name
        field_name = (field_name or "bad_winding").strip() or "bad_winding"

        result_meshes = []
        summary_lines = []
        ui_data = []

        for mesh in meshes:
            num_faces = len(mesh.faces)

            # Directed edges, one row per (face, edge) in face order, and the face each
            # belongs to. A shared edge is winding-consistent iff its two directed
            # half-edges are reverses of one another.
            edges = np.asarray(mesh.edges)          # (3F, 2) directed
            edges_face = np.asarray(mesh.edges_face)  # (3F,)
            edges_sorted = np.sort(edges, axis=1)

            bad_winding = np.zeros(num_faces, dtype=np.int32)
            num_inconsistent_edges = 0

            # Manifold edges: shared by exactly two faces.
            for e0, e1 in group_rows(edges_sorted, require_count=2):
                consistent = (edges[e0][0] == edges[e1][1]) and (edges[e0][1] == edges[e1][0])
                if not consistent:
                    bad_winding[edges_face[e0]] += 1
                    bad_winding[edges_face[e1]] += 1
                    num_inconsistent_edges += 1

            bad_face_ids = np.where(bad_winding > 0)[0]
            num_bad_faces = int(bad_face_ids.size)
            is_consistent = bool(getattr(mesh, "is_winding_consistent", num_inconsistent_edges == 0))

            mesh_name = mesh.metadata.get('file_name', 'mesh') if hasattr(mesh, 'metadata') else 'mesh'
            mesh_name_short = os.path.splitext(mesh_name)[0]

            detail = [
                f"{mesh_name_short}: {num_bad_faces} bad-winding face(s), "
                f"{num_inconsistent_edges} inconsistent edge(s)",
                f"  Winding consistent: {is_consistent}",
                f"  Field: '{field_name}' (0 = ok, >0 = inconsistent shared edges)",
            ]
            if num_bad_faces:
                preview = ", ".join(str(i) for i in bad_face_ids[:20].tolist())
                detail.append(f"  Face ids: {preview}{' ...' if num_bad_faces > 20 else ''}")
            summary_lines.append("\n".join(detail))

            log.info("%s: %d bad-winding faces, %d inconsistent edges (consistent=%s)",
                     mesh_name_short, num_bad_faces, num_inconsistent_edges, is_consistent)

            result_mesh = mesh.copy()
            result_mesh.face_attributes[field_name] = bad_winding
            if not hasattr(result_mesh, 'metadata'):
                result_mesh.metadata = {}
            result_mesh.metadata['is_winding_consistent'] = is_consistent
            result_mesh.metadata['num_bad_winding_faces'] = num_bad_faces
            result_mesh.metadata['num_inconsistent_winding_edges'] = num_inconsistent_edges
            result_mesh.metadata['bad_winding_face_ids'] = bad_face_ids[:1000].tolist()
            result_meshes.append(result_mesh)

            ui_data.append({
                "mesh_name": mesh_name_short,
                "is_winding_consistent": is_consistent,
                "num_bad_winding_faces": num_bad_faces,
                "num_inconsistent_edges": num_inconsistent_edges,
                "total_faces": num_faces,
                "total_vertices": len(mesh.vertices),
                "field_name": field_name,
                "bad_face_ids": bad_face_ids[:200].tolist(),
            })

        summary = "\n\n".join(summary_lines)
        log.info("Processed %d mesh(es)", len(meshes))

        return io.NodeOutput(result_meshes, summary, ui={"text": [summary], "bad_winding_data": ui_data})


NODE_CLASS_MAPPINGS = {
    "GeomPackBadWindingFaces": BadWindingFacesNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackBadWindingFaces": "Bad Winding Faces",
}
