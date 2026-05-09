# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""CGAL edge collapse decimation backend node."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _cgal_edge_collapse(mesh, target_edge_count, cost_strategy):
    """CGAL surface mesh simplification via edge collapse."""
    from CGAL import CGAL_Surface_mesh_simplification
    from CGAL.CGAL_Kernel import Point_3
    from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
    from CGAL import CGAL_Polygon_mesh_processing

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("Mesh is empty")

    # Build CGAL Polyhedron_3
    points = CGAL_Polygon_mesh_processing.Point_3_Vector()
    points.reserve(len(vertices))
    for v in vertices:
        points.append(Point_3(float(v[0]), float(v[1]), float(v[2])))

    polygons = [[int(idx) for idx in face] for face in faces]

    P = Polyhedron_3()
    CGAL_Polygon_mesh_processing.polygon_soup_to_polygon_mesh(points, polygons, P)

    # Choose cost strategy
    if cost_strategy == "lindstrom_turk":
        cost = CGAL_Surface_mesh_simplification.LindstromTurk_cost()
        placement = CGAL_Surface_mesh_simplification.LindstromTurk_placement()
    else:
        cost = CGAL_Surface_mesh_simplification.Edge_length_cost()
        placement = CGAL_Surface_mesh_simplification.Midpoint_placement()

    stop = CGAL_Surface_mesh_simplification.Count_stop_predicate(target_edge_count)

    CGAL_Surface_mesh_simplification.edge_collapse(P, stop, cost, placement)

    # Extract result
    new_vertices = []
    vertex_map = {}
    for i, vertex in enumerate(P.vertices()):
        point = vertex.point()
        new_vertices.append([float(point.x()), float(point.y()), float(point.z())])
        vertex_map[vertex] = i

    new_faces = []
    for facet in P.facets():
        halfedge = facet.halfedge()
        face_vertices = []
        start = halfedge
        current = start
        while True:
            face_vertices.append(vertex_map[current.vertex()])
            current = current.next()
            if current == start:
                break
        if len(face_vertices) == 3:
            new_faces.append(face_vertices)

    return trimesh_module.Trimesh(
        vertices=np.array(new_vertices, dtype=np.float64),
        faces=np.array(new_faces, dtype=np.int32),
        process=False,
    )


class DecimateCGALEdgeCollapseNode(io.ComfyNode):
    """CGAL Lindstrom-Turk edge collapse decimation backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackDecimate_CGALEdgeCollapse",
            display_name="Decimate CGAL Edge Collapse (backend)",
            category="geompack/decimation",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("target_face_count", default=5000, min=4, max=10000000, step=100, tooltip="Target number of output faces."),
                io.Combo.Input("cost_strategy", options=["lindstrom_turk", "edge_length"], default="lindstrom_turk", tooltip="CGAL cost strategy. lindstrom_turk=optimizes geometry+volume+boundary (best), edge_length=collapses shortest edges first (fast)."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="decimated_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, target_face_count=5000, cost_strategy="lindstrom_turk"):
        log.info("Backend: cgal_edge_collapse")
        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)
        log.info("Input: %s vertices, %s faces", f"{initial_vertices:,}", f"{initial_faces:,}")

        target_edges = int(target_face_count * 1.5)
        log.info("Parameters: target_edges=%d (from target_faces=%d), cost=%s",
                 target_edges, target_face_count, cost_strategy)

        decimated = _cgal_edge_collapse(trimesh, target_edges, cost_strategy)

        if hasattr(trimesh, "metadata") and trimesh.metadata:
            decimated.metadata = trimesh.metadata.copy()
        decimated.metadata["decimation"] = {
            "algorithm": "cgal_edge_collapse",
            "original_vertices": initial_vertices,
            "original_faces": initial_faces,
            "result_vertices": len(decimated.vertices),
            "result_faces": len(decimated.faces),
        }

        face_change = len(decimated.faces) - initial_faces
        face_pct = (face_change / initial_faces) * 100 if initial_faces > 0 else 0

        info = f"""Decimate Results (cgal_edge_collapse):

Target Face Count: {target_face_count:,}
Cost Strategy: {cost_strategy}

Before:
  Vertices: {initial_vertices:,}
  Faces: {initial_faces:,}

After:
  Vertices: {len(decimated.vertices):,}
  Faces: {len(decimated.faces):,}
  Reduction: {abs(face_pct):.1f}%
"""
        return io.NodeOutput(decimated, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackDecimate_CGALEdgeCollapse": DecimateCGALEdgeCollapseNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackDecimate_CGALEdgeCollapse": "Decimate CGAL Edge Collapse (backend)"}
