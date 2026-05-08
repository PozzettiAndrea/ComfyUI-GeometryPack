# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""CGAL fill holes backend node — triangulate + refine + optional fairing."""

import logging
import numpy as np
import trimesh
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _mesh_to_polyhedron(vertices, faces):
    """Convert numpy arrays to CGAL Polyhedron_3."""
    from CGAL import CGAL_Polygon_mesh_processing
    from CGAL.CGAL_Kernel import Point_3
    from CGAL.CGAL_Polyhedron_3 import Polyhedron_3

    points = CGAL_Polygon_mesh_processing.Point_3_Vector()
    points.reserve(len(vertices))
    for v in vertices:
        points.append(Point_3(float(v[0]), float(v[1]), float(v[2])))

    polygons = [[int(idx) for idx in face] for face in faces]

    P = Polyhedron_3()
    CGAL_Polygon_mesh_processing.polygon_soup_to_polygon_mesh(points, polygons, P)
    return P


def _polyhedron_to_arrays(P):
    """Extract vertices and faces from CGAL Polyhedron_3."""
    verts = []
    vertex_map = {}
    for i, vertex in enumerate(P.vertices()):
        point = vertex.point()
        verts.append([float(point.x()), float(point.y()), float(point.z())])
        vertex_map[vertex] = i

    faces = []
    for facet in P.facets():
        he = facet.halfedge()
        face_verts = []
        start = he
        current = start
        while True:
            face_verts.append(vertex_map[current.vertex()])
            current = current.next()
            if current == start:
                break
        if len(face_verts) == 3:
            faces.append(face_verts)

    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32)


class FillHolesCGALNode(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackFillHoles_CGAL",
            display_name="Fill Holes CGAL (backend)",
            category="geompack/repair",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("mesh"),
                io.Combo.Input("quality", options=["triangulate", "refine", "fair"],
                               default="refine",
                               tooltip="triangulate: fast, basic fill. refine: better triangle quality. fair: smooth fill to match surrounding surface.",
                               optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="filled_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, mesh, quality="refine"):
        from CGAL import CGAL_Polygon_mesh_processing

        initial_faces = len(mesh.faces)
        was_watertight = mesh.is_watertight

        V = np.asarray(mesh.vertices, dtype=np.float64)
        F = np.asarray(mesh.faces, dtype=np.int32)

        P = _mesh_to_polyhedron(V, F)
        log.info("CGAL: built Polyhedron_3 (%d vertices, %d facets)", P.size_of_vertices(), P.size_of_facets())

        # Find border halfedges (one per hole)
        border_halfedges = []
        visited = set()
        for he in P.halfedges():
            if he.is_border() and id(he) not in visited:
                # Walk the border loop to mark all halfedges as visited
                border_halfedges.append(he)
                current = he
                while True:
                    visited.add(id(current))
                    current = current.next()
                    if current == he:
                        break

        log.info("CGAL: found %d holes", len(border_halfedges))

        holes_filled = 0
        for he in border_halfedges:
            try:
                output = []
                if quality == "fair":
                    CGAL_Polygon_mesh_processing.triangulate_refine_and_fair_hole(P, he, output)
                elif quality == "refine":
                    CGAL_Polygon_mesh_processing.triangulate_and_refine_hole(P, he, output)
                else:
                    CGAL_Polygon_mesh_processing.triangulate_hole(P, he, output)
                holes_filled += 1
            except Exception as e:
                log.warning("CGAL: failed to fill hole: %s", e)

        result_V, result_F = _polyhedron_to_arrays(P)
        filled = trimesh.Trimesh(vertices=result_V, faces=result_F, process=False)

        added = len(filled.faces) - initial_faces
        is_watertight = filled.is_watertight

        log.info("CGAL %s: filled %d/%d holes, +%d faces, watertight: %s -> %s",
                 quality, holes_filled, len(border_halfedges), added, was_watertight, is_watertight)

        info = (f"Method: cgal ({quality})\n"
                f"Holes found: {len(border_halfedges)}\n"
                f"Holes filled: {holes_filled}\n"
                f"Faces added: {added}\n"
                f"Watertight: {was_watertight} -> {is_watertight}")

        return io.NodeOutput(filled, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackFillHoles_CGAL": FillHolesCGALNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackFillHoles_CGAL": "Fill Holes CGAL (backend)"}
