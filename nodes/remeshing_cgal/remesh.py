# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Remesh CGAL Node - CGAL isotropic remeshing
Requires CGAL Python bindings.
"""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _cgal_isotropic_remesh(vertices, faces, target_edge_length, iterations, protect_boundaries):
    """CGAL isotropic remeshing."""
    from CGAL import CGAL_Polygon_mesh_processing
    from CGAL.CGAL_Kernel import Point_3
    from CGAL.CGAL_Polyhedron_3 import Polyhedron_3

    if len(vertices) == 0 or len(faces) == 0:
        return {'error': "Mesh is empty"}

    # Convert to CGAL Polyhedron_3
    points = CGAL_Polygon_mesh_processing.Point_3_Vector()
    points.reserve(len(vertices))
    for v in vertices:
        points.append(Point_3(float(v[0]), float(v[1]), float(v[2])))

    polygons = [[int(idx) for idx in face] for face in faces]

    P = Polyhedron_3()
    CGAL_Polygon_mesh_processing.polygon_soup_to_polygon_mesh(points, polygons, P)

    # Collect all facets for remeshing
    flist = []
    for fh in P.facets():
        flist.append(fh)

    # Handle boundary protection if requested
    if protect_boundaries:
        hlist = []
        for hh in P.halfedges():
            if hh.is_border() or hh.opposite().is_border():
                hlist.append(hh)

        CGAL_Polygon_mesh_processing.isotropic_remeshing(
            flist, target_edge_length, P, iterations, hlist, True
        )
    else:
        CGAL_Polygon_mesh_processing.isotropic_remeshing(
            flist, target_edge_length, P, iterations
        )

    # Extract vertices back to list
    new_vertices = []
    vertex_map = {}

    for i, vertex in enumerate(P.vertices()):
        point = vertex.point()
        new_vertices.append([float(point.x()), float(point.y()), float(point.z())])
        vertex_map[vertex] = i

    # Extract faces back to list
    new_faces = []
    for facet in P.facets():
        halfedge = facet.halfedge()
        face_vertices = []

        start = halfedge
        current = start
        while True:
            vertex_handle = current.vertex()
            face_vertices.append(vertex_map[vertex_handle])
            current = current.next()
            if current == start:
                break

        if len(face_vertices) == 3:
            new_faces.append(face_vertices)

    return {'vertices': new_vertices, 'faces': new_faces}


class RemeshCGALNode(io.ComfyNode):
    """
    Remesh CGAL - High-quality isotropic remeshing using CGAL.

    Requires CGAL Python bindings to be installed.
    Produces high-quality isotropic triangulations with good angle properties.
    """


    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackRemesh_CGAL",
            display_name="Remesh CGAL (backend)",
            category="geompack/remeshing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("target_edge_length", default=1.00, min=0.001, max=10.0, step=0.01, display_mode="number", tooltip="Target edge length for output triangles. Value is relative to mesh scale.", optional=True),
                io.Int.Input("iterations", default=3, min=1, max=20, step=1, tooltip="Number of remeshing passes. More iterations = smoother result, slower processing.", optional=True),
                io.Combo.Input("protect_boundaries", options=["true", "false"], default="true", tooltip="Lock boundary/open edges in place during remeshing. Prevents modification of mesh borders and holes.", optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="remeshed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, target_edge_length=1.0, iterations=3, protect_boundaries="true"):
        """Apply CGAL isotropic remeshing."""
        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        log.info("Backend: cgal_isotropic")
        log.info("Input: %s vertices, %s faces", f"{initial_vertices:,}", f"{initial_faces:,}")
        log.info("Parameters: target_edge_length=%s, iterations=%s, protect_boundaries=%s",
                 target_edge_length, iterations, protect_boundaries)

        protect = (protect_boundaries == "true")
        log.info("Running CGAL isotropic remesh (target_edge_length=%s)...", target_edge_length)

        result = _cgal_isotropic_remesh(
            vertices=np.asarray(trimesh.vertices, dtype=np.float64),
            faces=np.asarray(trimesh.faces, dtype=np.int32),
            target_edge_length=target_edge_length,
            iterations=iterations,
            protect_boundaries=protect
        )

        if 'error' in result:
            raise ValueError(f"CGAL remeshing failed: {result['error']}")

        remeshed_mesh = trimesh_module.Trimesh(
            vertices=np.array(result['vertices'], dtype=np.float64),
            faces=np.array(result['faces'], dtype=np.int32),
            process=False
        )

        vertex_change = len(remeshed_mesh.vertices) - initial_vertices
        face_change = len(remeshed_mesh.faces) - initial_faces

        log.info("Output: %d vertices (%+d), %d faces (%+d)",
                 len(remeshed_mesh.vertices), vertex_change, len(remeshed_mesh.faces), face_change)

        info = f"""Remesh Results (CGAL Isotropic):

Target Edge Length: {target_edge_length}
Iterations: {iterations}
Protect Boundaries: {protect_boundaries}

Before:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

After:
  Vertices: {len(remeshed_mesh.vertices):,}
  Faces: {len(remeshed_mesh.faces):,}
"""
        return io.NodeOutput(remeshed_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {
    "GeomPackRemesh_CGAL": RemeshCGALNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackRemesh_CGAL": "Remesh CGAL (backend)",
}
