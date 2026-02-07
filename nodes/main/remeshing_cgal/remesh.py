# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Remesh CGAL Node - CGAL isotropic remeshing
Requires CGAL Python bindings.
"""

import numpy as np
import trimesh as trimesh_module


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


class RemeshCGALNode:
    """
    Remesh CGAL - High-quality isotropic remeshing using CGAL.

    Requires CGAL Python bindings to be installed.
    Produces high-quality isotropic triangulations with good angle properties.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
            "optional": {
                "target_edge_length": ("FLOAT", {
                    "default": 1.00,
                    "min": 0.001,
                    "max": 10.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Target edge length for output triangles. Value is relative to mesh scale.",
                }),
                "iterations": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Number of remeshing passes. More iterations = smoother result, slower processing.",
                }),
                "protect_boundaries": (["true", "false"], {
                    "default": "true",
                    "tooltip": "Lock boundary/open edges in place during remeshing. Prevents modification of mesh borders and holes.",
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("remeshed_mesh", "info")
    FUNCTION = "remesh"
    CATEGORY = "geompack/remeshing"
    OUTPUT_NODE = True

    def remesh(self, trimesh, target_edge_length=1.0, iterations=3, protect_boundaries="true"):
        """Apply CGAL isotropic remeshing."""
        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        print(f"\n{'='*60}")
        print(f"[Remesh CGAL] Backend: cgal_isotropic")
        print(f"[Remesh CGAL] Input: {initial_vertices:,} vertices, {initial_faces:,} faces")
        print(f"[Remesh CGAL] Parameters: target_edge_length={target_edge_length}, iterations={iterations}, protect_boundaries={protect_boundaries}")
        print(f"{'='*60}\n")

        protect = (protect_boundaries == "true")
        print(f"[Remesh CGAL] Running CGAL isotropic remesh (target_edge_length={target_edge_length})...")

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

        print(f"[Remesh CGAL] Output: {len(remeshed_mesh.vertices)} vertices ({vertex_change:+d}), "
              f"{len(remeshed_mesh.faces)} faces ({face_change:+d})")

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
        return {"ui": {"text": [info]}, "result": (remeshed_mesh, info)}


NODE_CLASS_MAPPINGS = {
    "GeomPackRemeshCGAL": RemeshCGALNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackRemeshCGAL": "Remesh CGAL",
}
