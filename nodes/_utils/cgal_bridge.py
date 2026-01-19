# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
CGAL bridge for isolated environment operations.

This module provides CGAL functions that run in an isolated Python 3.11
environment with CGAL installed. Functions are called via comfy-env's
VenvWorker.call_module() mechanism.

Data is passed via IPC (serialized as lists), so functions accept lists and
return lists/dicts with list values.

Usage from host environment:
    from .cgal_worker import call_cgal
    result = call_cgal('cgal_isotropic_remesh', vertices=v, faces=f, ...)
"""

import numpy as np


def _to_list(arr):
    """Convert numpy array or list to nested list for IPC serialization."""
    if hasattr(arr, 'tolist'):
        return arr.tolist()
    return arr


def _to_numpy(data, dtype=np.float32):
    """Convert list to numpy array."""
    return np.array(data, dtype=dtype)


def cgal_isotropic_remesh(vertices, faces, target_edge_length, iterations=3, protect_boundaries=True):
    """
    Apply CGAL isotropic remeshing using official Python bindings.

    Creates a uniform triangle mesh with specified edge length using CGAL's
    high-quality remeshing algorithm.

    Args:
        vertices: list of [x,y,z] vertex positions
        faces: list of [i,j,k] face indices
        target_edge_length: Target edge length for output triangles
        iterations: Number of remeshing iterations (1-20)
        protect_boundaries: Preserve boundary edges

    Returns:
        dict with 'vertices', 'faces', 'info', 'error' keys
    """
    from CGAL import CGAL_Polygon_mesh_processing
    from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
    from CGAL.CGAL_Kernel import Point_3

    print(f"[cgal_bridge] ===== Starting CGAL Isotropic Remeshing =====")

    # Convert inputs
    vertices = _to_numpy(vertices, np.float64)
    faces = _to_numpy(faces, np.int32)

    print(f"[cgal_bridge] Input: {len(vertices)} vertices, {len(faces)} faces")
    print(f"[cgal_bridge] Parameters: target_edge_length={target_edge_length}, iterations={iterations}, protect_boundaries={protect_boundaries}")

    if len(vertices) == 0 or len(faces) == 0:
        return {'vertices': [], 'faces': [], 'info': '', 'error': 'Mesh is empty'}

    if target_edge_length <= 0:
        return {'vertices': [], 'faces': [], 'info': '', 'error': f'Target edge length must be positive, got {target_edge_length}'}

    if iterations < 1 or iterations > 20:
        return {'vertices': [], 'faces': [], 'info': '', 'error': f'Iterations must be between 1 and 20, got {iterations}'}

    try:
        # Step 1: Convert to CGAL Polyhedron_3
        print(f"[cgal_bridge] Converting to CGAL format...")

        # Create Point_3_Vector for vertices
        points = CGAL_Polygon_mesh_processing.Point_3_Vector()
        points.reserve(len(vertices))
        for v in vertices:
            points.append(Point_3(float(v[0]), float(v[1]), float(v[2])))

        # Create plain Python list of lists for faces
        polygons = [[int(idx) for idx in face] for face in faces]

        # Create polyhedron from polygon soup
        P = Polyhedron_3()
        CGAL_Polygon_mesh_processing.polygon_soup_to_polygon_mesh(points, polygons, P)

        print(f"[cgal_bridge] CGAL mesh created: {P.size_of_vertices()} vertices, {P.size_of_facets()} facets")

        # Step 2: Collect all facets for remeshing
        flist = []
        for fh in P.facets():
            flist.append(fh)

        # Step 3: Handle boundary protection if requested
        if protect_boundaries:
            print(f"[cgal_bridge] Collecting boundary halfedges for protection...")
            hlist = []
            for hh in P.halfedges():
                if hh.is_border() or hh.opposite().is_border():
                    hlist.append(hh)

            print(f"[cgal_bridge] Found {len(hlist)} boundary halfedges")

            # Perform remeshing with boundary protection
            print(f"[cgal_bridge] Running CGAL isotropic_remeshing (with boundary protection)...")
            CGAL_Polygon_mesh_processing.isotropic_remeshing(
                flist,
                target_edge_length,
                P,
                iterations,
                hlist,
                True  # protect_constraints
            )
        else:
            # Perform remeshing without boundary protection
            print(f"[cgal_bridge] Running CGAL isotropic_remeshing...")
            CGAL_Polygon_mesh_processing.isotropic_remeshing(
                flist,
                target_edge_length,
                P,
                iterations
            )

        print(f"[cgal_bridge] Remeshing complete: {P.size_of_vertices()} vertices, {P.size_of_facets()} facets")

        # Step 4: Extract vertices back to numpy arrays
        print(f"[cgal_bridge] Extracting results...")
        new_vertices = []
        vertex_map = {}

        for i, vertex in enumerate(P.vertices()):
            point = vertex.point()
            new_vertices.append([point.x(), point.y(), point.z()])
            vertex_map[vertex] = i

        # Step 5: Extract faces back to numpy arrays
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

        # Build info string
        vertex_change = len(new_vertices) - len(vertices)
        face_change = len(new_faces) - len(faces)
        vertex_pct = (vertex_change / len(vertices)) * 100 if len(vertices) > 0 else 0
        face_pct = (face_change / len(faces)) * 100 if len(faces) > 0 else 0

        info = (
            f"CGAL Isotropic Remeshing:\n"
            f"  Vertices: {len(vertices)} -> {len(new_vertices)} ({vertex_change:+d}, {vertex_pct:+.1f}%)\n"
            f"  Faces: {len(faces)} -> {len(new_faces)} ({face_change:+d}, {face_pct:+.1f}%)\n"
            f"  Target edge length: {target_edge_length}\n"
            f"  Iterations: {iterations}\n"
            f"  Protect boundaries: {protect_boundaries}"
        )

        print(f"[cgal_bridge] ===== Remeshing Complete =====")
        print(f"[cgal_bridge] Vertices: {len(vertices)} -> {len(new_vertices)}")
        print(f"[cgal_bridge] Faces: {len(faces)} -> {len(new_faces)}")

        return {
            'vertices': new_vertices,
            'faces': new_faces,
            'info': info,
            'error': ''
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = f"Error during CGAL remesh: {str(e)}"
        print(f"[cgal_bridge] ERROR: {error_msg}")
        return {
            'vertices': [],
            'faces': [],
            'info': '',
            'error': error_msg
        }
