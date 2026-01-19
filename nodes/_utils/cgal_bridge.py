# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
CGAL Bridge - CGAL operations that run in the isolated _env_geometrypack environment.

This module is called via comfy-env's VenvWorker.call_module() mechanism.
Functions accept/return lists for IPC serialization.

Usage from host environment:
    from comfy_env import VenvWorker
    worker = VenvWorker(python='_env_geometrypack/bin/python', sys_path=[...])
    result = worker.call_module('cgal_bridge', 'cgal_isotropic_remesh', **kwargs)
"""


def _to_list(arr):
    """Convert numpy array or list to nested list for IPC serialization."""
    if hasattr(arr, 'tolist'):
        return arr.tolist()
    return arr


def cgal_isotropic_remesh(vertices, faces, target_edge_length, iterations, protect_boundaries):
    """
    CGAL isotropic remeshing - runs in isolated environment.

    Creates a uniform triangle mesh with specified edge length using CGAL's
    high-quality remeshing algorithm.

    Args:
        vertices: list of [x,y,z] vertex positions
        faces: list of [i,j,k] face indices
        target_edge_length: Target edge length for output triangles
        iterations: Number of remeshing iterations (1-20)
        protect_boundaries: Whether to preserve boundary edges

    Returns:
        dict with:
            'vertices': list of [x,y,z] vertex positions
            'faces': list of [i,j,k] face indices
        Or dict with 'error': error message if failed
    """
    print(f"[cgal_bridge] ===== Starting CGAL Isotropic Remeshing =====")
    print(f"[cgal_bridge] Input: {len(vertices)} vertices, {len(faces)} faces")
    print(f"[cgal_bridge] Parameters: target_edge_length={target_edge_length}, iterations={iterations}, protect_boundaries={protect_boundaries}")

    # Import CGAL (only available in isolated environment)
    try:
        from CGAL import CGAL_Polygon_mesh_processing
        from CGAL.CGAL_Kernel import Point_3
        from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
    except ImportError as e:
        error_msg = f"CGAL import failed: {e}"
        print(f"[cgal_bridge] ERROR: {error_msg}")
        return {'error': error_msg}

    # Validate inputs
    if len(vertices) == 0 or len(faces) == 0:
        return {'error': "Mesh is empty"}

    if target_edge_length <= 0:
        return {'error': f"Target edge length must be positive, got {target_edge_length}"}

    if iterations < 1 or iterations > 20:
        return {'error': f"Iterations must be between 1 and 20, got {iterations}"}

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

        # Step 4: Extract vertices back to list
        print(f"[cgal_bridge] Extracting result...")
        new_vertices = []
        vertex_map = {}

        for i, vertex in enumerate(P.vertices()):
            point = vertex.point()
            new_vertices.append([float(point.x()), float(point.y()), float(point.z())])
            vertex_map[vertex] = i

        # Step 5: Extract faces back to list
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

        print(f"[cgal_bridge] ===== Remeshing Complete =====")
        print(f"[cgal_bridge] Results: {len(new_vertices)} vertices, {len(new_faces)} faces")

        return {
            'vertices': new_vertices,
            'faces': new_faces
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = f"Error during CGAL remesh: {str(e)}"
        print(f"[cgal_bridge] ERROR: {error_msg}")
        return {'error': error_msg}
