"""
Mesh utilities for ComfyUI GeomPack nodes
Handles mesh data structures, I/O, and processing using trimesh and CGAL
"""

import numpy as np
import trimesh
import igl
import os
from typing import Tuple, Optional

# PyMeshLab for remeshing (alternative to CGAL)
try:
    import pymeshlab
    PYMESHLAB_AVAILABLE = True
except ImportError:
    PYMESHLAB_AVAILABLE = False
    print("[mesh_utils] Warning: pymeshlab not available. Install with: pip install pymeshlab")

# Official CGAL Python bindings for isotropic remeshing
try:
    from CGAL import CGAL_Polygon_mesh_processing
    from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
    from CGAL.CGAL_Kernel import Point_3
    from CGAL.CGAL_Polygon_mesh_processing import Point_3_Vector, Polygon_Vector, Int_Vector
    CGAL_AVAILABLE = True
except ImportError:
    CGAL_AVAILABLE = False
    print("[mesh_utils] Warning: CGAL not available. Install with: pip install cgal")


def load_mesh_file(file_path: str) -> Tuple[Optional[trimesh.Trimesh], str]:
    """
    Load a mesh from file.

    Ensures the returned mesh has only triangular faces and is properly processed.

    Args:
        file_path: Path to mesh file (OBJ, PLY, STL, OFF, etc.)

    Returns:
        Tuple of (mesh, error_message)
    """
    if not os.path.exists(file_path):
        return None, f"File not found: {file_path}"

    try:
        print(f"[load_mesh_file] Loading: {file_path}")

        # Try to load with trimesh first (supports many formats)
        loaded = trimesh.load(file_path, force='mesh')

        print(f"[load_mesh_file] Loaded type: {type(loaded).__name__}")

        # Handle case where trimesh.load returns a Scene instead of a mesh
        if isinstance(loaded, trimesh.Scene):
            print(f"[load_mesh_file] Converting Scene to single mesh (scene has {len(loaded.geometry)} geometries)")
            # If it's a scene, dump it to a single mesh
            mesh = loaded.dump(concatenate=True)
        else:
            mesh = loaded

        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return None, f"Failed to read mesh or mesh is empty: {file_path}"

        print(f"[load_mesh_file] Initial mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Ensure mesh is properly triangulated
        # Trimesh should handle this, but some file formats might have issues
        if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
            # Check if faces are triangular
            if mesh.faces.shape[1] != 3:
                # Need to triangulate - this shouldn't normally happen but handle it
                print(f"[load_mesh_file] Warning: Mesh has non-triangular faces, triangulating...")
                # trimesh.Trimesh constructor should triangulate automatically with process=True
                mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
                print(f"[load_mesh_file] After triangulation: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Count before cleanup
        verts_before = len(mesh.vertices)
        faces_before = len(mesh.faces)

        # Merge duplicate vertices and clean up
        mesh.merge_vertices()
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()

        verts_after = len(mesh.vertices)
        faces_after = len(mesh.faces)

        if verts_before != verts_after or faces_before != faces_after:
            print(f"[load_mesh_file] Cleanup: {verts_before}->{verts_after} vertices, {faces_before}->{faces_after} faces")
            print(f"[load_mesh_file]   Removed: {verts_before - verts_after} duplicate vertices, {faces_before - faces_after} bad faces")

        # Store file metadata
        mesh.metadata['file_path'] = file_path
        mesh.metadata['file_name'] = os.path.basename(file_path)
        mesh.metadata['file_format'] = os.path.splitext(file_path)[1].lower()

        print(f"[load_mesh_file] ✓ Successfully loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        return mesh, ""

    except Exception as e:
        print(f"[load_mesh_file] Trimesh failed: {str(e)}, trying libigl fallback...")
        # Fallback to libigl
        try:
            v, f = igl.read_triangle_mesh(file_path)
            if v is None or f is None or len(v) == 0 or len(f) == 0:
                return None, f"Failed to read mesh: {file_path}"

            print(f"[load_mesh_file] libigl loaded: {len(v)} vertices, {len(f)} faces")

            mesh = trimesh.Trimesh(vertices=v, faces=f, process=True)

            # Count before cleanup
            verts_before = len(mesh.vertices)
            faces_before = len(mesh.faces)

            # Clean up the mesh
            mesh.merge_vertices()
            mesh.remove_duplicate_faces()
            mesh.remove_degenerate_faces()

            verts_after = len(mesh.vertices)
            faces_after = len(mesh.faces)

            if verts_before != verts_after or faces_before != faces_after:
                print(f"[load_mesh_file] Cleanup: {verts_before}->{verts_after} vertices, {faces_before}->{faces_after} faces")

            # Store metadata
            mesh.metadata['file_path'] = file_path
            mesh.metadata['file_name'] = os.path.basename(file_path)
            mesh.metadata['file_format'] = os.path.splitext(file_path)[1].lower()

            print(f"[load_mesh_file] ✓ Successfully loaded via libigl: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            return mesh, ""
        except Exception as e2:
            print(f"[load_mesh_file] ✗ Both loaders failed!")
            return None, f"Error loading mesh: {str(e)}; Fallback error: {str(e2)}"


def save_mesh_file(mesh: trimesh.Trimesh, file_path: str) -> Tuple[bool, str]:
    """
    Save a mesh to file.

    Args:
        mesh: Trimesh object
        file_path: Output file path

    Returns:
        Tuple of (success, error_message)
    """
    if not isinstance(mesh, trimesh.Trimesh):
        return False, "Input must be a trimesh.Trimesh object"

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return False, "Mesh is empty"

    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Export the mesh
        mesh.export(file_path)

        return True, ""

    except Exception as e:
        return False, f"Error saving mesh: {str(e)}"


def create_cube(size: float = 1.0) -> trimesh.Trimesh:
    """
    Create a cube mesh.

    Args:
        size: Side length of the cube

    Returns:
        Trimesh object
    """
    mesh = trimesh.creation.box(extents=[size, size, size])
    mesh.metadata['primitive_type'] = 'cube'
    mesh.metadata['size'] = size
    return mesh


def create_sphere(radius: float = 1.0, subdivisions: int = 2) -> trimesh.Trimesh:
    """
    Create a sphere mesh using icosphere subdivision.

    Args:
        radius: Radius of the sphere
        subdivisions: Number of subdivision levels (0-4 recommended)

    Returns:
        Trimesh object
    """
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    mesh.metadata['primitive_type'] = 'sphere'
    mesh.metadata['radius'] = radius
    mesh.metadata['subdivisions'] = subdivisions
    return mesh


def create_plane(size: float = 1.0, subdivisions: int = 1) -> trimesh.Trimesh:
    """
    Create a plane mesh.

    Args:
        size: Side length of the plane
        subdivisions: Number of subdivisions per side

    Returns:
        Trimesh object
    """
    n = subdivisions + 1
    s = size / 2.0

    # Create grid of vertices
    x = np.linspace(-s, s, n)
    y = np.linspace(-s, s, n)
    xx, yy = np.meshgrid(x, y)

    vertices = np.stack([
        xx.flatten(),
        yy.flatten(),
        np.zeros(n * n)
    ], axis=1).astype(np.float64)

    # Create faces
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            idx = i * n + j
            # Two triangles per quad
            faces.append([idx, idx + n, idx + n + 1])
            faces.append([idx, idx + n + 1, idx + 1])

    faces = np.array(faces, dtype=np.int32)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.metadata['primitive_type'] = 'plane'
    mesh.metadata['size'] = size
    mesh.metadata['subdivisions'] = subdivisions
    return mesh


def compute_mesh_info(mesh: trimesh.Trimesh) -> str:
    """
    Generate a formatted string with mesh information.

    Args:
        mesh: Trimesh object

    Returns:
        Formatted info string
    """
    if not isinstance(mesh, trimesh.Trimesh):
        return "Error: Input must be a trimesh.Trimesh object"

    info = "=== Mesh Information ===\n\n"
    info += f"Vertices: {len(mesh.vertices):,}\n"
    info += f"Faces: {len(mesh.faces):,}\n"
    info += f"Edges: {len(mesh.edges):,}\n\n"

    # Geometric properties
    info += f"Volume: {mesh.volume:.6f}\n"
    info += f"Surface Area: {mesh.area:.6f}\n"
    info += f"Is Watertight: {mesh.is_watertight}\n"
    info += f"Is Winding Consistent: {mesh.is_winding_consistent}\n\n"

    # Bounding box
    bounds = mesh.bounds
    center = mesh.centroid
    extents = mesh.extents

    info += "Bounding Box:\n"
    info += f"  Min: [{bounds[0][0]:.3f}, {bounds[0][1]:.3f}, {bounds[0][2]:.3f}]\n"
    info += f"  Max: [{bounds[1][0]:.3f}, {bounds[1][1]:.3f}, {bounds[1][2]:.3f}]\n"
    info += f"  Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]\n"
    info += f"  Extents: [{extents[0]:.3f}, {extents[1]:.3f}, {extents[2]:.3f}]\n\n"

    # Additional data
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        info += f"Vertex Colors: Yes\n"

    if mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0:
        info += f"Vertex Normals: Yes\n"

    # Metadata
    if mesh.metadata:
        info += "\nMetadata:\n"
        for key, value in mesh.metadata.items():
            info += f"  {key}: {value}\n"

    return info


# ============================================================================
# Remeshing via PyMeshLab
# ============================================================================

def pymeshlab_isotropic_remesh(
    mesh: trimesh.Trimesh,
    target_edge_length: float,
    iterations: int = 3
) -> Tuple[Optional[trimesh.Trimesh], str]:
    """
    Apply isotropic remeshing to create uniform triangle sizes using PyMeshLab.

    Args:
        mesh: Input trimesh object
        target_edge_length: Target edge length for remeshed triangles
        iterations: Number of remeshing iterations (default: 3)

    Returns:
        Tuple of (remeshed_mesh, error_message)
    """
    print(f"[pymeshlab_isotropic_remesh] ===== Starting Isotropic Remeshing =====")
    print(f"[pymeshlab_isotropic_remesh] Input mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"[pymeshlab_isotropic_remesh] Parameters:")
    print(f"[pymeshlab_isotropic_remesh]   target_edge_length: {target_edge_length}")
    print(f"[pymeshlab_isotropic_remesh]   iterations: {iterations}")

    if not PYMESHLAB_AVAILABLE:
        return None, "pymeshlab is not installed. Install with: pip install pymeshlab"

    if not isinstance(mesh, trimesh.Trimesh):
        return None, "Input must be a trimesh.Trimesh object"

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return None, "Mesh is empty"

    if target_edge_length <= 0:
        return None, f"Target edge length must be positive, got {target_edge_length}"

    if iterations < 1:
        return None, f"Iterations must be at least 1, got {iterations}"

    try:
        # Convert trimesh to PyMeshLab
        print(f"[pymeshlab_isotropic_remesh] Converting to PyMeshLab format...")
        ms = pymeshlab.MeshSet()

        # Create PyMeshLab mesh from numpy arrays
        pml_mesh = pymeshlab.Mesh(
            vertex_matrix=mesh.vertices,
            face_matrix=mesh.faces
        )
        ms.add_mesh(pml_mesh)

        print(f"[pymeshlab_isotropic_remesh] Applying isotropic remeshing...")
        # PyMeshLab's isotropic remeshing
        # targetlen is specified as PercentageValue (percentage of bounding box diagonal)
        # We need to convert our absolute target_edge_length to a percentage
        bbox_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
        target_pct = (target_edge_length / bbox_diag) * 100.0

        # Try new API name (v2022.2+), fall back to old name for backward compatibility
        try:
            ms.meshing_isotropic_explicit_remeshing(
                targetlen=pymeshlab.PercentageValue(target_pct),
                iterations=iterations,
                adaptive=False
            )
        except AttributeError:
            # Older PyMeshLab versions use 'remeshing_' prefix
            try:
                ms.remeshing_isotropic_explicit_remeshing(
                    targetlen=pymeshlab.PercentageValue(target_pct),
                    iterations=iterations,
                    adaptive=False
                )
            except AttributeError as e:
                return None, (
                    "PyMeshLab meshing filter not available. "
                    "This usually means the libfilter_meshing.so plugin failed to load. "
                    "On Linux, install OpenGL libraries: sudo apt-get install libgl1-mesa-glx libglu1-mesa"
                )

        # Convert back to trimesh
        print(f"[pymeshlab_isotropic_remesh] Converting back to trimesh...")
        remeshed_pml = ms.current_mesh()
        remeshed_mesh = trimesh.Trimesh(
            vertices=remeshed_pml.vertex_matrix(),
            faces=remeshed_pml.face_matrix()
        )

        # Preserve metadata
        remeshed_mesh.metadata = mesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'pymeshlab_isotropic',
            'target_edge_length': target_edge_length,
            'target_percentage': target_pct,
            'iterations': iterations,
            'original_vertices': len(mesh.vertices),
            'original_faces': len(mesh.faces),
            'remeshed_vertices': len(remeshed_mesh.vertices),
            'remeshed_faces': len(remeshed_mesh.faces)
        }

        # Calculate statistics
        vertex_change = len(remeshed_mesh.vertices) - len(mesh.vertices)
        face_change = len(remeshed_mesh.faces) - len(mesh.faces)
        vertex_pct = (vertex_change / len(mesh.vertices)) * 100 if len(mesh.vertices) > 0 else 0
        face_pct = (face_change / len(mesh.faces)) * 100 if len(mesh.faces) > 0 else 0

        print(f"[pymeshlab_isotropic_remesh] ===== Remeshing Complete =====")
        print(f"[pymeshlab_isotropic_remesh] Results:")
        print(f"[pymeshlab_isotropic_remesh]   Vertices: {len(mesh.vertices)} -> {len(remeshed_mesh.vertices)} ({vertex_change:+d}, {vertex_pct:+.1f}%)")
        print(f"[pymeshlab_isotropic_remesh]   Faces:    {len(mesh.faces)} -> {len(remeshed_mesh.faces)} ({face_change:+d}, {face_pct:+.1f}%)")

        return remeshed_mesh, ""

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error during remeshing: {str(e)}"


# ============================================================================
# Boundary Edge Detection
# ============================================================================

def mark_boundary_vertices(mesh: trimesh.Trimesh) -> Tuple[Optional[trimesh.Trimesh], str]:
    """
    Detect boundary edges and mark boundary vertices with a scalar field.

    A boundary edge is an edge that belongs to only one face (not shared).
    This function creates a vertex attribute 'boundary_vertex' where:
    - 1.0 = vertex is on a boundary edge
    - 0.0 = vertex is interior (not on boundary)

    Args:
        mesh: Input trimesh object

    Returns:
        Tuple of (mesh_with_field, error_message)
    """
    print(f"[mark_boundary_vertices] ===== Detecting Boundary Edges =====")
    print(f"[mark_boundary_vertices] Input mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    if not isinstance(mesh, trimesh.Trimesh):
        return None, "Input must be a trimesh.Trimesh object"

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return None, "Mesh is empty"

    try:
        # Get all edges and their face adjacency
        # edges_unique: unique edges in the mesh
        # edges_face: which faces each edge belongs to
        edges = mesh.edges
        edges_unique = mesh.edges_unique
        edges_sorted = mesh.edges_sorted

        print(f"[mark_boundary_vertices] Total edges: {len(edges)}")
        print(f"[mark_boundary_vertices] Unique edges: {len(edges_unique)}")

        # Find boundary edges (edges that appear only once in edges_sorted)
        # trimesh stores edges as sorted pairs, so we can use grouping
        from trimesh.grouping import group_rows

        # Group identical edges and count occurrences
        # require_count=1 means edges that appear exactly once (boundary edges)
        boundary_edge_indices = group_rows(edges_sorted, require_count=1)

        print(f"[mark_boundary_vertices] Boundary edge groups: {len(boundary_edge_indices)}")

        # Get the actual boundary edges
        boundary_edges = edges_sorted[boundary_edge_indices]

        print(f"[mark_boundary_vertices] Boundary edges: {len(boundary_edges)}")

        # Create vertex field: 1.0 for boundary vertices, 0.0 for interior
        boundary_field = np.zeros(len(mesh.vertices), dtype=np.float32)

        # Mark all vertices that are part of boundary edges
        boundary_vertices = np.unique(boundary_edges.flatten())
        boundary_field[boundary_vertices] = 1.0

        num_boundary_verts = np.sum(boundary_field > 0.5)
        print(f"[mark_boundary_vertices] Boundary vertices: {num_boundary_verts} / {len(mesh.vertices)} ({100.0 * num_boundary_verts / len(mesh.vertices):.1f}%)")

        # Create a copy of the mesh to avoid modifying the original
        result_mesh = mesh.copy()

        # Add the boundary field as a vertex attribute
        result_mesh.vertex_attributes['boundary_vertex'] = boundary_field

        # Store metadata
        result_mesh.metadata['has_boundary_field'] = True
        result_mesh.metadata['boundary_vertices_count'] = int(num_boundary_verts)
        result_mesh.metadata['boundary_edges_count'] = len(boundary_edges)

        print(f"[mark_boundary_vertices] ===== Boundary Detection Complete =====")
        print(f"[mark_boundary_vertices] Added 'boundary_vertex' field to mesh")

        return result_mesh, ""

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error detecting boundary edges: {str(e)}"


def cgal_isotropic_remesh(
    mesh: trimesh.Trimesh,
    target_edge_length: float,
    iterations: int = 3,
    protect_boundaries: bool = True
) -> Tuple[Optional[trimesh.Trimesh], str]:
    """
    Apply CGAL isotropic remeshing using official Python bindings.

    Creates a uniform triangle mesh with specified edge length using CGAL's
    high-quality remeshing algorithm.

    Args:
        mesh: Input trimesh object
        target_edge_length: Target edge length for output triangles
        iterations: Number of remeshing iterations (1-20)
        protect_boundaries: Preserve boundary edges

    Returns:
        Tuple of (remeshed_mesh, error_message)
    """
    print(f"[cgal_isotropic_remesh] ===== Starting CGAL Isotropic Remeshing =====")
    print(f"[cgal_isotropic_remesh] Input: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"[cgal_isotropic_remesh] Parameters: target_edge_length={target_edge_length}, iterations={iterations}, protect_boundaries={protect_boundaries}")

    if not CGAL_AVAILABLE:
        error_msg = "CGAL is not installed. Install with: pip install cgal"
        print(f"[cgal_isotropic_remesh] ERROR: {error_msg}")
        return None, error_msg

    if not isinstance(mesh, trimesh.Trimesh):
        return None, "Input must be a trimesh.Trimesh object"

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return None, "Mesh is empty"

    if target_edge_length <= 0:
        return None, f"Target edge length must be positive, got {target_edge_length}"

    if iterations < 1 or iterations > 20:
        return None, f"Iterations must be between 1 and 20, got {iterations}"

    try:
        # Step 1: Convert trimesh to CGAL Polyhedron_3
        print(f"[cgal_isotropic_remesh] Converting to CGAL format...")

        # Create Point_3_Vector for vertices
        # Use fully qualified module paths to avoid namespace issues
        points = CGAL_Polygon_mesh_processing.Point_3_Vector()
        points.reserve(len(mesh.vertices))
        for v in mesh.vertices:
            points.append(Point_3(float(v[0]), float(v[1]), float(v[2])))

        # Create Polygon_Vector for faces
        # Use fully qualified module path to avoid namespace issues
        polygons = CGAL_Polygon_mesh_processing.Polygon_Vector()
        for face in mesh.faces:
            polygon = CGAL_Polygon_mesh_processing.Int_Vector()
            polygon.reserve(3)
            polygon.append(int(face[0]))
            polygon.append(int(face[1]))
            polygon.append(int(face[2]))
            polygons.append(polygon)

        # Create polyhedron from polygon soup
        P = Polyhedron_3()
        CGAL_Polygon_mesh_processing.polygon_soup_to_polygon_mesh(points, polygons, P)

        print(f"[cgal_isotropic_remesh] CGAL mesh created: {P.size_of_vertices()} vertices, {P.size_of_facets()} facets")

        # Step 2: Collect all facets for remeshing
        flist = []
        for fh in P.facets():
            flist.append(fh)

        # Step 3: Handle boundary protection if requested
        if protect_boundaries:
            print(f"[cgal_isotropic_remesh] Collecting boundary halfedges for protection...")
            hlist = []
            for hh in P.halfedges():
                if hh.is_border() or hh.opposite().is_border():
                    hlist.append(hh)

            print(f"[cgal_isotropic_remesh] Found {len(hlist)} boundary halfedges")

            # Perform remeshing with boundary protection
            print(f"[cgal_isotropic_remesh] Running CGAL isotropic_remeshing (with boundary protection)...")
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
            print(f"[cgal_isotropic_remesh] Running CGAL isotropic_remeshing...")
            CGAL_Polygon_mesh_processing.isotropic_remeshing(
                flist,
                target_edge_length,
                P,
                iterations
            )

        print(f"[cgal_isotropic_remesh] Remeshing complete: {P.size_of_vertices()} vertices, {P.size_of_facets()} facets")

        # Step 4: Extract vertices back to numpy arrays
        print(f"[cgal_isotropic_remesh] Converting back to trimesh...")
        new_vertices = []
        vertex_map = {}

        for i, vertex in enumerate(P.vertices()):
            point = vertex.point()
            new_vertices.append([point.x(), point.y(), point.z()])
            vertex_map[vertex] = i

        new_vertices = np.array(new_vertices, dtype=np.float64)

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

        new_faces = np.array(new_faces, dtype=np.int32)

        # Create new trimesh object
        remeshed_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

        # Preserve metadata
        remeshed_mesh.metadata = mesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'cgal_isotropic_python',
            'target_edge_length': target_edge_length,
            'iterations': iterations,
            'protect_boundaries': protect_boundaries,
            'original_vertices': len(mesh.vertices),
            'original_faces': len(mesh.faces),
            'remeshed_vertices': len(remeshed_mesh.vertices),
            'remeshed_faces': len(remeshed_mesh.faces)
        }

        # Calculate statistics
        vertex_change = len(remeshed_mesh.vertices) - len(mesh.vertices)
        face_change = len(remeshed_mesh.faces) - len(mesh.faces)
        vertex_pct = (vertex_change / len(mesh.vertices)) * 100 if len(mesh.vertices) > 0 else 0
        face_pct = (face_change / len(mesh.faces)) * 100 if len(mesh.faces) > 0 else 0

        print(f"[cgal_isotropic_remesh] ===== Remeshing Complete =====")
        print(f"[cgal_isotropic_remesh] Results:")
        print(f"[cgal_isotropic_remesh]   Vertices: {len(mesh.vertices)} -> {len(remeshed_mesh.vertices)} ({vertex_change:+d}, {vertex_pct:+.1f}%)")
        print(f"[cgal_isotropic_remesh]   Faces:    {len(mesh.faces)} -> {len(remeshed_mesh.faces)} ({face_change:+d}, {face_pct:+.1f}%)")

        return remeshed_mesh, ""

    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = f"Error during CGAL remesh: {str(e)}"
        print(f"[cgal_isotropic_remesh] ERROR: {error_msg}")
        return None, error_msg
