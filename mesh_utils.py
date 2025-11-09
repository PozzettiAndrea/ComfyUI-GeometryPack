"""
Mesh utilities for ComfyUI GeomPack nodes
Handles mesh data structures, I/O, and processing using trimesh and seagullmesh (CGAL)
"""

import numpy as np
import trimesh
import igl
import os
from typing import Tuple, Optional

# PyMeshLab for remeshing (TODO: will switch back to CGAL eventually)
try:
    import pymeshlab
    PYMESHLAB_AVAILABLE = True
except ImportError:
    PYMESHLAB_AVAILABLE = False
    print("[mesh_utils] Warning: pymeshlab not available. Install with: pip install pymeshlab")


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
# Remeshing via PyMeshLab (TODO: will switch back to CGAL eventually)
# ============================================================================

def cgal_isotropic_remesh(
    mesh: trimesh.Trimesh,
    target_edge_length: float,
    iterations: int = 3
) -> Tuple[Optional[trimesh.Trimesh], str]:
    """
    Apply isotropic remeshing to create uniform triangle sizes.

    Currently uses PyMeshLab. Will switch back to CGAL once CGAL Python bindings
    issues are resolved.

    Args:
        mesh: Input trimesh object
        target_edge_length: Target edge length for remeshed triangles
        iterations: Number of remeshing iterations (default: 3)

    Returns:
        Tuple of (remeshed_mesh, error_message)
    """
    print(f"[cgal_isotropic_remesh] ===== Starting Isotropic Remeshing (via PyMeshLab) =====")
    print(f"[cgal_isotropic_remesh] Input mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"[cgal_isotropic_remesh] Parameters:")
    print(f"[cgal_isotropic_remesh]   target_edge_length: {target_edge_length}")
    print(f"[cgal_isotropic_remesh]   iterations: {iterations}")

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
        print(f"[cgal_isotropic_remesh] Converting to PyMeshLab format...")
        ms = pymeshlab.MeshSet()

        # Create PyMeshLab mesh from numpy arrays
        pml_mesh = pymeshlab.Mesh(
            vertex_matrix=mesh.vertices,
            face_matrix=mesh.faces
        )
        ms.add_mesh(pml_mesh)

        print(f"[cgal_isotropic_remesh] Applying isotropic remeshing...")
        # PyMeshLab's isotropic remeshing
        # targetlen is specified as PercentageValue (percentage of bounding box diagonal)
        # We need to convert our absolute target_edge_length to a percentage
        bbox_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
        target_pct = (target_edge_length / bbox_diag) * 100.0

        ms.meshing_isotropic_explicit_remeshing(
            targetlen=pymeshlab.PercentageValue(target_pct),
            iterations=iterations,
            adaptive=False
        )

        # Convert back to trimesh
        print(f"[cgal_isotropic_remesh] Converting back to trimesh...")
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

        print(f"[cgal_isotropic_remesh] ===== Remeshing Complete =====")
        print(f"[cgal_isotropic_remesh] Results:")
        print(f"[cgal_isotropic_remesh]   Vertices: {len(mesh.vertices)} -> {len(remeshed_mesh.vertices)} ({vertex_change:+d}, {vertex_pct:+.1f}%)")
        print(f"[cgal_isotropic_remesh]   Faces:    {len(mesh.faces)} -> {len(remeshed_mesh.faces)} ({face_change:+d}, {face_pct:+.1f}%)")

        return remeshed_mesh, ""

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error during remeshing: {str(e)}"
