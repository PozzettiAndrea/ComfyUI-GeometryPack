# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Mesh utilities for ComfyUI GeomPack nodes
Handles mesh data structures, I/O, and processing using trimesh and CGAL
"""

import numpy as np
import trimesh
import os
from typing import Tuple, Optional

# libigl for mesh processing operations
try:
    import igl
    IGL_AVAILABLE = True
except ImportError:
    IGL_AVAILABLE = False
    print("[GeomPack] Warning: libigl not available. Some mesh operations will be limited.")
    print("[GeomPack]   Install with: pip install libigl")

# PyMeshLab for remeshing (alternative to CGAL)
try:
    import pymeshlab
    PYMESHLAB_AVAILABLE = True
except ImportError:
    PYMESHLAB_AVAILABLE = False
    print("[mesh_utils] Warning: pymeshlab not available. Install with: pip install pymeshlab")

# PyVista for VTK format support (VTP, VTU, etc.)
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    # Don't print warning - pyvista is optional

# Official CGAL Python bindings for isotropic remeshing
# Note: CGAL remeshing runs in an isolated environment (_env_geometrypack),
# so it's normal for this import to fail in the main ComfyUI environment.
# The critical CGAL functionality (boolean ops, self-intersections) is
# provided by libigl (igl.copyleft.cgal), not this package.
try:
    from CGAL import CGAL_Polygon_mesh_processing
    from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
    from CGAL.CGAL_Kernel import Point_3
    from CGAL.CGAL_Polygon_mesh_processing import Point_3_Vector, Polygon_Vector, Int_Vector
    CGAL_AVAILABLE = True
except ImportError:
    CGAL_AVAILABLE = False
    # Don't warn - CGAL remeshing uses isolated environment, not main env


def is_point_cloud(mesh) -> bool:
    """
    Check if a trimesh object is a point cloud (has no faces).

    Args:
        mesh: trimesh.Trimesh or trimesh.PointCloud object

    Returns:
        True if the object is a point cloud (no faces), False if it's a mesh with faces
    """
    return not (hasattr(mesh, 'faces') and mesh.faces is not None and len(mesh.faces) > 0)


def get_face_count(mesh) -> int:
    """
    Safely get the number of faces from a mesh object.

    Args:
        mesh: trimesh.Trimesh or trimesh.PointCloud object

    Returns:
        Number of faces, or 0 if the object is a point cloud
    """
    return len(mesh.faces) if hasattr(mesh, 'faces') and mesh.faces is not None else 0


def get_geometry_type(mesh) -> str:
    """
    Get a human-readable string describing the geometry type.

    Args:
        mesh: trimesh.Trimesh or trimesh.PointCloud object

    Returns:
        "Point Cloud" or "Mesh"
    """
    return "Point Cloud" if is_point_cloud(mesh) else "Mesh"


def _load_vtk_mesh(file_path: str) -> Tuple[Optional[trimesh.Trimesh], str]:
    """
    Load VTK format files (VTP, VTU, VTK) using pyvista.

    Args:
        file_path: Path to VTK format file

    Returns:
        Tuple of (mesh, error_message)
    """
    if not PYVISTA_AVAILABLE:
        return None, (
            f"VTK format files ({os.path.splitext(file_path)[1]}) require pyvista. "
            f"Install with: pip install pyvista"
        )

    try:
        print(f"[load_mesh_file] Loading VTK format: {file_path}")

        # Load with pyvista
        pv_mesh = pv.read(file_path)

        # Ensure we have a surface mesh (triangulated)
        if hasattr(pv_mesh, 'extract_surface'):
            pv_mesh = pv_mesh.extract_surface()

        # Triangulate if needed
        if hasattr(pv_mesh, 'triangulate'):
            pv_mesh = pv_mesh.triangulate()

        # Extract vertices and faces
        vertices = np.array(pv_mesh.points)

        # PyVista faces are stored as [n, v0, v1, v2, n, v0, v1, v2, ...]
        # where n is the number of vertices per face (3 for triangles)
        if hasattr(pv_mesh, 'faces') and pv_mesh.faces is not None and len(pv_mesh.faces) > 0:
            faces_flat = np.array(pv_mesh.faces)
            # Parse the flat array into triangle indices
            faces = []
            i = 0
            while i < len(faces_flat):
                n_verts = faces_flat[i]
                if n_verts == 3:
                    faces.append([faces_flat[i+1], faces_flat[i+2], faces_flat[i+3]])
                elif n_verts == 4:
                    # Triangulate quads
                    faces.append([faces_flat[i+1], faces_flat[i+2], faces_flat[i+3]])
                    faces.append([faces_flat[i+1], faces_flat[i+3], faces_flat[i+4]])
                i += n_verts + 1
            faces = np.array(faces, dtype=np.int32)
        else:
            return None, f"VTK file has no faces: {file_path}"

        if len(vertices) == 0 or len(faces) == 0:
            return None, f"VTK file is empty: {file_path}"

        print(f"[load_mesh_file] VTK mesh: {len(vertices)} vertices, {len(faces)} faces")

        # Create trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

        # Transfer scalar fields from VTK to trimesh attributes
        if hasattr(pv_mesh, 'point_data') and pv_mesh.point_data:
            for name in pv_mesh.point_data.keys():
                try:
                    data = np.array(pv_mesh.point_data[name])
                    if len(data) == len(vertices):
                        mesh.vertex_attributes[name] = data.astype(np.float32)
                        print(f"[load_mesh_file] Transferred vertex attribute: {name}")
                except Exception:
                    pass

        if hasattr(pv_mesh, 'cell_data') and pv_mesh.cell_data:
            for name in pv_mesh.cell_data.keys():
                try:
                    data = np.array(pv_mesh.cell_data[name])
                    if len(data) == len(faces):
                        mesh.face_attributes[name] = data.astype(np.float32)
                        print(f"[load_mesh_file] Transferred face attribute: {name}")
                except Exception:
                    pass

        # Store metadata
        mesh.metadata['file_path'] = file_path
        mesh.metadata['file_name'] = os.path.basename(file_path)
        mesh.metadata['file_format'] = os.path.splitext(file_path)[1].lower()

        print(f"[load_mesh_file] [OK] Successfully loaded VTK: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        return mesh, ""

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error loading VTK file: {str(e)}"


def load_mesh_file(file_path: str) -> Tuple[Optional[trimesh.Trimesh], str]:
    """
    Load a mesh from file.

    Ensures the returned mesh has only triangular faces and is properly processed.

    Args:
        file_path: Path to mesh file (OBJ, PLY, STL, OFF, VTP, VTU, etc.)

    Returns:
        Tuple of (mesh, error_message)
    """
    if not os.path.exists(file_path):
        return None, f"File not found: {file_path}"

    # Check for VTK formats (VTP, VTU) - require pyvista
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.vtp', '.vtu', '.vtk']:
        return _load_vtk_mesh(file_path)

    try:
        print(f"[load_mesh_file] Loading: {file_path}")

        # Try to load with trimesh first (supports many formats)
        # Don't force='mesh' so we can also load pointclouds
        loaded = trimesh.load(file_path)

        print(f"[load_mesh_file] Loaded type: {type(loaded).__name__}")

        # Handle pointclouds (PLY files with only vertices, no faces)
        if isinstance(loaded, trimesh.PointCloud):
            print(f"[load_mesh_file] Loaded pointcloud: {len(loaded.vertices)} points")
            # Store file metadata
            loaded.metadata['file_path'] = file_path
            loaded.metadata['file_name'] = os.path.basename(file_path)
            loaded.metadata['file_format'] = os.path.splitext(file_path)[1].lower()
            loaded.metadata['is_pointcloud'] = True
            print(f"[load_mesh_file] [OK] Successfully loaded pointcloud: {len(loaded.vertices)} points")
            return loaded, ""

        # Handle case where trimesh.load returns a Scene instead of a mesh
        if isinstance(loaded, trimesh.Scene):
            print(f"[load_mesh_file] Converting Scene to single mesh (scene has {len(loaded.geometry)} geometries)")
            # If it's a scene, dump it to a single mesh
            mesh = loaded.dump(concatenate=True)
        else:
            mesh = loaded

        if mesh is None or len(mesh.vertices) == 0:
            return None, f"Failed to read mesh or mesh is empty: {file_path}"

        # Check if it's actually a pointcloud (mesh with no faces)
        if not hasattr(mesh, 'faces') or mesh.faces is None or len(mesh.faces) == 0:
            # Convert to PointCloud
            pointcloud = trimesh.PointCloud(vertices=mesh.vertices)
            pointcloud.metadata['file_path'] = file_path
            pointcloud.metadata['file_name'] = os.path.basename(file_path)
            pointcloud.metadata['file_format'] = os.path.splitext(file_path)[1].lower()
            pointcloud.metadata['is_pointcloud'] = True
            print(f"[load_mesh_file] [OK] Successfully loaded as pointcloud: {len(pointcloud.vertices)} points")
            return pointcloud, ""

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

        # Merge duplicate vertices and clean up (handle API changes in newer trimesh versions)
        if hasattr(mesh, 'merge_vertices'):
            mesh.merge_vertices()

        # Try different API names for removing duplicate faces (changed in newer trimesh)
        if hasattr(mesh, 'remove_duplicate_faces'):
            mesh.remove_duplicate_faces()
        elif hasattr(mesh, 'update_faces'):
            # Newer trimesh uses update_faces with a mask
            pass  # Skip - mesh should already be clean from trimesh.load

        # Try different API names for removing degenerate faces
        if hasattr(mesh, 'remove_degenerate_faces'):
            mesh.remove_degenerate_faces()
        elif hasattr(mesh, 'nondegenerate_faces'):
            # Newer API: get mask of non-degenerate faces and update
            mask = mesh.nondegenerate_faces()
            if not mask.all():
                mesh.update_faces(mask)

        verts_after = len(mesh.vertices)
        faces_after = len(mesh.faces)

        if verts_before != verts_after or faces_before != faces_after:
            print(f"[load_mesh_file] Cleanup: {verts_before}->{verts_after} vertices, {faces_before}->{faces_after} faces")
            print(f"[load_mesh_file]   Removed: {verts_before - verts_after} duplicate vertices, {faces_before - faces_after} bad faces")

        # Store file metadata
        mesh.metadata['file_path'] = file_path
        mesh.metadata['file_name'] = os.path.basename(file_path)
        mesh.metadata['file_format'] = os.path.splitext(file_path)[1].lower()

        print(f"[load_mesh_file] Successfully loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        return mesh, ""

    except Exception as e:
        print(f"[load_mesh_file] Trimesh failed: {str(e)}, trying libigl fallback...")
        # Fallback to libigl
        if not IGL_AVAILABLE:
            return None, f"Failed to load mesh with trimesh: {str(e)}. libigl fallback not available."
        try:
            v, f = igl.read_triangle_mesh(file_path)
            if v is None or f is None or len(v) == 0 or len(f) == 0:
                return None, f"Failed to read mesh: {file_path}"

            print(f"[load_mesh_file] libigl loaded: {len(v)} vertices, {len(f)} faces")

            mesh = trimesh.Trimesh(vertices=v, faces=f, process=True)

            # Count before cleanup
            verts_before = len(mesh.vertices)
            faces_before = len(mesh.faces)

            # Clean up the mesh (handle API changes in newer trimesh versions)
            if hasattr(mesh, 'merge_vertices'):
                mesh.merge_vertices()

            if hasattr(mesh, 'remove_duplicate_faces'):
                mesh.remove_duplicate_faces()

            if hasattr(mesh, 'remove_degenerate_faces'):
                mesh.remove_degenerate_faces()
            elif hasattr(mesh, 'nondegenerate_faces'):
                mask = mesh.nondegenerate_faces()
                if not mask.all():
                    mesh.update_faces(mask)

            verts_after = len(mesh.vertices)
            faces_after = len(mesh.faces)

            if verts_before != verts_after or faces_before != faces_after:
                print(f"[load_mesh_file] Cleanup: {verts_before}->{verts_after} vertices, {faces_before}->{faces_after} faces")

            # Store metadata
            mesh.metadata['file_path'] = file_path
            mesh.metadata['file_name'] = os.path.basename(file_path)
            mesh.metadata['file_format'] = os.path.splitext(file_path)[1].lower()

            print(f"[load_mesh_file] Successfully loaded via libigl: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            return mesh, ""
        except Exception as e2:
            print(f"[load_mesh_file] Both loaders failed!")
            return None, f"Error loading mesh: {str(e)}; Fallback error: {str(e2)}"


def save_mesh_file(mesh, file_path: str) -> Tuple[bool, str]:
    """
    Save a mesh or point cloud to file.

    Args:
        mesh: Trimesh or PointCloud object
        file_path: Output file path

    Returns:
        Tuple of (success, error_message)
    """
    # Check for valid trimesh types (Trimesh or PointCloud)
    is_point_cloud = isinstance(mesh, trimesh.PointCloud)
    is_trimesh = isinstance(mesh, trimesh.Trimesh)

    if not is_trimesh and not is_point_cloud:
        return False, "Input must be a trimesh.Trimesh or trimesh.PointCloud object"

    if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
        return False, "Geometry has no vertices"

    # For meshes (not point clouds), check for faces
    if is_trimesh and len(mesh.faces) == 0:
        # Treat as point cloud - convert to PointCloud for proper export
        is_point_cloud = True

    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Handle VTP format specially - preserves vertex/face attributes (e.g., cad_face_id)
        if file_path.lower().endswith('.vtp'):
            from ..visualization._vtp_export import export_mesh_with_scalars_vtp
            export_mesh_with_scalars_vtp(mesh, file_path)
            return True, ""

        # Point cloud export - use PLY format
        if is_point_cloud:
            # For Trimesh with 0 faces, convert to PointCloud
            if is_trimesh and len(mesh.faces) == 0:
                # Get colors if available
                colors = None
                if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
                    colors = mesh.visual.vertex_colors
                point_cloud = trimesh.PointCloud(mesh.vertices, colors=colors)
                point_cloud.export(file_path)
            else:
                # Already a PointCloud
                mesh.export(file_path)
            return True, ""

        # Default: use trimesh export for meshes
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


def extract_visual_info(mesh: trimesh.Trimesh) -> dict:
    """
    Extract comprehensive visual and material information from a mesh.

    Args:
        mesh: Trimesh object

    Returns:
        Dictionary with visual information
    """
    info = {
        'visual_type': 'none',
        'has_material': False,
        'material_type': None,
        'has_uv': False,
        'uv_count': 0,
        'uv_range_u': None,
        'uv_range_v': None,
        'has_vertex_colors': False,
        'has_face_colors': False,
        'texture_dimensions': None,
        'texture_format': None,
    }

    if not hasattr(mesh, 'visual') or mesh.visual is None:
        return info

    visual = mesh.visual

    # Determine visual type
    visual_class = type(visual).__name__
    if 'TextureVisuals' in visual_class:
        info['visual_type'] = 'texture'
    elif 'ColorVisuals' in visual_class:
        info['visual_type'] = 'color'
    else:
        info['visual_type'] = visual_class.lower()

    # Check for material
    if hasattr(visual, 'material') and visual.material is not None:
        info['has_material'] = True
        info['material_type'] = type(visual.material).__name__

    # Check for UV coordinates
    if hasattr(visual, 'uv') and visual.uv is not None and len(visual.uv) > 0:
        info['has_uv'] = True
        info['uv_count'] = len(visual.uv)
        uv_array = np.array(visual.uv)
        info['uv_range_u'] = (float(uv_array[:, 0].min()), float(uv_array[:, 0].max()))
        info['uv_range_v'] = (float(uv_array[:, 1].min()), float(uv_array[:, 1].max()))

    # Check for vertex colors
    if hasattr(visual, 'vertex_colors') and visual.vertex_colors is not None and len(visual.vertex_colors) > 0:
        info['has_vertex_colors'] = True

    # Check for face colors
    if hasattr(visual, 'face_colors') and visual.face_colors is not None and len(visual.face_colors) > 0:
        info['has_face_colors'] = True

    # Check for texture image
    if hasattr(visual, 'material') and visual.material is not None:
        material = visual.material
        # Try to get texture image from various sources
        texture_image = None
        if hasattr(material, 'image') and material.image is not None:
            texture_image = material.image
        elif hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
            texture_image = material.baseColorTexture

        if texture_image is not None:
            # PIL Image object
            if hasattr(texture_image, 'size'):
                info['texture_dimensions'] = texture_image.size
                info['texture_format'] = texture_image.format or 'unknown'

    return info


def extract_pbr_properties(material) -> dict:
    """
    Extract PBR material properties from a trimesh material object.

    Args:
        material: trimesh material object (e.g., PBRMaterial, SimpleMaterial)

    Returns:
        Dictionary with PBR properties
    """
    props = {
        'has_base_color_texture': False,
        'has_metallic_roughness_texture': False,
        'has_normal_texture': False,
        'has_occlusion_texture': False,
        'has_emissive_texture': False,
        'metallic_factor': None,
        'roughness_factor': None,
        'base_color_factor': None,
        'emissive_factor': None,
        'alpha_mode': None,
        'alpha_cutoff': None,
        'double_sided': None,
    }

    if material is None:
        return props

    # Base color texture
    if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
        props['has_base_color_texture'] = True

    # Metallic/roughness texture
    if hasattr(material, 'metallicRoughnessTexture') and material.metallicRoughnessTexture is not None:
        props['has_metallic_roughness_texture'] = True

    # Normal texture
    if hasattr(material, 'normalTexture') and material.normalTexture is not None:
        props['has_normal_texture'] = True

    # Occlusion texture
    if hasattr(material, 'occlusionTexture') and material.occlusionTexture is not None:
        props['has_occlusion_texture'] = True

    # Emissive texture
    if hasattr(material, 'emissiveTexture') and material.emissiveTexture is not None:
        props['has_emissive_texture'] = True

    # Metallic factor
    if hasattr(material, 'metallicFactor'):
        props['metallic_factor'] = material.metallicFactor

    # Roughness factor
    if hasattr(material, 'roughnessFactor'):
        props['roughness_factor'] = material.roughnessFactor

    # Base color factor
    if hasattr(material, 'baseColorFactor'):
        props['base_color_factor'] = material.baseColorFactor

    # Emissive factor
    if hasattr(material, 'emissiveFactor'):
        props['emissive_factor'] = material.emissiveFactor

    # Alpha mode
    if hasattr(material, 'alphaMode'):
        props['alpha_mode'] = material.alphaMode

    # Alpha cutoff
    if hasattr(material, 'alphaCutoff'):
        props['alpha_cutoff'] = material.alphaCutoff

    # Double sided
    if hasattr(material, 'doubleSided'):
        props['double_sided'] = material.doubleSided

    return props


def extract_custom_attributes(mesh: trimesh.Trimesh) -> dict:
    """
    Extract custom vertex and face attributes from a mesh.

    Args:
        mesh: Trimesh object

    Returns:
        Dictionary with attribute information
    """
    attrs = {
        'vertex_attributes': {},
        'face_attributes': {},
    }

    # Vertex attributes
    if hasattr(mesh, 'vertex_attributes') and mesh.vertex_attributes:
        for name, values in mesh.vertex_attributes.items():
            attr_info = {
                'count': len(values),
                'dtype': str(values.dtype) if hasattr(values, 'dtype') else 'unknown',
                'shape': values.shape if hasattr(values, 'shape') else None,
            }
            # Add value range if numeric
            if hasattr(values, 'dtype') and np.issubdtype(values.dtype, np.number):
                attr_info['min'] = float(np.min(values))
                attr_info['max'] = float(np.max(values))
            attrs['vertex_attributes'][name] = attr_info

    # Face attributes
    if hasattr(mesh, 'face_attributes') and mesh.face_attributes:
        for name, values in mesh.face_attributes.items():
            attr_info = {
                'count': len(values),
                'dtype': str(values.dtype) if hasattr(values, 'dtype') else 'unknown',
                'shape': values.shape if hasattr(values, 'shape') else None,
            }
            # Add value range if numeric
            if hasattr(values, 'dtype') and np.issubdtype(values.dtype, np.number):
                attr_info['min'] = float(np.min(values))
                attr_info['max'] = float(np.max(values))
            attrs['face_attributes'][name] = attr_info

    return attrs


def compute_mesh_info(mesh: trimesh.Trimesh) -> str:
    """
    Generate a formatted string with comprehensive mesh information including PBR materials.

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

    # Visual & Material information
    visual_info = extract_visual_info(mesh)
    info += "=== Visual & Material ===\n\n"
    info += f"Visual Type: {visual_info['visual_type']}\n"
    info += f"Has Material: {visual_info['has_material']}\n"
    if visual_info['material_type']:
        info += f"Material Type: {visual_info['material_type']}\n"
    info += f"UV Coordinates: {'Yes' if visual_info['has_uv'] else 'No'}"
    if visual_info['has_uv']:
        info += f" ({visual_info['uv_count']:,} entries)\n"
        if visual_info['uv_range_u'] and visual_info['uv_range_v']:
            info += f"  UV Range: U[{visual_info['uv_range_u'][0]:.3f}, {visual_info['uv_range_u'][1]:.3f}], "
            info += f"V[{visual_info['uv_range_v'][0]:.3f}, {visual_info['uv_range_v'][1]:.3f}]\n"
    else:
        info += "\n"
    info += f"Vertex Colors: {'Yes' if visual_info['has_vertex_colors'] else 'No'}\n"
    info += f"Face Colors: {'Yes' if visual_info['has_face_colors'] else 'No'}\n"
    if visual_info['texture_dimensions']:
        info += f"Texture Dimensions: {visual_info['texture_dimensions'][0]}x{visual_info['texture_dimensions'][1]}\n"
        if visual_info['texture_format']:
            info += f"Texture Format: {visual_info['texture_format']}\n"
    info += "\n"

    # PBR Material Properties
    if visual_info['has_material'] and hasattr(mesh.visual, 'material'):
        pbr_props = extract_pbr_properties(mesh.visual.material)
        info += "=== PBR Material Properties ===\n\n"
        info += f"Base Color Texture: {'Yes' if pbr_props['has_base_color_texture'] else 'No'}\n"
        info += f"Metallic/Roughness Texture: {'Yes' if pbr_props['has_metallic_roughness_texture'] else 'No'}\n"
        info += f"Normal Map: {'Yes' if pbr_props['has_normal_texture'] else 'No'}\n"
        info += f"Occlusion Texture: {'Yes' if pbr_props['has_occlusion_texture'] else 'No'}\n"
        info += f"Emissive Texture: {'Yes' if pbr_props['has_emissive_texture'] else 'No'}\n"

        if pbr_props['metallic_factor'] is not None:
            info += f"Metallic Factor: {pbr_props['metallic_factor']:.3f}\n"
        if pbr_props['roughness_factor'] is not None:
            info += f"Roughness Factor: {pbr_props['roughness_factor']:.3f}\n"
        if pbr_props['base_color_factor'] is not None:
            bcf = pbr_props['base_color_factor']
            if hasattr(bcf, '__len__') and len(bcf) >= 3:
                info += f"Base Color Factor: [{bcf[0]:.3f}, {bcf[1]:.3f}, {bcf[2]:.3f}"
                if len(bcf) >= 4:
                    info += f", {bcf[3]:.3f}]\n"
                else:
                    info += "]\n"
            else:
                info += f"Base Color Factor: {bcf}\n"
        if pbr_props['emissive_factor'] is not None:
            ef = pbr_props['emissive_factor']
            if hasattr(ef, '__len__') and len(ef) >= 3:
                info += f"Emissive Factor: [{ef[0]:.3f}, {ef[1]:.3f}, {ef[2]:.3f}]\n"
            else:
                info += f"Emissive Factor: {ef}\n"
        if pbr_props['alpha_mode'] is not None:
            info += f"Alpha Mode: {pbr_props['alpha_mode']}\n"
        if pbr_props['alpha_cutoff'] is not None:
            info += f"Alpha Cutoff: {pbr_props['alpha_cutoff']:.3f}\n"
        if pbr_props['double_sided'] is not None:
            info += f"Double Sided: {pbr_props['double_sided']}\n"
        info += "\n"

    # Custom Attributes
    custom_attrs = extract_custom_attributes(mesh)
    info += "=== Custom Attributes ===\n\n"

    if custom_attrs['vertex_attributes']:
        info += "Vertex Attributes:\n"
        for name, attr in custom_attrs['vertex_attributes'].items():
            info += f"  {name}: {attr['dtype']}"
            if attr['shape']:
                info += f" {attr['shape']}"
            if 'min' in attr and 'max' in attr:
                info += f" range=[{attr['min']:.3f}, {attr['max']:.3f}]"
            info += "\n"
    else:
        info += "Vertex Attributes: (none)\n"

    if custom_attrs['face_attributes']:
        info += "Face Attributes:\n"
        for name, attr in custom_attrs['face_attributes'].items():
            info += f"  {name}: {attr['dtype']}"
            if attr['shape']:
                info += f" {attr['shape']}"
            if 'min' in attr and 'max' in attr:
                info += f" range=[{attr['min']:.3f}, {attr['max']:.3f}]"
            info += "\n"
    else:
        info += "Face Attributes: (none)\n"

    # Vertex normals
    if mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0:
        info += f"\nVertex Normals: Yes ({len(mesh.vertex_normals):,} vectors)\n"

    # Metadata
    if mesh.metadata:
        info += "\n=== Metadata ===\n\n"
        for key, value in mesh.metadata.items():
            info += f"  {key}: {value}\n"

    return info


def transfer_texture_via_closest_point(original_mesh: trimesh.Trimesh,
                                       remeshed_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Transfer texture from original mesh to remeshed mesh using closest-point projection.

    For each vertex on the remeshed mesh:
    1. Find closest point on original mesh surface
    2. Determine which triangle contains that point
    3. Compute barycentric coordinates within the triangle
    4. Interpolate original UV coordinates using barycentric weights
    5. Sample original texture at interpolated UV position
    6. Store sampled color as vertex color on remeshed mesh

    Args:
        original_mesh: Original mesh with texture (must have visual.uv and visual.material)
        remeshed_mesh: Remeshed mesh (will receive vertex colors)

    Returns:
        Remeshed mesh with vertex colors from texture transfer
    """
    print(f"[transfer_texture] Starting texture transfer via closest-point projection")
    print(f"[transfer_texture] Original: {len(original_mesh.vertices)} verts, {len(original_mesh.faces)} faces")
    print(f"[transfer_texture] Remeshed: {len(remeshed_mesh.vertices)} verts, {len(remeshed_mesh.faces)} faces")

    # Check original mesh has texture data
    if not hasattr(original_mesh, 'visual') or original_mesh.visual is None:
        raise ValueError("Original mesh has no visual data")

    if not hasattr(original_mesh.visual, 'uv') or original_mesh.visual.uv is None:
        raise ValueError("Original mesh has no UV coordinates")

    if not hasattr(original_mesh.visual, 'material') or original_mesh.visual.material is None:
        raise ValueError("Original mesh has no material")

    # Get texture image
    texture_image = None
    if hasattr(original_mesh.visual.material, 'baseColorTexture'):
        texture_image = original_mesh.visual.material.baseColorTexture
    elif hasattr(original_mesh.visual.material, 'image'):
        texture_image = original_mesh.visual.material.image

    if texture_image is None:
        raise ValueError("Original mesh material has no texture image")

    print(f"[transfer_texture] Original texture size: {texture_image.size}")

    # Convert texture to numpy array for fast sampling
    texture_array = np.array(texture_image)
    tex_height, tex_width = texture_array.shape[:2]
    print(f"[transfer_texture] Texture array shape: {texture_array.shape}")

    # Get original UVs
    original_uvs = original_mesh.visual.uv
    print(f"[transfer_texture] Original UVs: {len(original_uvs)} entries")

    # Step 1: Find closest point on original mesh for each remeshed vertex
    print(f"[transfer_texture] Finding closest points...")
    closest_points, distances, triangle_ids = original_mesh.nearest.on_surface(remeshed_mesh.vertices)

    print(f"[transfer_texture] Closest points found, max distance: {distances.max():.6f}")

    # Step 2: Get barycentric coordinates of closest points within their triangles
    print(f"[transfer_texture] Computing barycentric coordinates...")
    triangles = original_mesh.vertices[original_mesh.faces[triangle_ids]]
    bary_coords = trimesh.triangles.points_to_barycentric(triangles, closest_points)

    # Step 3: Interpolate original UVs using barycentric coordinates
    print(f"[transfer_texture] Interpolating UV coordinates...")
    triangle_uvs = original_uvs[original_mesh.faces[triangle_ids]]  # Shape: (N, 3, 2)
    interpolated_uvs = np.einsum('ij,ijk->ik', bary_coords, triangle_uvs)  # Shape: (N, 2)

    # Clamp UVs to [0, 1] range
    interpolated_uvs = np.clip(interpolated_uvs, 0.0, 1.0)

    print(f"[transfer_texture] UV range: U[{interpolated_uvs[:, 0].min():.3f}, {interpolated_uvs[:, 0].max():.3f}], "
          f"V[{interpolated_uvs[:, 1].min():.3f}, {interpolated_uvs[:, 1].max():.3f}]")

    # Step 4: Sample texture at interpolated UV positions
    print(f"[transfer_texture] Sampling texture...")

    # Convert UV [0,1] to pixel coordinates
    # UV convention: (0,0) = bottom-left, but image array is top-left origin
    pixel_x = (interpolated_uvs[:, 0] * (tex_width - 1)).astype(int)
    pixel_y = ((1.0 - interpolated_uvs[:, 1]) * (tex_height - 1)).astype(int)  # Flip V

    # Clamp to image bounds
    pixel_x = np.clip(pixel_x, 0, tex_width - 1)
    pixel_y = np.clip(pixel_y, 0, tex_height - 1)

    # Sample colors from texture
    vertex_colors = texture_array[pixel_y, pixel_x]  # Shape: (N, 3) or (N, 4)

    # Ensure we have RGBA (add alpha channel if missing)
    if vertex_colors.shape[1] == 3:
        alpha = np.full((len(vertex_colors), 1), 255, dtype=vertex_colors.dtype)
        vertex_colors = np.hstack([vertex_colors, alpha])

    # Check how many colors are non-black
    non_black = np.sum((vertex_colors[:, 0] > 10) | (vertex_colors[:, 1] > 10) | (vertex_colors[:, 2] > 10))
    print(f"[transfer_texture] Non-black vertices: {non_black}/{len(vertex_colors)} ({100*non_black/len(vertex_colors):.1f}%)")

    # Step 5: Create a copy of remeshed mesh and assign vertex colors
    result_mesh = remeshed_mesh.copy()
    result_mesh.visual.vertex_colors = vertex_colors

    print(f"[transfer_texture] Texture transfer complete")

    return result_mesh


# ============================================================================
# Remeshing via PyMeshLab
# ============================================================================

def pymeshlab_isotropic_remesh(
    mesh: trimesh.Trimesh,
    target_edge_length: float,
    iterations: int = 3,
    adaptive: bool = False,
    feature_angle: float = 30.0
) -> Tuple[Optional[trimesh.Trimesh], str]:
    """
    Apply isotropic remeshing to create uniform triangle sizes using PyMeshLab.

    Args:
        mesh: Input trimesh object
        target_edge_length: Target edge length for remeshed triangles
        iterations: Number of remeshing iterations (default: 3)
        adaptive: Use curvature-adaptive edge lengths (default: False)
        feature_angle: Angle threshold in degrees for feature edge detection (default: 30.0)

    Returns:
        Tuple of (remeshed_mesh, error_message)
    """
    print(f"[pymeshlab_isotropic_remesh] ===== Starting Isotropic Remeshing =====")
    print(f"[pymeshlab_isotropic_remesh] Input mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"[pymeshlab_isotropic_remesh] Parameters:")
    print(f"[pymeshlab_isotropic_remesh]   target_edge_length: {target_edge_length}")
    print(f"[pymeshlab_isotropic_remesh]   iterations: {iterations}")
    print(f"[pymeshlab_isotropic_remesh]   adaptive: {adaptive}")
    print(f"[pymeshlab_isotropic_remesh]   feature_angle: {feature_angle}")

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
                adaptive=adaptive,
                featuredeg=feature_angle
            )
        except AttributeError:
            # Older PyMeshLab versions use 'remeshing_' prefix
            try:
                ms.remeshing_isotropic_explicit_remeshing(
                    targetlen=pymeshlab.PercentageValue(target_pct),
                    iterations=iterations,
                    adaptive=adaptive,
                    featuredeg=feature_angle
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
            'adaptive': adaptive,
            'feature_angle': feature_angle,
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
        points = CGAL_Polygon_mesh_processing.Point_3_Vector()
        points.reserve(len(mesh.vertices))
        for v in mesh.vertices:
            points.append(Point_3(float(v[0]), float(v[1]), float(v[2])))

        # Create plain Python list of lists for faces
        # Note: Polygon_Vector doesn't properly convert to std::vector<std::vector<int>>
        # Using plain Python list works correctly with the SWIG bindings
        polygons = [[int(idx) for idx in face] for face in mesh.faces]

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


# CuMesh availability check
try:
    import cumesh as CuMesh
    import torch
    CUMESH_AVAILABLE = True
except ImportError:
    CUMESH_AVAILABLE = False


def cumesh_dc_remesh(
    mesh: trimesh.Trimesh,
    grid_resolution: int = 128,
    fill_holes_first: bool = True,
    band: float = 1.0,
) -> Tuple[Optional[trimesh.Trimesh], str]:
    """
    GPU-accelerated dual-contouring remeshing using CuMesh.

    This method voxelizes the mesh and reconstructs it using dual contouring,
    producing a clean, manifold mesh with uniform triangle sizes.

    Uses the same algorithm as TRELLIS2: CuMesh.remeshing.remesh_narrow_band_dc()

    Args:
        mesh: Input trimesh.Trimesh
        grid_resolution: Voxel grid resolution (higher = more detail, default 128)
        fill_holes_first: Fill holes before remeshing (default True)
        band: Narrow band width for dual contouring (default 1.0)

    Returns:
        Tuple of (remeshed_mesh, error_message)
        - remeshed_mesh: The remeshed trimesh.Trimesh, or None on error
        - error_message: Empty string on success, error description on failure
    """
    if not CUMESH_AVAILABLE:
        return None, "CuMesh not available. Install with CUDA support."

    try:
        print(f"[cumesh_dc_remesh] ===== CuMesh Dual-Contouring Remesh =====")
        print(f"[cumesh_dc_remesh] Input: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"[cumesh_dc_remesh] Grid resolution: {grid_resolution}")
        print(f"[cumesh_dc_remesh] Fill holes first: {fill_holes_first}")
        print(f"[cumesh_dc_remesh] Band width: {band}")

        # Convert to GPU tensors
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32).cuda()
        faces = torch.tensor(mesh.faces, dtype=torch.int32).cuda()

        # Calculate bounding box and scale (same approach as TRELLIS2)
        bbox_min = vertices.min(dim=0).values
        bbox_max = vertices.max(dim=0).values
        bbox_size = bbox_max - bbox_min
        scale = bbox_size.max().item()

        # Center the mesh
        center = (bbox_min + bbox_max) / 2
        vertices_centered = vertices - center

        # Initialize CuMesh for pre-processing
        cumesh = CuMesh.CuMesh()
        cumesh.init(vertices_centered, faces)
        print(f"[cumesh_dc_remesh] Initialized CuMesh on GPU")

        # Pre-unify face orientations (fixes inconsistent winding)
        cumesh.unify_face_orientations()
        print(f"[cumesh_dc_remesh] Unified face orientations (pre-remesh)")

        # Optionally fill holes before remeshing
        if fill_holes_first:
            cumesh.fill_holes()
            print(f"[cumesh_dc_remesh] Filled holes")

        # Read current state after preprocessing
        curr_verts, curr_faces = cumesh.read()

        # Build BVH for the remeshing operation
        bvh = CuMesh.cuBVH(curr_verts, curr_faces)

        # Run dual-contouring remesh (same parameters as TRELLIS2 nodes_unwrap.py)
        print(f"[cumesh_dc_remesh] Running dual-contouring remesh...")
        new_verts, new_faces = CuMesh.remeshing.remesh_narrow_band_dc(
            curr_verts, curr_faces,
            center=torch.zeros(3, device='cuda'),  # Already centered above
            scale=(grid_resolution + 3 * band) / grid_resolution * scale,
            resolution=grid_resolution,
            band=band,
            project_back=0.0,
            verbose=True,
            bvh=bvh,
        )

        # Clean up BVH after remesh
        del bvh, curr_verts, curr_faces

        # After remesh we have ~5M faces - DON'T call unify_face_orientations here!
        # It crashes on large meshes. TRELLIS2 does unify AFTER simplify (on smaller mesh).
        # Caller (remesh.py _cumesh) should call unify after simplify.
        print(f"[cumesh_dc_remesh] After remesh: {len(new_verts)} vertices, {len(new_faces)} faces")
        print(f"[cumesh_dc_remesh] NOTE: Skipping unify_face_orientations (do it after simplify)")

        final_verts, final_faces = new_verts, new_faces

        # Restore center offset
        final_verts = final_verts + center

        # Create result mesh
        remeshed_mesh = trimesh.Trimesh(
            vertices=final_verts.cpu().numpy().astype(np.float32),
            faces=final_faces.cpu().numpy(),
            process=False
        )

        # Calculate statistics
        vertex_change = len(remeshed_mesh.vertices) - len(mesh.vertices)
        face_change = len(remeshed_mesh.faces) - len(mesh.faces)
        vertex_pct = (vertex_change / len(mesh.vertices)) * 100 if len(mesh.vertices) > 0 else 0
        face_pct = (face_change / len(mesh.faces)) * 100 if len(mesh.faces) > 0 else 0

        print(f"[cumesh_dc_remesh] ===== Remeshing Complete =====")
        print(f"[cumesh_dc_remesh] Results:")
        print(f"[cumesh_dc_remesh]   Vertices: {len(mesh.vertices)} -> {len(remeshed_mesh.vertices)} ({vertex_change:+d}, {vertex_pct:+.1f}%)")
        print(f"[cumesh_dc_remesh]   Faces:    {len(mesh.faces)} -> {len(remeshed_mesh.faces)} ({face_change:+d}, {face_pct:+.1f}%)")

        # Cleanup GPU memory (curr_verts, curr_faces already deleted after BVH)
        del cumesh, vertices, faces, vertices_centered
        del new_verts, new_faces, final_verts, final_faces
        torch.cuda.empty_cache()

        return remeshed_mesh, ""

    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = f"Error during CuMesh remesh: {str(e)}"
        print(f"[cumesh_dc_remesh] ERROR: {error_msg}")
        return None, error_msg
