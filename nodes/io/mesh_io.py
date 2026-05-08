import logging
import numpy as np
import trimesh
import os
from typing import Tuple, Optional

log = logging.getLogger("geometrypack")


def _load_vtk_mesh(file_path: str) -> Tuple[Optional[trimesh.Trimesh], str]:
    """
    Load VTK format files (VTP, VTU, VTK) using pyvista.

    Args:
        file_path: Path to VTK format file

    Returns:
        Tuple of (mesh, error_message)
    """
    try:
        import pyvista as pv
    except (ImportError, OSError):
        return None, (
            f"VTK format files ({os.path.splitext(file_path)[1]}) require pyvista. "
            f"Install with: pip install pyvista"
        )

    try:
        log.info("Loading VTK format: %s", file_path)

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

        log.info("VTK mesh: %d vertices, %d faces", len(vertices), len(faces))

        # Create trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

        # Transfer scalar fields from VTK to trimesh attributes
        if hasattr(pv_mesh, 'point_data') and pv_mesh.point_data:
            for name in pv_mesh.point_data.keys():
                try:
                    data = np.array(pv_mesh.point_data[name])
                    if len(data) == len(vertices):
                        mesh.vertex_attributes[name] = data.astype(np.float32)
                        log.debug("Transferred vertex attribute: %s", name)
                except Exception:
                    pass

        if hasattr(pv_mesh, 'cell_data') and pv_mesh.cell_data:
            for name in pv_mesh.cell_data.keys():
                try:
                    data = np.array(pv_mesh.cell_data[name])
                    if len(data) == len(faces):
                        mesh.face_attributes[name] = data.astype(np.float32)
                        log.debug("Transferred face attribute: %s", name)
                except Exception:
                    pass

        # Store metadata
        mesh.metadata['file_path'] = file_path
        mesh.metadata['file_name'] = os.path.basename(file_path)
        mesh.metadata['file_format'] = os.path.splitext(file_path)[1].lower()

        log.info("Successfully loaded VTK: %d vertices, %d faces", len(mesh.vertices), len(mesh.faces))
        return mesh, ""

    except Exception as e:
        import traceback
        log.error("Error loading VTK file", exc_info=True)
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
        log.info("Loading: %s", file_path)

        # Try to load with trimesh first (supports many formats)
        # Don't force='mesh' so we can also load pointclouds
        loaded = trimesh.load(file_path)

        log.debug("Loaded type: %s", type(loaded).__name__)

        # Handle pointclouds (PLY files with only vertices, no faces)
        if isinstance(loaded, trimesh.PointCloud):
            log.info("Loaded pointcloud: %d points", len(loaded.vertices))
            # Store file metadata
            loaded.metadata['file_path'] = file_path
            loaded.metadata['file_name'] = os.path.basename(file_path)
            loaded.metadata['file_format'] = os.path.splitext(file_path)[1].lower()
            loaded.metadata['is_pointcloud'] = True
            log.info("Successfully loaded pointcloud: %d points", len(loaded.vertices))
            return loaded, ""

        # Handle case where trimesh.load returns a Scene instead of a mesh
        if isinstance(loaded, trimesh.Scene):
            log.info("Converting Scene to single mesh (scene has %d geometries)", len(loaded.geometry))
            # If it's a scene, dump it to a single mesh
            mesh = loaded.dump(concatenate=True)
        else:
            mesh = loaded

        if mesh is None or len(mesh.vertices) == 0:
            return None, f"Failed to read mesh or mesh is empty: {file_path}"

        # Check if it's actually a pointcloud (mesh with no faces)
        if not hasattr(mesh, 'faces') or mesh.faces is None or len(mesh.faces) == 0:
            # Convert to PointCloud
            pointcloud = trimesh.Trimesh(vertices=mesh.vertices)
            pointcloud.metadata['file_path'] = file_path
            pointcloud.metadata['file_name'] = os.path.basename(file_path)
            pointcloud.metadata['file_format'] = os.path.splitext(file_path)[1].lower()
            pointcloud.metadata['is_pointcloud'] = True
            log.info("Successfully loaded as pointcloud: %d points", len(pointcloud.vertices))
            return pointcloud, ""

        log.info("Initial mesh: %d vertices, %d faces", len(mesh.vertices), len(mesh.faces))

        # Ensure mesh is properly triangulated
        # Trimesh should handle this, but some file formats might have issues
        if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
            # Check if faces are triangular
            if mesh.faces.shape[1] != 3:
                # Need to triangulate - this shouldn't normally happen but handle it
                log.warning("Mesh has non-triangular faces, triangulating...")
                # trimesh.Trimesh constructor should triangulate automatically with process=True
                mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
                log.info("After triangulation: %d vertices, %d faces", len(mesh.vertices), len(mesh.faces))

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
            log.info("Cleanup: %d->%d vertices, %d->%d faces", verts_before, verts_after, faces_before, faces_after)
            log.info("Removed: %d duplicate vertices, %d bad faces", verts_before - verts_after, faces_before - faces_after)

        # Store file metadata
        mesh.metadata['file_path'] = file_path
        mesh.metadata['file_name'] = os.path.basename(file_path)
        mesh.metadata['file_format'] = os.path.splitext(file_path)[1].lower()

        log.info("Successfully loaded: %d vertices, %d faces", len(mesh.vertices), len(mesh.faces))
        return mesh, ""

    except Exception as e:
        log.warning("Trimesh failed: %s, trying libigl fallback...", e)
        # Fallback to libigl
        try:
            import igl
        except (ImportError, OSError):
            return None, f"Failed to load mesh with trimesh: {str(e)}. libigl fallback not available."
        try:
            v, f = igl.read_triangle_mesh(file_path)
            if v is None or f is None or len(v) == 0 or len(f) == 0:
                return None, f"Failed to read mesh: {file_path}"

            log.info("libigl loaded: %d vertices, %d faces", len(v), len(f))

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
                log.info("Cleanup: %d->%d vertices, %d->%d faces", verts_before, verts_after, faces_before, faces_after)

            # Store metadata
            mesh.metadata['file_path'] = file_path
            mesh.metadata['file_name'] = os.path.basename(file_path)
            mesh.metadata['file_format'] = os.path.splitext(file_path)[1].lower()

            log.info("Successfully loaded via libigl: %d vertices, %d faces", len(mesh.vertices), len(mesh.faces))
            return mesh, ""
        except Exception as e2:
            log.error("Both loaders failed!")
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
    is_pc = isinstance(mesh, trimesh.PointCloud)
    is_trimesh = isinstance(mesh, trimesh.Trimesh)

    if not is_trimesh and not is_pc:
        return False, "Input must be a trimesh.Trimesh or trimesh.PointCloud object"

    if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
        return False, "Geometry has no vertices"

    # For meshes (not point clouds), check for faces
    if is_trimesh and len(mesh.faces) == 0:
        # Treat as point cloud - convert to PointCloud for proper export
        is_pc = True

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
        if is_pc:
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
