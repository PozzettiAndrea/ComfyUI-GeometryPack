import numpy as np
import trimesh
import os
from typing import Tuple, Optional


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
            pointcloud = trimesh.Trimesh(vertices=mesh.vertices)
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
        try:
            import igl
        except (ImportError, OSError):
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
