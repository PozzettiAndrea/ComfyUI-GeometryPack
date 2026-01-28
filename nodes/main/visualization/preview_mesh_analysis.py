# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Preview mesh with analysis buttons for computing mesh quality fields.

Provides interactive buttons to compute:
- Open/boundary edges
- Connected components
- Self-intersections

Fields are added to the mesh and visualized in the VTK.js viewer.
"""

import trimesh as trimesh_module
import os
import uuid
import numpy as np

from ..._utils.mesh_ops import is_point_cloud, get_face_count, get_geometry_type
from ._vtp_export import export_mesh_with_scalars_vtp

try:
    import folder_paths
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except (ImportError, AttributeError):
    COMFYUI_OUTPUT_FOLDER = None

# Global mesh cache for API access
# Key: mesh_id, Value: (trimesh, current_filename, fields_added)
_MESH_CACHE = {}


def get_cached_mesh(mesh_id):
    """Get mesh from cache by ID."""
    return _MESH_CACHE.get(mesh_id)


def set_cached_mesh(mesh_id, mesh, filename):
    """Store mesh in cache."""
    _MESH_CACHE[mesh_id] = {
        'mesh': mesh,
        'filename': filename,
        'fields': []
    }


def add_field_to_cached_mesh(mesh_id, field_name):
    """Track that a field was added to the cached mesh."""
    if mesh_id in _MESH_CACHE:
        if field_name not in _MESH_CACHE[mesh_id]['fields']:
            _MESH_CACHE[mesh_id]['fields'].append(field_name)


def compute_boundary_vertices(mesh):
    """
    Compute boundary/open edge vertices.

    Returns mesh with 'boundary_vertex' vertex attribute:
    - 1.0 = vertex is on a boundary edge
    - 0.0 = vertex is interior
    """
    from trimesh.grouping import group_rows

    edges_sorted = mesh.edges_sorted

    # Find boundary edges (edges that appear only once)
    boundary_edge_indices = group_rows(edges_sorted, require_count=1)
    boundary_edges = edges_sorted[boundary_edge_indices]

    # Create vertex field
    boundary_field = np.zeros(len(mesh.vertices), dtype=np.float32)
    boundary_vertices = np.unique(boundary_edges.flatten())
    boundary_field[boundary_vertices] = 1.0

    mesh.vertex_attributes['boundary_vertex'] = boundary_field

    num_boundary = int(np.sum(boundary_field > 0.5))
    print(f"[MeshAnalysis] Open edges: {len(boundary_edges)} edges, {num_boundary} vertices")

    return mesh, num_boundary


def compute_connected_components(mesh):
    """
    Compute connected components.

    Returns mesh with 'part_id' face attribute.
    """
    components = trimesh_module.graph.connected_components(
        mesh.face_adjacency,
        nodes=np.arange(len(mesh.faces))
    )

    part_ids = np.zeros(len(mesh.faces), dtype=np.float32)
    for component_id, face_indices in enumerate(components):
        part_ids[face_indices] = float(component_id)

    mesh.face_attributes['part_id'] = part_ids

    print(f"[MeshAnalysis] Connected components: {len(components)}")

    return mesh, len(components)


def compute_self_intersections(mesh):
    """
    Compute self-intersecting faces.

    Returns mesh with 'self_intersect' face attribute:
    - 1.0 = face is involved in self-intersection
    - 0.0 = face is clean
    """
    try:
        import igl

        V = np.asarray(mesh.vertices, dtype=np.float64)
        F = np.asarray(mesh.faces, dtype=np.int32)

        # Find self-intersecting face pairs
        IF, _ = igl.self_intersect(V, F)

        # Mark all faces involved in intersections
        intersect_field = np.zeros(len(mesh.faces), dtype=np.float32)
        if len(IF) > 0:
            intersecting_faces = np.unique(IF.flatten())
            intersect_field[intersecting_faces] = 1.0

        mesh.face_attributes['self_intersect'] = intersect_field

        num_intersecting = int(np.sum(intersect_field > 0.5))
        print(f"[MeshAnalysis] Self-intersections: {num_intersecting} faces")

        return mesh, num_intersecting

    except ImportError:
        print("[MeshAnalysis] WARNING: libigl not available for self-intersection detection")
        # Fallback: mark no intersections
        mesh.face_attributes['self_intersect'] = np.zeros(len(mesh.faces), dtype=np.float32)
        return mesh, 0
    except Exception as e:
        print(f"[MeshAnalysis] Error computing self-intersections: {e}")
        mesh.face_attributes['self_intersect'] = np.zeros(len(mesh.faces), dtype=np.float32)
        return mesh, 0


class PreviewMeshAnalysisNode:
    """
    Preview mesh with analysis buttons.

    Displays mesh in VTK.js viewer with interactive buttons to compute:
    - Open edges (boundary vertices)
    - Connected components (part_id)
    - Self-intersections

    Click a button to compute and visualize that analysis.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_mesh_analysis"
    CATEGORY = "geompack/visualization"

    def preview_mesh_analysis(self, trimesh):
        """
        Export mesh and prepare for analysis preview.

        Args:
            trimesh: Input trimesh object

        Returns:
            dict: UI data for frontend widget with mesh_id for API calls
        """
        print(f"[PreviewMeshAnalysis] Preparing preview: {len(trimesh.vertices)} vertices, {get_face_count(trimesh)} faces")

        # Generate unique mesh ID for this execution
        mesh_id = uuid.uuid4().hex[:12]

        # Export to VTP for field visualization
        filename = f"analysis_{mesh_id}.vtp"

        if COMFYUI_OUTPUT_FOLDER:
            filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
        else:
            import tempfile
            filepath = os.path.join(tempfile.gettempdir(), filename)

        # Make a copy of the mesh for caching
        mesh_copy = trimesh.copy()

        # Export mesh
        try:
            export_mesh_with_scalars_vtp(mesh_copy, filepath)
            print(f"[PreviewMeshAnalysis] Exported VTP to: {filepath}")
        except Exception as e:
            print(f"[PreviewMeshAnalysis] VTP export failed: {e}, using STL")
            filename = f"analysis_{mesh_id}.stl"
            filepath = filepath.replace('.vtp', '.stl')
            mesh_copy.export(filepath, file_type='stl')

        # Cache mesh for API access
        set_cached_mesh(mesh_id, mesh_copy, filename)

        # Calculate bounds
        bounds = trimesh.bounds
        extents = trimesh.extents

        if extents is None or bounds is None:
            vertices_arr = np.asarray(trimesh.vertices)
            if len(vertices_arr) > 0:
                bounds = np.array([vertices_arr.min(axis=0), vertices_arr.max(axis=0)])
                extents = bounds[1] - bounds[0]
            else:
                bounds = np.array([[0, 0, 0], [1, 1, 1]])
                extents = np.array([1, 1, 1])

        max_extent = max(extents)
        is_watertight = False if is_point_cloud(trimesh) else trimesh.is_watertight

        # Get existing field names
        field_names = []
        if hasattr(trimesh, 'vertex_attributes') and trimesh.vertex_attributes:
            field_names.extend(list(trimesh.vertex_attributes.keys()))
        if hasattr(trimesh, 'face_attributes') and trimesh.face_attributes:
            field_names.extend([f"face.{k}" for k in trimesh.face_attributes.keys()])

        # Return metadata for frontend
        ui_data = {
            "mesh_file": [filename],
            "mesh_id": [mesh_id],  # For API calls
            "vertex_count": [len(trimesh.vertices)],
            "face_count": [get_face_count(trimesh)],
            "bounds_min": [bounds[0].tolist()],
            "bounds_max": [bounds[1].tolist()],
            "extents": [extents.tolist()],
            "max_extent": [float(max_extent)],
            "is_watertight": [bool(is_watertight)],
            "field_names": [field_names],
        }

        return {"ui": ui_data}


NODE_CLASS_MAPPINGS = {
    "GeomPackPreviewMeshAnalysis": PreviewMeshAnalysisNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackPreviewMeshAnalysis": "Preview Mesh (Analysis)",
}
