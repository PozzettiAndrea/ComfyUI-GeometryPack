# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Preview mesh with synchronized 3D and UV layout views.

Displays mesh in a split-pane interactive viewer:
- Left: 3D mesh view with wireframe overlay (rotatable)
- Right: 2D UV layout in 0-1 space
- Click on either view highlights corresponding point on the other

Useful for inspecting UV unwrapping quality, seam placement, and texture mapping.
"""

import trimesh as trimesh_module
import os
import tempfile
import uuid
import json
import numpy as np

from .mesh_helpers import is_point_cloud, get_face_count, get_geometry_type

try:
    import folder_paths
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except (ImportError, AttributeError):
    COMFYUI_OUTPUT_FOLDER = None


class PreviewMeshUVNode:
    """
    Preview mesh with synchronized 3D and UV layout views.

    Displays mesh in a split-pane interactive viewer:
    - Left: 3D mesh view with wireframe overlay (rotatable)
    - Right: 2D UV layout in 0-1 space
    - Click on either view highlights corresponding point on the other

    Useful for inspecting UV unwrapping quality, seam placement, and texture mapping.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
            "optional": {
                "show_checker": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply checker pattern to visualize UV distortion"
                }),
                "show_wireframe": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show mesh wireframe on 3D view"
                }),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_mesh_uv"
    CATEGORY = "geompack/visualization"

    def preview_mesh_uv(self, trimesh, show_checker=False, show_wireframe=True):
        """
        Export mesh and UV data for synchronized 3D + UV layout preview.

        Args:
            trimesh: Input trimesh_module.Trimesh object
            show_checker: Apply checker pattern texture
            show_wireframe: Show wireframe overlay on 3D mesh

        Returns:
            dict: UI data for frontend widget
        """
        print(f"[PreviewMeshUV] Preparing preview: {get_geometry_type(trimesh)} - {len(trimesh.vertices)} vertices, {get_face_count(trimesh)} faces")

        # Point clouds don't have UVs or faces
        if is_point_cloud(trimesh):
            print(f"[PreviewMeshUV] Warning: UV viewing not applicable to point clouds")
            return {
                "ui": {
                    "error": ["UV viewing requires a mesh with faces. This is a point cloud."]
                }
            }

        # Check for UV data
        has_uvs = False
        uv_data = None

        if hasattr(trimesh, 'visual') and trimesh.visual is not None:
            if hasattr(trimesh.visual, 'uv') and trimesh.visual.uv is not None:
                uvs = trimesh.visual.uv
                if len(uvs) > 0:
                    has_uvs = True
                    # Convert UV data to serializable format
                    uv_data = {
                        "uvs": uvs.tolist(),
                        "faces": trimesh.faces.tolist(),
                    }
                    print(f"[PreviewMeshUV] Found UV data: {len(uvs)} UV coordinates")

                    # Calculate UV statistics
                    uv_min = uvs.min(axis=0)
                    uv_max = uvs.max(axis=0)
                    uv_range = uv_max - uv_min

                    # Check if UVs are in 0-1 range
                    in_unit_square = bool(uv_min[0] >= 0 and uv_min[1] >= 0 and
                                         uv_max[0] <= 1 and uv_max[1] <= 1)

                    # Estimate UV coverage (area of UV triangles / unit square area)
                    try:
                        uv_area = 0.0
                        for face in trimesh.faces:
                            # Get UV coordinates for this face
                            uv0 = uvs[face[0]]
                            uv1 = uvs[face[1]]
                            uv2 = uvs[face[2]]
                            # Triangle area using cross product
                            v1 = uv1 - uv0
                            v2 = uv2 - uv0
                            uv_area += abs(v1[0] * v2[1] - v1[1] * v2[0]) / 2.0
                        uv_coverage = float(uv_area)
                    except Exception as e:
                        print(f"[PreviewMeshUV] Could not calculate UV coverage: {e}")
                        uv_coverage = 0.0

                    print(f"[PreviewMeshUV] UV stats: range={uv_range}, in_unit_square={in_unit_square}, coverage={uv_coverage:.4f}")

        if not has_uvs:
            print(f"[PreviewMeshUV] WARNING: No UV data found. UV layout view will be empty.")

        # Generate unique filename for mesh
        mesh_filename = f"preview_uv_{uuid.uuid4().hex[:8]}.glb"

        # Use ComfyUI's output directory
        if COMFYUI_OUTPUT_FOLDER:
            mesh_filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, mesh_filename)
        else:
            mesh_filepath = os.path.join(tempfile.gettempdir(), mesh_filename)

        # Export mesh to GLB (preserves UVs)
        try:
            trimesh.export(mesh_filepath, file_type='glb')
            print(f"[PreviewMeshUV] Exported mesh to: {mesh_filepath}")
        except Exception as e:
            print(f"[PreviewMeshUV] GLB export failed: {e}")
            # Fallback to OBJ
            mesh_filename = mesh_filename.replace('.glb', '.obj')
            mesh_filepath = mesh_filepath.replace('.glb', '.obj')
            trimesh.export(mesh_filepath, file_type='obj')
            print(f"[PreviewMeshUV] Exported to OBJ: {mesh_filepath}")

        # Save UV data as JSON for the frontend
        uv_json_filename = None
        if uv_data:
            uv_json_filename = f"preview_uv_{uuid.uuid4().hex[:8]}_uvdata.json"
            if COMFYUI_OUTPUT_FOLDER:
                uv_json_filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, uv_json_filename)
            else:
                uv_json_filepath = os.path.join(tempfile.gettempdir(), uv_json_filename)

            with open(uv_json_filepath, 'w') as f:
                json.dump(uv_data, f)
            print(f"[PreviewMeshUV] Exported UV data to: {uv_json_filepath}")

        # Calculate bounding box info
        bounds = trimesh.bounds
        extents = trimesh.extents
        max_extent = max(extents)

        # Return metadata for frontend widget
        ui_data = {
            "mesh_file": [mesh_filename],
            "vertex_count": [len(trimesh.vertices)],
            "face_count": [get_face_count(trimesh)],
            "bounds_min": [bounds[0].tolist()],
            "bounds_max": [bounds[1].tolist()],
            "extents": [extents.tolist()],
            "max_extent": [float(max_extent)],
            "has_uvs": [has_uvs],
            "show_checker": [show_checker],
            "show_wireframe": [show_wireframe],
        }

        # Add UV-specific data if available
        if uv_json_filename:
            ui_data["uv_data_file"] = [uv_json_filename]
        if has_uvs:
            ui_data["uv_coverage"] = [uv_coverage]
            ui_data["uv_in_unit_square"] = [in_unit_square]
            ui_data["uv_min"] = [uv_min.tolist()]
            ui_data["uv_max"] = [uv_max.tolist()]

        print(f"[PreviewMeshUV] Preview ready: has_uvs={has_uvs}, checker={show_checker}, wireframe={show_wireframe}")

        return {"ui": ui_data}


NODE_CLASS_MAPPINGS = {
    "GeomPackPreviewMeshUV": PreviewMeshUVNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackPreviewMeshUV": "Preview Mesh UV",
}
