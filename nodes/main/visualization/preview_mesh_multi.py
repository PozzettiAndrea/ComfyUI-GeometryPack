# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Multi mesh preview with VTK.js - displays up to 4 meshes in a grid layout.

Grid layouts:
- 1 mesh: 1x1
- 2 meshes: 1x2 (side by side)
- 3 meshes: 1x3 (horizontal row)
- 4 meshes: 2x2

Supports scalar field visualization with synchronized cameras across viewports.
"""

import trimesh as trimesh_module
import numpy as np
import os
import tempfile
import uuid

from ..._utils.mesh_ops import is_point_cloud, get_face_count, get_geometry_type

from ._vtp_export import export_mesh_with_scalars_vtp

try:
    import folder_paths
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except (ImportError, AttributeError):
    COMFYUI_OUTPUT_FOLDER = None


def extract_field_names(mesh):
    """Extract all vertex and face attribute field names from a mesh."""
    field_names = []
    if hasattr(mesh, 'vertex_attributes') and mesh.vertex_attributes:
        field_names.extend(list(mesh.vertex_attributes.keys()))
    if hasattr(mesh, 'face_attributes') and mesh.face_attributes:
        field_names.extend([f"face.{k}" for k in mesh.face_attributes.keys()])
    return field_names


def has_fields(mesh):
    """Check if mesh has any vertex or face attributes."""
    has_vertex_attrs = hasattr(mesh, 'vertex_attributes') and len(mesh.vertex_attributes) > 0
    has_face_attrs = hasattr(mesh, 'face_attributes') and len(mesh.face_attributes) > 0
    return has_vertex_attrs or has_face_attrs


def get_texture_info(mesh):
    """Extract texture/visual information from a mesh."""
    has_visual = hasattr(mesh, 'visual') and mesh.visual is not None
    visual_kind = mesh.visual.kind if has_visual else None
    has_texture = visual_kind == 'texture' and hasattr(mesh.visual, 'material') if has_visual else False
    has_vertex_colors = visual_kind == 'vertex' if has_visual else False
    has_material = has_texture
    return {
        'has_visual': has_visual,
        'visual_kind': visual_kind,
        'has_texture': has_texture,
        'has_vertex_colors': has_vertex_colors,
        'has_material': has_material
    }


class PreviewMeshMultiNode:
    """
    Multi mesh preview with VTK.js - displays up to 4 meshes in a grid layout.

    Grid layouts:
    - 1 mesh: 1x1
    - 2 meshes: 1x2 (side by side)
    - 3 meshes: 1x3 (horizontal row)
    - 4 meshes: 2x2

    Supports scalar field visualization with synchronized cameras.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_1": ("TRIMESH",),
            },
            "optional": {
                "mesh_2": ("TRIMESH",),
                "mesh_3": ("TRIMESH",),
                "mesh_4": ("TRIMESH",),
                "mode": (["fields", "texture"], {"default": "fields"}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_multi"
    CATEGORY = "geompack/visualization"

    def preview_multi(self, mesh_1, mesh_2=None, mesh_3=None, mesh_4=None, mode="fields"):
        """
        Preview multiple meshes in a grid layout.

        Args:
            mesh_1: First mesh (required)
            mesh_2, mesh_3, mesh_4: Optional additional meshes
            mode: "fields" (scientific visualization) or "texture" (textured rendering)

        Returns:
            dict: UI data for frontend widget
        """
        # Collect all provided meshes
        meshes = [mesh_1]
        if mesh_2 is not None:
            meshes.append(mesh_2)
        if mesh_3 is not None:
            meshes.append(mesh_3)
        if mesh_4 is not None:
            meshes.append(mesh_4)

        num_meshes = len(meshes)
        print(f"[PreviewMeshMulti] Mode: {mode}, Meshes: {num_meshes}")

        # Generate unique ID for this preview
        preview_id = uuid.uuid4().hex[:8]

        # Export each mesh and collect metadata
        mesh_files = []
        vertex_counts = []
        face_counts = []
        bounds_list = []
        extents_list = []
        is_watertight_list = []
        field_names_list = []
        texture_info_list = []

        for i, mesh in enumerate(meshes):
            print(f"[PreviewMeshMulti] Mesh {i+1}: {get_geometry_type(mesh)} - {len(mesh.vertices)} vertices, {get_face_count(mesh)} faces")

            # Check for field data and texture info
            mesh_has_fields = has_fields(mesh)
            mesh_is_pc = is_point_cloud(mesh)
            texture_info = get_texture_info(mesh)

            # Export mesh
            if mode == "texture":
                filename = f"preview_multi_{i+1}_{preview_id}.glb"
            elif mesh_has_fields or mesh_is_pc:
                filename = f"preview_multi_{i+1}_{preview_id}.vtp"
            else:
                filename = f"preview_multi_{i+1}_{preview_id}.stl"

            if COMFYUI_OUTPUT_FOLDER:
                filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
            else:
                filepath = os.path.join(tempfile.gettempdir(), filename)

            try:
                if mode == "texture":
                    mesh.export(filepath, file_type='glb', include_normals=True)
                    print(f"[PreviewMeshMulti] Exported GLB: {filepath}")
                elif mesh_has_fields or mesh_is_pc:
                    export_mesh_with_scalars_vtp(mesh, filepath)
                    print(f"[PreviewMeshMulti] Exported VTP: {filepath}")
                else:
                    mesh.export(filepath, file_type='stl')
                    print(f"[PreviewMeshMulti] Exported STL: {filepath}")
            except Exception as e:
                print(f"[PreviewMeshMulti] Export failed: {e}, trying OBJ fallback")
                filename = f"preview_multi_{i+1}_{preview_id}.obj"
                filepath = os.path.join(COMFYUI_OUTPUT_FOLDER or tempfile.gettempdir(), filename)
                mesh.export(filepath, file_type='obj')

            # Collect metadata
            mesh_files.append(filename)
            vertex_counts.append(len(mesh.vertices))
            face_counts.append(get_face_count(mesh))

            bounds = np.array([mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)])
            extents = bounds[1] - bounds[0]
            bounds_list.append(bounds.tolist())
            extents_list.append(extents.tolist())

            is_watertight_list.append(bool(mesh.is_watertight) if not mesh_is_pc else False)
            field_names_list.append(extract_field_names(mesh))
            texture_info_list.append(texture_info)

        # Determine grid layout
        if num_meshes == 1:
            grid_cols, grid_rows = 1, 1
        elif num_meshes == 2:
            grid_cols, grid_rows = 2, 1
        elif num_meshes == 3:
            grid_cols, grid_rows = 3, 1
        else:  # 4
            grid_cols, grid_rows = 2, 2

        # Build UI data
        ui_data = {
            "mode": [mode],
            "num_meshes": [num_meshes],
            "grid_cols": [grid_cols],
            "grid_rows": [grid_rows],
            "mesh_files": [mesh_files],
            "vertex_counts": [vertex_counts],
            "face_counts": [face_counts],
            "bounds_list": [bounds_list],
            "extents_list": [extents_list],
            "is_watertight_list": [is_watertight_list],
        }

        # Add mode-specific metadata
        if mode == "texture":
            ui_data["texture_info_list"] = [[t for t in texture_info_list]]
        else:
            ui_data["field_names_list"] = [field_names_list]

        print(f"[PreviewMeshMulti] Grid: {grid_cols}x{grid_rows}, Preview ready")
        return {"ui": ui_data}


NODE_CLASS_MAPPINGS = {
    "GeomPackPreviewMeshMulti": PreviewMeshMultiNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackPreviewMeshMulti": "Preview Mesh Multi",
}
