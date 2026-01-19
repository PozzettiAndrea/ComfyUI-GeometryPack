# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Unified dual mesh preview with VTK.js - supports both side-by-side and overlay layouts.

Combines and enhances PreviewMeshVTKDual and PreviewMeshVTKSideBySide with full
field visualization support. Displays two meshes either:
- Side-by-side: Synchronized cameras in separate viewports
- Overlaid: Combined in single viewport with color coding

Supports scalar field visualization with shared colormap when meshes have fields.
"""

import trimesh as trimesh_module
import numpy as np
import os
import tempfile
import uuid

from .._utils.mesh_ops import is_point_cloud, get_face_count, get_geometry_type

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


class PreviewMeshDualNode:
    """
    Unified dual mesh preview with VTK.js - supports both side-by-side and overlay layouts.

    Combines two meshes for comparison with full field visualization support.
    Choose between synchronized side-by-side viewports or single overlaid viewport.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_1": ("TRIMESH",),
                "mesh_2": ("TRIMESH",),
            },
            "optional": {
                "layout": (["side_by_side", "overlay", "slider"], {"default": "side_by_side"}),
                "mode": (["fields", "texture"], {"default": "fields"}),
                "opacity_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "opacity_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_dual"
    CATEGORY = "geompack/visualization"

    def preview_dual(self, mesh_1, mesh_2, layout="side_by_side", mode="fields", opacity_1=1.0, opacity_2=1.0):
        """
        Preview two meshes with chosen layout and visualization mode.

        Args:
            mesh_1: First trimesh object
            mesh_2: Second trimesh object
            layout: "side_by_side" or "overlay"
            mode: "fields" (scientific visualization) or "texture" (textured rendering)
            opacity_1: Opacity for mesh 1 (0.0-1.0)
            opacity_2: Opacity for mesh 2 (0.0-1.0)

        Returns:
            dict: UI data for frontend widget
        """
        print(f"[PreviewMeshDual] Layout: {layout}, Mode: {mode}")
        print(f"[PreviewMeshDual] Mesh 1: {get_geometry_type(mesh_1)} - {len(mesh_1.vertices)} vertices, {get_face_count(mesh_1)} faces")
        print(f"[PreviewMeshDual] Mesh 2: {get_geometry_type(mesh_2)} - {len(mesh_2.vertices)} vertices, {get_face_count(mesh_2)} faces")

        # Check for field data
        mesh_1_has_fields = has_fields(mesh_1)
        mesh_2_has_fields = has_fields(mesh_2)
        field_names_1 = extract_field_names(mesh_1)
        field_names_2 = extract_field_names(mesh_2)
        common_fields = list(set(field_names_1) & set(field_names_2))

        print(f"[PreviewMeshDual] Mesh 1 fields: {field_names_1}")
        print(f"[PreviewMeshDual] Mesh 2 fields: {field_names_2}")
        print(f"[PreviewMeshDual] Common fields: {common_fields}")

        # Check for texture/visual data
        texture_info_1 = get_texture_info(mesh_1)
        texture_info_2 = get_texture_info(mesh_2)

        print(f"[PreviewMeshDual] Mesh 1 visual: kind={texture_info_1['visual_kind']}, texture={texture_info_1['has_texture']}, vertex_colors={texture_info_1['has_vertex_colors']}")
        print(f"[PreviewMeshDual] Mesh 2 visual: kind={texture_info_2['visual_kind']}, texture={texture_info_2['has_texture']}, vertex_colors={texture_info_2['has_vertex_colors']}")

        # Check if meshes are point clouds (need VTP, STL doesn't support point clouds)
        mesh_1_is_pc = is_point_cloud(mesh_1)
        mesh_2_is_pc = is_point_cloud(mesh_2)

        # Generate unique ID for this preview
        preview_id = uuid.uuid4().hex[:8]

        if layout == "side_by_side" or layout == "slider":
            # Export meshes separately based on mode
            if mode == "texture":
                # Texture mode: export as GLB
                filename_1, filepath_1 = self._export_mesh(mesh_1, f"preview_dual_1_{preview_id}", use_vtp=False, use_glb=True)
                filename_2, filepath_2 = self._export_mesh(mesh_2, f"preview_dual_2_{preview_id}", use_vtp=False, use_glb=True)
            else:
                # Fields mode: use VTP for fields OR point clouds
                filename_1, filepath_1 = self._export_mesh(mesh_1, f"preview_dual_1_{preview_id}", use_vtp=(mesh_1_has_fields or mesh_1_is_pc), use_glb=False)
                filename_2, filepath_2 = self._export_mesh(mesh_2, f"preview_dual_2_{preview_id}", use_vtp=(mesh_2_has_fields or mesh_2_is_pc), use_glb=False)

            # Compute bounds from vertices (works for both meshes and point clouds)
            bounds_1 = np.array([mesh_1.vertices.min(axis=0), mesh_1.vertices.max(axis=0)])
            bounds_2 = np.array([mesh_2.vertices.min(axis=0), mesh_2.vertices.max(axis=0)])
            extents_1 = bounds_1[1] - bounds_1[0]
            extents_2 = bounds_2[1] - bounds_2[0]

            # Build UI data for side-by-side mode
            ui_data = {
                "layout": [layout],
                "mode": [mode],
                "mesh_1_file": [filename_1],
                "mesh_2_file": [filename_2],
                "vertex_count_1": [len(mesh_1.vertices)],
                "vertex_count_2": [len(mesh_2.vertices)],
                "face_count_1": [get_face_count(mesh_1)],
                "face_count_2": [get_face_count(mesh_2)],
                "bounds_min_1": [bounds_1[0].tolist()],
                "bounds_max_1": [bounds_1[1].tolist()],
                "bounds_min_2": [bounds_2[0].tolist()],
                "bounds_max_2": [bounds_2[1].tolist()],
                "extents_1": [extents_1.tolist()],
                "extents_2": [extents_2.tolist()],
                "is_watertight_1": [bool(mesh_1.is_watertight) if not is_point_cloud(mesh_1) else False],
                "is_watertight_2": [bool(mesh_2.is_watertight) if not is_point_cloud(mesh_2) else False],
                "opacity_1": [float(opacity_1)],
                "opacity_2": [float(opacity_2)],
            }

            # Add mode-specific metadata
            if mode == "texture":
                # Texture mode metadata
                ui_data.update({
                    "has_texture_1": [texture_info_1['has_texture']],
                    "has_texture_2": [texture_info_2['has_texture']],
                    "visual_kind_1": [texture_info_1['visual_kind'] if texture_info_1['visual_kind'] else "none"],
                    "visual_kind_2": [texture_info_2['visual_kind'] if texture_info_2['visual_kind'] else "none"],
                    "has_vertex_colors_1": [texture_info_1['has_vertex_colors']],
                    "has_vertex_colors_2": [texture_info_2['has_vertex_colors']],
                    "has_material_1": [texture_info_1['has_material']],
                    "has_material_2": [texture_info_2['has_material']],
                })
            else:
                # Fields mode metadata
                ui_data.update({
                    "field_names_1": [field_names_1],
                    "field_names_2": [field_names_2],
                    "common_fields": [common_fields],
                })


        else:  # overlay
            # Combine meshes with color coding
            if mode == "texture":
                # Texture mode: export combined mesh as GLB
                filename, filepath = self._export_combined_mesh(
                    mesh_1, mesh_2, preview_id, opacity_1, opacity_2,
                    mesh_1_has_fields, mesh_2_has_fields, use_glb=True
                )
            else:
                # Fields mode: export combined mesh as VTP
                filename, filepath = self._export_combined_mesh(
                    mesh_1, mesh_2, preview_id, opacity_1, opacity_2,
                    mesh_1_has_fields, mesh_2_has_fields, use_glb=False
                )

            # Compute bounds from vertices (works for both meshes and point clouds)
            bounds_1 = np.array([mesh_1.vertices.min(axis=0), mesh_1.vertices.max(axis=0)])
            bounds_2 = np.array([mesh_2.vertices.min(axis=0), mesh_2.vertices.max(axis=0)])
            combined_bounds_min = np.minimum(bounds_1[0], bounds_2[0])
            combined_bounds_max = np.maximum(bounds_1[1], bounds_2[1])
            combined_extents = combined_bounds_max - combined_bounds_min

            # Build UI data for overlay mode
            ui_data = {
                "layout": [layout],
                "mode": [mode],
                "mesh_file": [filename],
                "vertex_count_1": [len(mesh_1.vertices)],
                "vertex_count_2": [len(mesh_2.vertices)],
                "face_count_1": [get_face_count(mesh_1)],
                "face_count_2": [get_face_count(mesh_2)],
                "bounds_min": [combined_bounds_min.tolist()],
                "bounds_max": [combined_bounds_max.tolist()],
                "extents": [combined_extents.tolist()],
                "opacity_1": [float(opacity_1)],
                "opacity_2": [float(opacity_2)],
                "is_watertight_1": [bool(mesh_1.is_watertight) if not is_point_cloud(mesh_1) else False],
                "is_watertight_2": [bool(mesh_2.is_watertight) if not is_point_cloud(mesh_2) else False],
            }

            # Add mode-specific metadata
            if mode == "texture":
                # Texture mode metadata
                ui_data.update({
                    "has_texture_1": [texture_info_1['has_texture']],
                    "has_texture_2": [texture_info_2['has_texture']],
                    "visual_kind_1": [texture_info_1['visual_kind'] if texture_info_1['visual_kind'] else "none"],
                    "visual_kind_2": [texture_info_2['visual_kind'] if texture_info_2['visual_kind'] else "none"],
                    "has_vertex_colors_1": [texture_info_1['has_vertex_colors']],
                    "has_vertex_colors_2": [texture_info_2['has_vertex_colors']],
                    "has_material_1": [texture_info_1['has_material']],
                    "has_material_2": [texture_info_2['has_material']],
                })
            else:
                # Fields mode metadata - include mesh_id which is added during export for overlay
                overlay_fields_1 = field_names_1 + ["mesh_id"] if layout == "overlay" else field_names_1
                overlay_fields_2 = field_names_2 + ["mesh_id"] if layout == "overlay" else field_names_2
                overlay_common = common_fields + ["mesh_id"] if layout == "overlay" else common_fields
                ui_data.update({
                    "field_names_1": [overlay_fields_1],
                    "field_names_2": [overlay_fields_2],
                    "common_fields": [overlay_common],
                })

        print(f"[PreviewMeshDual] Preview ready")
        return {"ui": ui_data}

    def _export_mesh(self, mesh, base_filename, use_vtp, use_glb):
        """Export a single mesh to appropriate format."""
        if use_glb:
            filename = f"{base_filename}.glb"
        elif use_vtp:
            filename = f"{base_filename}.vtp"
        else:
            filename = f"{base_filename}.stl"

        if COMFYUI_OUTPUT_FOLDER:
            filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
        else:
            filepath = os.path.join(tempfile.gettempdir(), filename)

        try:
            if use_glb:
                mesh.export(filepath, file_type='glb', include_normals=True)
                print(f"[PreviewMeshDual] Exported GLB: {filepath}")
            elif use_vtp:
                export_mesh_with_scalars_vtp(mesh, filepath)
                print(f"[PreviewMeshDual] Exported VTP with fields: {filepath}")
            else:
                mesh.export(filepath, file_type='stl')
                print(f"[PreviewMeshDual] Exported STL: {filepath}")
        except Exception as e:
            print(f"[PreviewMeshDual] Export failed: {e}, trying fallback")
            # Fallback to OBJ
            filename = filename.replace('.vtp', '.obj').replace('.stl', '.obj').replace('.glb', '.obj')
            filepath = filepath.replace('.vtp', '.obj').replace('.stl', '.obj').replace('.glb', '.obj')
            mesh.export(filepath, file_type='obj')
            print(f"[PreviewMeshDual] Exported OBJ fallback: {filepath}")

        return filename, filepath

    def _export_combined_mesh(self, mesh_1, mesh_2, preview_id, opacity_1, opacity_2,
                              mesh_1_has_fields, mesh_2_has_fields, use_glb):
        """Export combined mesh for overlay mode as VTP or GLB.

        Automatically applies red color to mesh_1 and blue color to mesh_2
        for easy visual distinction in overlay mode.
        """

        # Combine meshes with automatic color distinction
        try:
            # Create copies to avoid modifying original meshes
            mesh_1_copy = mesh_1.copy()
            mesh_2_copy = mesh_2.copy()

            # Add mesh_id field for distinction in fields mode (0 = mesh_1, 1 = mesh_2)
            mesh_1_copy.vertex_attributes['mesh_id'] = np.zeros(len(mesh_1_copy.vertices), dtype=np.float32)
            mesh_2_copy.vertex_attributes['mesh_id'] = np.ones(len(mesh_2_copy.vertices), dtype=np.float32)

            # Apply red vertex colors to mesh_1 (RGBA: 255, 77, 77, 255) for texture mode
            red_colors = np.full((len(mesh_1_copy.vertices), 4), [255, 77, 77, 255], dtype=np.uint8)
            mesh_1_copy.visual.vertex_colors = red_colors

            # Apply blue vertex colors to mesh_2 (RGBA: 77, 77, 255, 255) for texture mode
            blue_colors = np.full((len(mesh_2_copy.vertices), 4), [77, 77, 255, 255], dtype=np.uint8)
            mesh_2_copy.visual.vertex_colors = blue_colors

            print(f"[PreviewMeshDual] Added mesh_id field and red/blue colors for overlay")

            combined = trimesh_module.util.concatenate([mesh_1_copy, mesh_2_copy])

            if use_glb:
                filename = f"preview_dual_overlay_{preview_id}.glb"
            else:
                filename = f"preview_dual_overlay_{preview_id}.vtp"

            if COMFYUI_OUTPUT_FOLDER:
                filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
            else:
                filepath = os.path.join(tempfile.gettempdir(), filename)

            if use_glb:
                combined.export(filepath, file_type='glb', include_normals=True)
                print(f"[PreviewMeshDual] Exported combined GLB: {filepath}")
            else:
                export_mesh_with_scalars_vtp(combined, filepath)
                print(f"[PreviewMeshDual] Exported combined VTP: {filepath}")

            print(f"[PreviewMeshDual] Combined {get_geometry_type(combined)}: {len(combined.vertices)} vertices, {get_face_count(combined)} faces")
            return filename, filepath
        except Exception as e:
            print(f"[PreviewMeshDual] Failed to export combined mesh: {e}")
            raise


NODE_CLASS_MAPPINGS = {
    "GeomPackPreviewMeshDual": PreviewMeshDualNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackPreviewMeshDual": "Preview Mesh Dual",
}
