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

import logging

import trimesh as trimesh_module
import numpy as np
import os
import tempfile
import uuid

from .mesh_helpers import is_point_cloud, get_face_count, get_geometry_type

from ._vtp_export import export_mesh_with_scalars_vtp

log = logging.getLogger("geometrypack")

try:
    import folder_paths
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except (ImportError, AttributeError):
    COMFYUI_OUTPUT_FOLDER = None
from comfy_api.latest import io


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


def with_normal_fields(mesh):
    """Return a copy of the mesh with 'vertex_normals' (point) and 'face_normals'
    (cell) attached as 3-component fields, so the viewer can colour by them and
    pick the X/Y/Z channel. No-op for point clouds / on failure. Idempotent."""
    try:
        if is_point_cloud(mesh) or not hasattr(mesh, "faces"):
            return mesh
        m = mesh.copy()
        va = m.vertex_attributes if m.vertex_attributes is not None else {}
        fa = m.face_attributes if m.face_attributes is not None else {}
        if "vertex_normals" not in va:
            vn = np.asarray(m.vertex_normals)
            if vn.shape == (len(m.vertices), 3):
                m.vertex_attributes["vertex_normals"] = vn.astype(np.float32)
        if "face_normals" not in fa:
            fn = np.asarray(m.face_normals)
            if fn.shape == (len(m.faces), 3):
                m.face_attributes["face_normals"] = fn.astype(np.float32)
        return m
    except Exception as e:
        log.warning("normal-field attach skipped: %s", e)
        return mesh


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


class PreviewMeshDualNode(io.ComfyNode):
    """
    Unified dual mesh preview with VTK.js - supports both side-by-side and overlay layouts.

    Combines two meshes for comparison with full field visualization support.
    Choose between synchronized side-by-side viewports or single overlaid viewport.
    """


    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackPreviewMeshDual",
            display_name="Preview Mesh Dual",
            category="geompack/visualization",
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("mesh_1"),
                io.Custom("TRIMESH").Input("mesh_2"),
                # layout and opacity are NOT node inputs: layout is a pure viewer
                # choice (both the separate pair AND the combined overlay file are
                # exported every run, so the frontend switches without re-running)
                # and opacity is applied client-side in the viewer.
                io.Combo.Input("mode", options=["fields", "texture"], default="fields", optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="meshes", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, mesh_1, mesh_2, mode="fields"):
        """
        Preview two meshes; the layout (side-by-side / overlay / slider) is
        chosen client-side, so BOTH the separate pair and the combined overlay
        file are exported every run.

        Args:
            mesh_1: First trimesh object
            mesh_2: Second trimesh object
            mode: "fields" (scientific visualization) or "texture" (textured rendering)

        Returns:
            io.NodeOutput with UI data for the frontend widget
        """
        log.info("Mode: %s", mode)
        log.info("Mesh 1: %s - %d vertices, %d faces", get_geometry_type(mesh_1), len(mesh_1.vertices), get_face_count(mesh_1))
        log.info("Mesh 2: %s - %d vertices, %d faces", get_geometry_type(mesh_2), len(mesh_2.vertices), get_face_count(mesh_2))

        # Expose face + vertex normals as selectable fields ONLY when the mesh
        # already carries fields (so it was going to export as VTP anyway). Forcing
        # it for plain meshes flipped STL->VTP and broke loading, so don't.
        if mode != "texture":
            if has_fields(mesh_1):
                mesh_1 = with_normal_fields(mesh_1)
            if has_fields(mesh_2):
                mesh_2 = with_normal_fields(mesh_2)

        # Check for field data
        mesh_1_has_fields = has_fields(mesh_1)
        mesh_2_has_fields = has_fields(mesh_2)
        field_names_1 = extract_field_names(mesh_1)
        field_names_2 = extract_field_names(mesh_2)
        common_fields = list(set(field_names_1) & set(field_names_2))

        log.info("Mesh 1 fields: %s", field_names_1)
        log.info("Mesh 2 fields: %s", field_names_2)
        log.info("Common fields: %s", common_fields)

        # Check for texture/visual data
        texture_info_1 = get_texture_info(mesh_1)
        texture_info_2 = get_texture_info(mesh_2)

        log.info("Mesh 1 visual: kind=%s, texture=%s, vertex_colors=%s", texture_info_1['visual_kind'], texture_info_1['has_texture'], texture_info_1['has_vertex_colors'])
        log.info("Mesh 2 visual: kind=%s, texture=%s, vertex_colors=%s", texture_info_2['visual_kind'], texture_info_2['has_texture'], texture_info_2['has_vertex_colors'])

        # Check if meshes are point clouds (need VTP, STL doesn't support point clouds)
        mesh_1_is_pc = is_point_cloud(mesh_1)
        mesh_2_is_pc = is_point_cloud(mesh_2)

        # Generate unique ID for this preview
        preview_id = uuid.uuid4().hex[:8]

        # Export BOTH view sets every run so the frontend can switch layouts
        # client-side: the separate pair (side_by_side / slider) and the
        # combined overlay file.
        use_glb = mode == "texture"
        filename_1, _ = cls._export_mesh(
            mesh_1, f"preview_dual_1_{preview_id}",
            use_vtp=(not use_glb and (mesh_1_has_fields or mesh_1_is_pc)), use_glb=use_glb)
        filename_2, _ = cls._export_mesh(
            mesh_2, f"preview_dual_2_{preview_id}",
            use_vtp=(not use_glb and (mesh_2_has_fields or mesh_2_is_pc)), use_glb=use_glb)
        filename_combined, _ = cls._export_combined_mesh(
            mesh_1, mesh_2, preview_id, mesh_1_has_fields, mesh_2_has_fields,
            use_glb=use_glb)

        # Bounds from vertices (works for both meshes and point clouds)
        bounds_1 = np.array([mesh_1.vertices.min(axis=0), mesh_1.vertices.max(axis=0)])
        bounds_2 = np.array([mesh_2.vertices.min(axis=0), mesh_2.vertices.max(axis=0)])
        extents_1 = bounds_1[1] - bounds_1[0]
        extents_2 = bounds_2[1] - bounds_2[0]

        ui_data = {
            "mode": [mode],
            "mesh_1_file": [filename_1],
            "mesh_2_file": [filename_2],
            "mesh_file": [filename_combined],   # overlay view
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
        }

        if mode == "texture":
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
            ui_data.update({
                "field_names_1": [field_names_1],
                "field_names_2": [field_names_2],
                "common_fields": [common_fields],
                # the combined overlay export injects a mesh_id field
                "common_fields_overlay": [common_fields + ["mesh_id"]],
            })

        log.info("Preview ready")
        return io.NodeOutput([mesh_1, mesh_2], ui=ui_data)

    @staticmethod
    def _export_mesh(mesh, base_filename, use_vtp, use_glb):
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
                log.info("Exported GLB: %s", filepath)
            elif use_vtp:
                export_mesh_with_scalars_vtp(mesh, filepath)
                log.info("Exported VTP with fields: %s", filepath)
            else:
                mesh.export(filepath, file_type='stl')
                log.info("Exported STL: %s", filepath)
        except Exception as e:
            log.error("Export failed: %s, trying fallback", e)
            # Fallback to OBJ
            filename = filename.replace('.vtp', '.obj').replace('.stl', '.obj').replace('.glb', '.obj')
            filepath = filepath.replace('.vtp', '.obj').replace('.stl', '.obj').replace('.glb', '.obj')
            mesh.export(filepath, file_type='obj')
            log.info("Exported OBJ fallback: %s", filepath)

        return filename, filepath

    @staticmethod
    def _export_combined_mesh(mesh_1, mesh_2, preview_id,
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

            log.info("Added mesh_id field and red/blue colors for overlay")

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
                log.info("Exported combined GLB: %s", filepath)
            else:
                export_mesh_with_scalars_vtp(combined, filepath)
                log.info("Exported combined VTP: %s", filepath)

            log.info("Combined %s: %d vertices, %d faces", get_geometry_type(combined), len(combined.vertices), get_face_count(combined))
            return filename, filepath
        except Exception as e:
            log.error("Failed to export combined mesh: %s", e)
            raise


NODE_CLASS_MAPPINGS = {
    "GeomPackPreviewMeshDual": PreviewMeshDualNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackPreviewMeshDual": "Preview Mesh Dual",
}
