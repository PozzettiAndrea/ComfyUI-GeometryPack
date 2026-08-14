# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Multi mesh preview with VTK.js - displays any number of meshes in a grid.

Inputs AUTOGROW: a fresh mesh socket appears as you connect each one, and
every socket accepts a single mesh OR a whole batch (INPUT_IS_LIST) -- so a
batch of 15 plus one extra mesh shows all 16. Everything is flattened, in
socket order, into one display list.

Grid dims default to ceil(sqrt(n)) and can be overridden client-side (the
widget's Cols/Rows inputs). Display is capped at 16 viewports: each viewport
is its own WebGL context and browsers hard-cap ~16 live contexts per page --
beyond that the oldest contexts get silently killed.

Supports scalar field visualization with synchronized cameras across viewports.
"""

import math

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


class PreviewMeshMultiNode(io.ComfyNode):
    """
    Multi mesh preview with VTK.js - any number of meshes in a grid.

    Autogrowing mesh sockets; every socket accepts a single mesh OR a batch
    (all flattened, in socket order). Display capped at 16 viewports (WebGL
    context limit). Supports scalar field visualization with synchronized
    cameras.
    """

    INPUT_IS_LIST = True     # a socket fed by a batch receives the WHOLE list
    MAX_VIEWPORTS = 16       # each viewport = one WebGL context; browsers cap ~16/page

    @classmethod
    def define_schema(cls):
        # Autogrow: a mesh_0/mesh_1/... socket list that grows a fresh slot as
        # you connect meshes (same pattern as core's MergeSplat).
        meshes_tmpl = io.Autogrow.TemplatePrefix(
            io.Custom("TRIMESH").Input("mesh"), prefix="mesh_", min=1, max=16)
        return io.Schema(
            node_id="GeomPackPreviewMeshMulti",
            display_name="Preview Mesh Multi",
            category="geompack/visualization",
            is_output_node=True,
            inputs=[
                io.Autogrow.Input("meshes", template=meshes_tmpl),
                io.Combo.Input("mode", options=["fields", "texture"], default="fields", optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="meshes", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, meshes, mode="fields"):
        """
        Preview any number of meshes in a grid layout.

        Args:
            meshes: dict of autogrow socket values (each a mesh or a batch list)
            mode: "fields" (scientific visualization) or "texture" (textured rendering)

        Returns:
            io.NodeOutput with the flattened mesh list + UI data
        """
        # INPUT_IS_LIST wraps every input; unwrap scalars defensively.
        mode = mode[0] if isinstance(mode, list) else mode

        # Flatten the autogrow sockets in order: each value may be a batch
        # (list, via INPUT_IS_LIST), a single mesh, or None (unconnected).
        meshes_flat = []
        for value in meshes.values():
            if value is None:
                continue
            meshes_flat.extend(value if isinstance(value, list) else [value])
        if not meshes_flat:
            raise ValueError("Preview Mesh Multi: connect at least one mesh.")

        total = len(meshes_flat)
        meshes = meshes_flat[:cls.MAX_VIEWPORTS]
        if total > cls.MAX_VIEWPORTS:
            log.warning("Preview Mesh Multi: %d meshes connected; displaying the "
                        "first %d (WebGL context limit). All %d still pass through "
                        "the output.", total, cls.MAX_VIEWPORTS, total)

        num_meshes = len(meshes)
        log.info("Mode: %s, Meshes: %d (of %d connected)", mode, num_meshes, total)

        # Generate unique ID for this preview
        preview_id = uuid.uuid4().hex[:8]

        # Export each mesh and collect metadata
        mesh_files = []
        vertex_counts = []
        face_counts = []
        bounds_list = []
        extents_list = []
        is_watertight_list = []
        avg_edge_lengths = []
        field_names_list = []
        texture_info_list = []

        for i, mesh in enumerate(meshes):
            log.info("Mesh %d: %s - %d vertices, %d faces", i + 1, get_geometry_type(mesh), len(mesh.vertices), get_face_count(mesh))

            # Check for field data and texture info
            mesh_has_fields = has_fields(mesh)
            mesh_is_pc = is_point_cloud(mesh)
            texture_info = get_texture_info(mesh)

            # Export mesh
            if mode == "texture":
                filename = f"preview_multi_{i+1}_{preview_id}.glb"
            else:
                filename = f"preview_multi_{i+1}_{preview_id}.vtp"

            if COMFYUI_OUTPUT_FOLDER:
                filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
            else:
                filepath = os.path.join(tempfile.gettempdir(), filename)

            try:
                if mode == "texture":
                    mesh.export(filepath, file_type='glb', include_normals=True)
                    log.info("Exported GLB: %s", filepath)
                else:
                    export_mesh_with_scalars_vtp(mesh, filepath)
                    log.info("Exported VTP: %s", filepath)
            except Exception as e:
                log.error("Export failed: %s, trying OBJ fallback", e)
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
            # average edge length (mesh resolution); None for point clouds / empty meshes
            avg_edge = None
            if not mesh_is_pc:
                try:
                    if get_face_count(mesh) > 0:
                        avg_edge = float(np.mean(mesh.edges_unique_length))
                except Exception as e:
                    log.info("Mesh %d: could not compute avg edge length: %s", i + 1, e)
            avg_edge_lengths.append(avg_edge)
            field_names_list.append(extract_field_names(mesh))
            texture_info_list.append(texture_info)

        # Auto grid: near-square, wide-first (1->1x1, 2->2x1, 3->3x1, 4->2x2,
        # 5-6->3x2, 7-9->3x3, 10-12->4x3, 13-16->4x4). The widget's Cols/Rows
        # bar inputs override these client-side.
        if num_meshes <= 3:
            grid_cols, grid_rows = num_meshes, 1
        else:
            grid_cols = math.ceil(math.sqrt(num_meshes))
            grid_rows = math.ceil(num_meshes / grid_cols)

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
            "avg_edge_lengths": [avg_edge_lengths],
        }

        # Add mode-specific metadata
        if mode == "texture":
            ui_data["texture_info_list"] = [[t for t in texture_info_list]]
        else:
            ui_data["field_names_list"] = [field_names_list]

        log.info("Grid: %dx%d, Preview ready", grid_cols, grid_rows)
        # Output the FULL flattened list (including any meshes beyond the
        # display cap), so downstream nodes see everything that was connected.
        return io.NodeOutput(meshes_flat, ui=ui_data)


NODE_CLASS_MAPPINGS = {
    "GeomPackPreviewMeshMulti": PreviewMeshMultiNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackPreviewMeshMulti": "Preview Mesh Multi",
}
