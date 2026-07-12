# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Multi mesh preview with VTK.js, batch-input variant.

Takes a single mesh batch (list) instead of 4 discrete mesh slots, always uses
scalar-field visualization (no texture mode), and lets the user set the grid's
row/column count directly instead of having it auto-derived from mesh count.
"""

import logging
import os
import tempfile
import uuid

import numpy as np

from .mesh_helpers import is_point_cloud, get_face_count, get_geometry_type
from ._vtp_export import export_mesh_with_scalars_vtp
from .preview_mesh_multi import extract_field_names

log = logging.getLogger("geometrypack")

try:
    import folder_paths
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except (ImportError, AttributeError):
    COMFYUI_OUTPUT_FOLDER = None
from comfy_api.latest import io


class PreviewMeshMultiBatchNode(io.ComfyNode):
    """
    Multi mesh preview (batch input) with VTK.js -- displays a mesh batch in a
    user-chosen rows x cols grid, always in scalar-field mode.
    """

    INPUT_IS_LIST = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackPreviewMeshMultiBatch",
            display_name="Preview Mesh Multi (Batch)",
            category="geompack/visualization",
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("meshes"),
                io.Int.Input("rows", default=2, min=1, max=8,
                             tooltip="Number of grid rows. Meshes beyond rows*cols are dropped (logged)."),
                io.Int.Input("cols", default=2, min=1, max=8,
                             tooltip="Number of grid columns. Meshes beyond rows*cols are dropped (logged)."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="meshes", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, meshes, rows, cols):
        rows_val = rows[0] if isinstance(rows, list) else rows
        cols_val = cols[0] if isinstance(cols, list) else cols

        if not meshes or len(meshes) == 0:
            raise ValueError("Empty mesh batch provided")

        capacity = rows_val * cols_val
        if len(meshes) > capacity:
            log.warning("Batch has %d meshes but grid is %dx%d (capacity %d) -- dropping the last %d",
                        len(meshes), rows_val, cols_val, capacity, len(meshes) - capacity)
        selected = meshes[:capacity]

        num_meshes = len(selected)
        log.info("Grid: %dx%d, showing %d/%d meshes", rows_val, cols_val, num_meshes, len(meshes))

        preview_id = uuid.uuid4().hex[:8]

        mesh_files = []
        vertex_counts = []
        face_counts = []
        bounds_list = []
        extents_list = []
        is_watertight_list = []
        avg_edge_lengths = []
        field_names_list = []

        for i, mesh in enumerate(selected):
            log.info("Mesh %d: %s - %d vertices, %d faces", i + 1, get_geometry_type(mesh), len(mesh.vertices), get_face_count(mesh))

            mesh_is_pc = is_point_cloud(mesh)

            filename = f"preview_multi_batch_{i+1}_{preview_id}.vtp"
            filepath = os.path.join(COMFYUI_OUTPUT_FOLDER or tempfile.gettempdir(), filename)

            try:
                export_mesh_with_scalars_vtp(mesh, filepath)
                log.info("Exported VTP: %s", filepath)
            except Exception as e:
                log.error("Export failed: %s, trying OBJ fallback", e)
                filename = f"preview_multi_batch_{i+1}_{preview_id}.obj"
                filepath = os.path.join(COMFYUI_OUTPUT_FOLDER or tempfile.gettempdir(), filename)
                mesh.export(filepath, file_type='obj')

            mesh_files.append(filename)
            vertex_counts.append(len(mesh.vertices))
            face_counts.append(get_face_count(mesh))

            bounds = np.array([mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)])
            extents = bounds[1] - bounds[0]
            bounds_list.append(bounds.tolist())
            extents_list.append(extents.tolist())

            is_watertight_list.append(bool(mesh.is_watertight) if not mesh_is_pc else False)
            avg_edge = None
            if not mesh_is_pc:
                try:
                    if get_face_count(mesh) > 0:
                        avg_edge = float(np.mean(mesh.edges_unique_length))
                except Exception as e:
                    log.info("Mesh %d: could not compute avg edge length: %s", i + 1, e)
            avg_edge_lengths.append(avg_edge)
            field_names_list.append(extract_field_names(mesh))

        ui_data = {
            "mode": ["fields"],
            "num_meshes": [num_meshes],
            "grid_cols": [cols_val],
            "grid_rows": [rows_val],
            "mesh_files": [mesh_files],
            "vertex_counts": [vertex_counts],
            "face_counts": [face_counts],
            "bounds_list": [bounds_list],
            "extents_list": [extents_list],
            "is_watertight_list": [is_watertight_list],
            "avg_edge_lengths": [avg_edge_lengths],
            "field_names_list": [field_names_list],
        }

        log.info("Grid: %dx%d, Preview ready", rows_val, cols_val)
        return io.NodeOutput(selected, ui=ui_data)


NODE_CLASS_MAPPINGS = {
    "GeomPackPreviewMeshMultiBatch": PreviewMeshMultiBatchNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackPreviewMeshMultiBatch": "Preview Mesh Multi (Batch)",
}
