# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Preview batch of meshes with VTK.js scientific visualization viewer with index navigation.

Displays meshes from a batch in an interactive VTK.js viewer with trackball controls.
Includes navigation buttons to cycle through meshes in the batch.
Better for scientific visualization, mesh analysis, and large datasets.

Supports scalar field visualization: automatically detects vertex and face
attributes and exports to VTP format to preserve field data for visualization.

The whole batch is exported on every run so the frontend can switch between
meshes CLIENT-SIDE (arrows / dropdown) without re-queuing the graph -- that is
what stops other preview nodes from reloading when you navigate this one.
"""

import logging

import trimesh as trimesh_module
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


def _export_and_describe(mesh, mode_val, output_folder):
    """Export ONE mesh for the VTK viewer and return (filename, meta).

    Format: GLB in texture mode; VTP for scalar fields / point clouds; STL for
    plain surface meshes; OBJ as a last-resort fallback. `meta` carries the
    per-mesh info the frontend info panel shows for whichever index is on
    screen (so navigation needs no server round-trip).
    """
    has_vertex_attrs = hasattr(mesh, 'vertex_attributes') and len(mesh.vertex_attributes) > 0
    has_face_attrs = hasattr(mesh, 'face_attributes') and len(mesh.face_attributes) > 0
    has_fields = has_vertex_attrs or has_face_attrs
    is_pc = is_point_cloud(mesh)

    has_visual = hasattr(mesh, 'visual') and mesh.visual is not None
    visual_kind = mesh.visual.kind if has_visual else None
    has_texture = bool(has_visual and visual_kind == 'texture' and hasattr(mesh.visual, 'material'))
    has_vertex_colors = bool(has_visual and visual_kind == 'vertex')

    uid = uuid.uuid4().hex[:8]
    if mode_val == "texture":
        filename = f"preview_vtk_batch_{uid}.glb"
    elif has_fields or is_pc:
        # STL cannot carry scalar fields or point clouds -> VTP
        filename = f"preview_vtk_batch_{uid}.vtp"
    else:
        filename = f"preview_vtk_batch_{uid}.stl"

    base = output_folder or tempfile.gettempdir()
    filepath = os.path.join(base, filename)
    try:
        if mode_val == "texture":
            mesh.export(filepath, file_type='glb', include_normals=True)
        elif has_fields or is_pc:
            export_mesh_with_scalars_vtp(mesh, filepath)
        else:
            mesh.export(filepath, file_type='stl')
    except Exception as e:
        log.error("Export failed (%s); falling back to OBJ: %s", filename, e)
        filename = filename.rsplit('.', 1)[0] + '.obj'
        filepath = os.path.join(base, filename)
        mesh.export(filepath, file_type='obj')

    bounds = mesh.bounds
    is_watertight = False if is_pc else bool(mesh.is_watertight)

    field_names = []
    if has_vertex_attrs:
        field_names.extend(list(mesh.vertex_attributes.keys()))
    if has_face_attrs:
        field_names.extend([f"face.{k}" for k in mesh.face_attributes.keys()])

    meta = {
        "vertex_count": len(mesh.vertices),
        "face_count": get_face_count(mesh),
        "bounds_min": bounds[0].tolist(),
        "bounds_max": bounds[1].tolist(),
        "extents": mesh.extents.tolist(),
        "is_watertight": is_watertight,
        "field_names": field_names,
        "has_texture": has_texture,
        "has_vertex_colors": has_vertex_colors,
        "visual_kind": visual_kind if visual_kind else "none",
    }
    return filename, meta


class PreviewMeshVTKBatchNode(io.ComfyNode):
    """
    Preview batch of meshes with VTK.js scientific visualization viewer with index navigation.

    Displays meshes from a batch in an interactive VTK.js viewer with trackball controls.
    Includes navigation buttons to cycle through meshes in the batch.
    Better for scientific visualization, mesh analysis, and large datasets.
    """


    INPUT_IS_LIST = True


    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackPreviewMeshVTKBatch",
            display_name="Preview Mesh Batch (VTK)",
            category="geompack/visualization",
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Combo.Input("mode", options=["fields", "texture"], default="fields"),
                io.Int.Input("index", default=0, min=0, max=100),
            ],
            outputs=[
                # Pass the batch through so the preview can sit mid-graph
                # (same pattern as Preview Mesh Multi).
                io.Custom("TRIMESH").Output(display_name="meshes", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, trimesh, mode, index):
        """
        Export the WHOLE batch and prepare for VTK.js preview.

        Every mesh is exported so the frontend can navigate the batch entirely
        client-side (no re-queue -> no reload of other preview nodes). `index`
        only picks which mesh is shown first.

        Args:
            trimesh: List of trimesh_module.Trimesh objects (the batch)
            mode: List with visualization mode - "fields" or "texture"
            index: List with the initial index to display

        Returns:
            io.NodeOutput with per-index UI arrays for the frontend widget
        """
        # Inputs arrive as lists (INPUT_IS_LIST=True); the batch itself is `trimesh`.
        mode_val = mode[0] if isinstance(mode, list) else mode
        index_val = index[0] if isinstance(index, list) else index

        if not trimesh or len(trimesh) == 0:
            raise ValueError("Empty mesh batch provided")

        batch_size = len(trimesh)
        actual_index = max(0, min(index_val, batch_size - 1))
        # viewer_type is global (a function of `mode` only), so switching meshes
        # never needs a different viewer HTML -- pure client-side LOAD_MESH.
        viewer_type = "texture" if mode_val == "texture" else "fields"

        log.info("Batch of %d mesh(es), mode=%s -- exporting all for client-side "
                 "navigation (initial index %d)", batch_size, mode_val, actual_index)

        filenames = []
        metas = []
        names = []  # source mesh names for the frontend dropdown (e.g. "apple.ply")
        for i, mesh in enumerate(trimesh):
            fname, meta = _export_and_describe(mesh, mode_val, COMFYUI_OUTPUT_FOLDER)
            filenames.append(fname)
            metas.append(meta)
            # Loaders stash the source basename in metadata['file_name'] (mesh_io.py);
            # fall back to a positional label if a mesh has no recorded name.
            md = getattr(mesh, "metadata", None) or {}
            names.append(md.get("file_name") or md.get("name") or f"mesh {i + 1}")
            log.debug("[%d/%d] %s (%s): %d verts, %d faces", i + 1, batch_size,
                      names[-1], fname, meta["vertex_count"], meta["face_count"])

        # Per-index arrays. Each value is wrapped in a one-element list (ComfyUI's
        # UI-message convention: the frontend reads message.<key>[0]).
        ui_data = {
            "mesh_files": [filenames],
            "mesh_names": [names],
            "viewer_type": [viewer_type],
            "mode": [mode_val],
            "batch_size": [batch_size],
            "current_index": [actual_index],
            "vertex_counts": [[m["vertex_count"] for m in metas]],
            "face_counts": [[m["face_count"] for m in metas]],
            "bounds_mins": [[m["bounds_min"] for m in metas]],
            "bounds_maxs": [[m["bounds_max"] for m in metas]],
            "extents_all": [[m["extents"] for m in metas]],
            "is_watertights": [[m["is_watertight"] for m in metas]],
            "field_names_all": [[m["field_names"] for m in metas]],
            "has_textures": [[m["has_texture"] for m in metas]],
            "has_vertex_colors_all": [[m["has_vertex_colors"] for m in metas]],
            "visual_kinds": [[m["visual_kind"] for m in metas]],
        }
        return io.NodeOutput(list(trimesh), ui=ui_data)


NODE_CLASS_MAPPINGS = {
    "GeomPackPreviewMeshVTKBatch": PreviewMeshVTKBatchNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackPreviewMeshVTKBatch": "Preview Mesh Batch (VTK)",
}
