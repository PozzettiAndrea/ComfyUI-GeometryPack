# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Preview Mesh Slicer - interactive VTK.js clip/slice preview.

Top: 3D view of the mesh with a live clipping plane. Bottom: an orthographic
pick view (toggle XY / YZ / XZ); click two points to define a cut line, which is
swept along the view axis into a clip plane. "Flip clip" slices the other side.

The Python node just exports the mesh to a .vtp; all the clipping is interactive
in the browser (viewer_slicer.html).
"""

import logging
import os
import uuid

from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class PreviewMeshSlicer(io.ComfyNode):
    """Interactive clip/slice preview of a mesh in VTK.js."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackPreviewMeshSlicer",
            display_name="Preview Mesh Slicer",
            category="geompack/visualization",
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("mesh", tooltip="Mesh to slice/clip interactively."),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, mesh):
        import folder_paths
        from ._vtp_export import export_mesh_with_scalars_vtp

        tmp = folder_paths.get_temp_directory()
        os.makedirs(tmp, exist_ok=True)
        filename = f"gp_slicer_{uuid.uuid4().hex[:8]}.vtp"
        export_mesh_with_scalars_vtp(mesh, os.path.join(tmp, filename))

        nv = len(getattr(mesh, "vertices", []))
        nf = len(getattr(mesh, "faces", []))
        log.info("[PreviewMeshSlicer] exported %s (%d verts, %d faces)", filename, nv, nf)
        return io.NodeOutput(ui={
            "mesh_file": [filename],
            "summary": [f"{nv:,} verts / {nf:,} faces — slice interactively in the viewer"],
        })


NODE_CLASS_MAPPINGS = {"GeomPackPreviewMeshSlicer": PreviewMeshSlicer}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackPreviewMeshSlicer": "Preview Mesh Slicer"}
