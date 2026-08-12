# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Preview Mesh Slicer - interactive VTK.js clip/slice preview.

Top: 3D view of the mesh with a live clipping plane. Bottom: an orthographic
pick view (toggle XY / YZ / XZ); click two points to define a cut line, which is
swept along the view axis into a clip plane. "Flip clip" slices the other side.
A "3d view:" row of +X/-X/+Y/-Y/+Z/-Z buttons sets an axis-aligned plane through
the mesh center directly, without needing to pick 2 points.

The Python node exports the mesh to a .vtp for the interactive viewer; the actual
clip plane (origin + normal) is picked entirely in the browser, then mirrored back
here via a hidden `plane_json` widget (see javascript/js/preview_mesh_slicer.js) so this
node can reproduce the same cut on the real mesh data and return it as `sliced_mesh`
-- matching whatever is currently shown in the top 3D view. Since the plane only
exists once the browser has posted it back, the very first run (or right after
loading a new mesh) has no plane yet and `sliced_mesh` is the mesh unchanged --
pick a cut in the viewer, then queue again to get the actual slice.
"""

import json
import logging
import os
import uuid

import numpy as np
import trimesh
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
                io.String.Input("plane_json", default="", optional=True,
                                 tooltip="Internal: clip plane picked in the viewer, mirrored "
                                         "back by the frontend. Not meant to be edited by hand."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="sliced_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, mesh, plane_json=""):
        import folder_paths
        from ._vtp_export import export_mesh_with_scalars_vtp

        tmp = folder_paths.get_temp_directory()
        os.makedirs(tmp, exist_ok=True)
        filename = f"gp_slicer_{uuid.uuid4().hex[:8]}.vtp"
        export_mesh_with_scalars_vtp(mesh, os.path.join(tmp, filename))

        nv = len(getattr(mesh, "vertices", []))
        nf = len(getattr(mesh, "faces", []))
        log.info("[PreviewMeshSlicer] exported %s (%d verts, %d faces)", filename, nv, nf)

        sliced_mesh = mesh
        info = (f"{nv:,} verts / {nf:,} faces -- no clip plane set yet. Pick a cut in the "
                f"viewer (2 points, or a +X/-X/+Y/-Y/+Z/-Z button), then queue again.")

        plane = None
        if plane_json:
            try:
                plane = json.loads(plane_json)
            except (ValueError, TypeError) as e:
                log.warning("[PreviewMeshSlicer] bad plane_json: %s", e)

        if plane and plane.get("hasPlane") and "origin" in plane and "normal" in plane:
            # Matches the VTK view exactly: no capping (VTK's clipping plane doesn't cap the
            # cross-section either), keep the side the normal points TOWARD (trimesh's own
            # "positive normal side" convention for slice_faces_plane == VTK's kept side).
            from trimesh.intersections import slice_faces_plane

            origin = np.asarray(plane["origin"], dtype=np.float64)
            normal = np.asarray(plane["normal"], dtype=np.float64)
            V, F, _ = slice_faces_plane(
                vertices=np.asarray(mesh.vertices, dtype=np.float64).copy(),
                faces=np.asarray(mesh.faces, dtype=np.int64).copy(),
                plane_normal=normal,
                plane_origin=origin,
            )
            sliced_mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
            if hasattr(mesh, "metadata") and mesh.metadata:
                sliced_mesh.metadata = mesh.metadata.copy()

            sv, sf = len(sliced_mesh.vertices), len(sliced_mesh.faces)
            log.info("[PreviewMeshSlicer] sliced origin=%s normal=%s: %dv/%df -> %dv/%df",
                     origin.tolist(), normal.tolist(), nv, nf, sv, sf)
            info = (f"Sliced at origin={np.round(origin, 4).tolist()}, "
                    f"normal={np.round(normal, 4).tolist()}\n"
                    f"{nv:,} -> {sv:,} verts, {nf:,} -> {sf:,} faces")

        return io.NodeOutput(sliced_mesh, info, ui={
            "mesh_file": [filename],
            "summary": [f"{nv:,} verts / {nf:,} faces — slice interactively in the viewer"],
        })


NODE_CLASS_MAPPINGS = {"GeomPackPreviewMeshSlicer": PreviewMeshSlicer}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackPreviewMeshSlicer": "Preview Mesh Slicer"}
