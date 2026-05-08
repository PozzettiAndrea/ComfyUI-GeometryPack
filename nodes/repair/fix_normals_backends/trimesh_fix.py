# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Trimesh fix normals backend node."""

import logging
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class FixNormalsTrimeshNode(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackFixNormals_Trimesh",
            display_name="Fix Normals Trimesh (backend)",
            category="geompack/repair",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="fixed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh):
        fixed_mesh = trimesh.copy()

        was_consistent = fixed_mesh.is_winding_consistent

        fixed_mesh.fix_normals()

        is_consistent = fixed_mesh.is_winding_consistent

        info = (
            f"Normal Orientation Fix:\n"
            f"\n"
            f"Method: trimesh\n"
            f"Before: {'Consistent' if was_consistent else 'Inconsistent'}\n"
            f"After:  {'Consistent' if is_consistent else 'Inconsistent'}\n"
            f"\n"
            f"Vertices: {len(fixed_mesh.vertices):,}\n"
            f"Faces: {len(fixed_mesh.faces):,}\n"
            f"\n"
            f"{'[OK] Normals are now consistently oriented!' if is_consistent else '[WARN] Some inconsistencies may remain (check mesh topology)'}"
        )

        log.info("Normal orientation: %s -> %s", was_consistent, is_consistent)

        return io.NodeOutput(fixed_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackFixNormals_Trimesh": FixNormalsTrimeshNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackFixNormals_Trimesh": "Fix Normals Trimesh (backend)"}
