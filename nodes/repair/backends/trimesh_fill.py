# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Trimesh fill holes backend node."""

import logging
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class FillHolesTrimeshNode(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackFillHoles_Trimesh",
            display_name="Fill Holes Trimesh (backend)",
            category="geompack/repair",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("mesh"),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="filled_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, mesh):
        initial_faces = len(mesh.faces)
        was_watertight = mesh.is_watertight

        filled = mesh.copy()
        filled.fill_holes()

        added = len(filled.faces) - initial_faces
        is_watertight = filled.is_watertight

        log.info("Trimesh: +%d faces, watertight: %s -> %s", added, was_watertight, is_watertight)

        info = (f"Method: trimesh\n"
                f"Faces added: {added}\n"
                f"Watertight: {was_watertight} -> {is_watertight}")

        return io.NodeOutput(filled, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackFillHoles_Trimesh": FillHolesTrimeshNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackFillHoles_Trimesh": "Fill Holes Trimesh (backend)"}
