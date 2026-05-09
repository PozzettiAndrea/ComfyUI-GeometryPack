# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""PyMeshLab fill holes backend node."""

import logging
import trimesh
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class FillHolesPyMeshLabNode(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackFillHoles_PyMeshLab",
            display_name="Fill Holes PyMeshLab (backend)",
            category="geompack/repair",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("mesh"),
                io.Int.Input("maxholesize", default=0, min=0, max=100000,
                             tooltip="Max number of boundary edges composing a hole. Only holes with this many edges or fewer are closed. 0 = close all.",
                             optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="filled_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, mesh, maxholesize=0):
        import pymeshlab

        initial_faces = len(mesh.faces)
        was_watertight = mesh.is_watertight

        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(
            vertex_matrix=mesh.vertices,
            face_matrix=mesh.faces,
        ))

        # maxholesize=0 means close all
        kwargs = {}
        if maxholesize > 0:
            kwargs["maxholesize"] = maxholesize
        else:
            kwargs["maxholesize"] = 2147483647

        ms.meshing_close_holes(**kwargs)

        m = ms.current_mesh()
        filled = trimesh.Trimesh(
            vertices=m.vertex_matrix(),
            faces=m.face_matrix(),
            process=False,
        )

        added = len(filled.faces) - initial_faces
        is_watertight = filled.is_watertight

        log.info("PyMeshLab: maxholesize=%s, +%d faces, watertight: %s -> %s",
                 maxholesize, added, was_watertight, is_watertight)

        info = (f"Method: pymeshlab\n"
                f"Max hole size (edges): {maxholesize} (0=all)\n"
                f"Faces added: {added}\n"
                f"Watertight: {was_watertight} -> {is_watertight}")

        return io.NodeOutput(filled, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackFillHoles_PyMeshLab": FillHolesPyMeshLabNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackFillHoles_PyMeshLab": "Fill Holes PyMeshLab (backend)"}
