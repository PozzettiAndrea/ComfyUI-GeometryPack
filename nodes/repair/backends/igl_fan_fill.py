# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""libigl fan-triangulation fill holes backend node."""

import logging
import numpy as np
import trimesh
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class FillHolesIglFanNode(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackFillHoles_IglFan",
            display_name="Fill Holes IGL Fan (backend)",
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
        import igl

        initial_faces = len(mesh.faces)
        was_watertight = mesh.is_watertight

        V = np.asarray(mesh.vertices, dtype=np.float64)
        F = np.asarray(mesh.faces, dtype=np.int32)

        loop = igl.boundary_loop(F)

        total_filled = 0
        if isinstance(loop, np.ndarray) and loop.size > 0 and loop.ndim == 1 and len(loop) >= 3:
            # Fan triangulation from first vertex of the loop
            new_faces = []
            center = loop[0]
            for i in range(1, len(loop) - 1):
                new_faces.append([center, loop[i], loop[i + 1]])

            if new_faces:
                F = np.vstack([F, np.array(new_faces, dtype=np.int32)])
                total_filled = 1
                log.info("igl_fan: filled boundary loop with %d faces", len(new_faces))
        else:
            log.warning("igl_fan: no boundary loop found")

        filled = trimesh.Trimesh(vertices=V, faces=F, process=False)

        added = len(filled.faces) - initial_faces
        is_watertight = filled.is_watertight

        log.info("igl_fan: +%d faces, watertight: %s -> %s", added, was_watertight, is_watertight)

        info = (f"Method: igl_fan\n"
                f"Holes filled: {total_filled}\n"
                f"Faces added: {added}\n"
                f"Watertight: {was_watertight} -> {is_watertight}")

        return io.NodeOutput(filled, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackFillHoles_IglFan": FillHolesIglFanNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackFillHoles_IglFan": "Fill Holes IGL Fan (backend)"}
