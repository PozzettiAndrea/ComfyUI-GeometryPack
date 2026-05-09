# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""PyMeshFix fill holes backend node."""

import logging
import numpy as np
import trimesh
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class FillHolesPyMeshFixNode(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackFillHoles_PyMeshFix",
            display_name="Fill Holes PyMeshFix (backend)",
            category="geompack/repair",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("mesh"),
                io.Int.Input("max_edges", default=0, min=0, max=100000,
                             tooltip="Max boundary edges for a hole to be filled. 0 = fill all holes.",
                             optional=True),
                io.Combo.Input("refine", options=["true", "false"], default="true",
                               tooltip="Refine filled regions for better triangle quality.",
                               optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="filled_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, mesh, max_edges=0, refine="true"):
        from pymeshfix import MeshFix

        initial_faces = len(mesh.faces)
        was_watertight = mesh.is_watertight

        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)

        mfix = MeshFix(verts, faces)
        n_boundaries = mfix.n_boundaries
        log.info("PyMeshFix: %d boundary loops detected", n_boundaries)

        num_filled = mfix.fill_holes(n_edges=max_edges, refine=(refine == "true"))

        filled = trimesh.Trimesh(
            vertices=mfix.points,
            faces=mfix.faces,
            process=False,
        )

        added = len(filled.faces) - initial_faces
        is_watertight = filled.is_watertight

        log.info("PyMeshFix: filled %d holes, +%d faces, watertight: %s -> %s",
                 num_filled, added, was_watertight, is_watertight)

        info = (f"Method: pymeshfix\n"
                f"Max edges: {max_edges} (0=all)\n"
                f"Refine: {refine}\n"
                f"Boundaries found: {n_boundaries}\n"
                f"Holes filled: {num_filled}\n"
                f"Faces added: {added}\n"
                f"Watertight: {was_watertight} -> {is_watertight}")

        return io.NodeOutput(filled, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackFillHoles_PyMeshFix": FillHolesPyMeshFixNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackFillHoles_PyMeshFix": "Fill Holes PyMeshFix (backend)"}
