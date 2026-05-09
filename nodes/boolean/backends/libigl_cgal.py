# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""libigl+CGAL boolean operations backend node."""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class BooleanLibiglCGALNode(io.ComfyNode):
    """libigl+CGAL boolean operations backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackBoolean_LibiglCGAL",
            display_name="Boolean libigl CGAL (backend)",
            category="geompack/boolean",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("mesh_a"),
                io.Custom("TRIMESH").Input("mesh_b"),
                io.Combo.Input("operation", options=["union", "difference", "intersection"]),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="result_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, mesh_a, mesh_b, operation="union"):
        log.info("Backend: libigl_cgal")
        log.info("Mesh A: %d vertices, %d faces", len(mesh_a.vertices), len(mesh_a.faces))
        log.info("Mesh B: %d vertices, %d faces", len(mesh_b.vertices), len(mesh_b.faces))
        log.info("Operation: %s", operation)

        import igl.copyleft.cgal as cgal

        VA = np.asarray(mesh_a.vertices, dtype=np.float64)
        FA = np.asarray(mesh_a.faces, dtype=np.int64)
        VB = np.asarray(mesh_b.vertices, dtype=np.float64)
        FB = np.asarray(mesh_b.faces, dtype=np.int64)

        VC, FC, J = cgal.mesh_boolean(VA, FA, VB, FB, operation)

        result = trimesh_module.Trimesh(vertices=VC, faces=FC, process=False)

        result.metadata = mesh_a.metadata.copy()
        result.metadata['boolean'] = {
            'operation': operation,
            'engine': 'libigl_cgal',
            'mesh_a_vertices': len(mesh_a.vertices),
            'mesh_a_faces': len(mesh_a.faces),
            'mesh_b_vertices': len(mesh_b.vertices),
            'mesh_b_faces': len(mesh_b.faces),
            'result_vertices': len(result.vertices),
            'result_faces': len(result.faces)
        }

        info = f"""Boolean Operation Results:

Operation: {operation.upper()}
Engine: libigl + CGAL

Mesh A:
  Vertices: {len(mesh_a.vertices):,}
  Faces: {len(mesh_a.faces):,}

Mesh B:
  Vertices: {len(mesh_b.vertices):,}
  Faces: {len(mesh_b.faces):,}

Result:
  Vertices: {len(result.vertices):,}
  Faces: {len(result.faces):,}

Watertight: {result.is_watertight}
"""

        log.info("Success: %d vertices, %d faces", len(result.vertices), len(result.faces))
        return io.NodeOutput(result, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackBoolean_LibiglCGAL": BooleanLibiglCGALNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackBoolean_LibiglCGAL": "Boolean libigl CGAL (backend)"}
