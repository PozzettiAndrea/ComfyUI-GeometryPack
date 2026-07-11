# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
PyMeshFix mesh-repair backend node (component cleanup + self-intersection/degenerate cleanup).

Trimmed from GeomPackMeshFix: hole-filling is intentionally NOT exposed here -- that's
GeomPackFillHoles's "pymeshfix" backend's job. This backend focuses on what's not
already covered by another dispatcher: small-component removal/joining and PyTMesh's
own clean() pass (self-intersection + degenerate-face removal).
"""

import logging
import numpy as np
import trimesh
from comfy_api.latest import io

log = logging.getLogger("geompack")


class MeshRepairPyMeshFixNode(io.ComfyNode):
    """PyMeshFix cleanup backend: small-component removal/joining + clean() pass."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackMeshRepair_PyMeshFix",
            display_name="Mesh Repair PyMeshFix (backend)",
            category="geompack/repair",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("mesh"),
                io.Combo.Input("remove_small_components", options=["true", "false"], default="true",
                               tooltip="Remove small isolated mesh fragments before cleaning.", optional=True),
                io.Combo.Input("join_components", options=["true", "false"], default="false",
                               tooltip="Attempt to join nearby disconnected components.", optional=True),
                io.Combo.Input("clean_mesh", options=["true", "false"], default="true",
                               tooltip="Remove self-intersections and degenerate faces (PyTMesh.clean).", optional=True),
                io.Int.Input("clean_iterations", default=10, min=1, max=100, step=1,
                             tooltip="Max iterations for self-intersection removal.", optional=True),
                io.Int.Input("inner_loops", default=3, min=1, max=10, step=1,
                             tooltip="Inner loops per clean iteration.", optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="repaired_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, mesh, remove_small_components="true", join_components="false",
                clean_mesh="true", clean_iterations=10, inner_loops=3):
        remove_small_components = remove_small_components == "true"
        join_components = join_components == "true"
        clean_mesh = clean_mesh == "true"

        initial_vertices, initial_faces = len(mesh.vertices), len(mesh.faces)
        was_watertight = mesh.is_watertight

        v = np.asarray(mesh.vertices, dtype=np.float64)
        f = np.asarray(mesh.faces, dtype=np.int32)

        import pymeshfix
        tin = pymeshfix.PyTMesh()
        tin.load_array(v, f)

        operations = []
        if remove_small_components:
            tin.remove_smallest_components()
            operations.append("Removed small components")
        if join_components:
            tin.join_closest_components()
            operations.append("Joined nearby components")
        if clean_mesh:
            tin.clean(max_iters=clean_iterations, inner_loops=inner_loops)
            operations.append(f"Cleaned (iters={clean_iterations})")

        vclean, fclean = tin.return_arrays()
        result_mesh = trimesh.Trimesh(vertices=vclean, faces=fclean, process=False)
        if hasattr(mesh, "metadata") and mesh.metadata:
            result_mesh.metadata = mesh.metadata.copy()

        final_vertices, final_faces = len(result_mesh.vertices), len(result_mesh.faces)
        is_watertight = result_mesh.is_watertight

        info = (
            f"Mesh Repair (pymeshfix):\n\n"
            f"Applied: {', '.join(operations) if operations else '(none enabled)'}\n"
            f"Vertices: {initial_vertices:,} -> {final_vertices:,} ({final_vertices - initial_vertices:+,})\n"
            f"Faces: {initial_faces:,} -> {final_faces:,} ({final_faces - initial_faces:+,})\n"
            f"Watertight: {was_watertight} -> {is_watertight}"
        )
        log.info("Mesh Repair (pymeshfix): applied [%s] | %dv/%df -> %dv/%df",
                 ", ".join(operations), initial_vertices, initial_faces, final_vertices, final_faces)
        return io.NodeOutput(result_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackMeshRepair_PyMeshFix": MeshRepairPyMeshFixNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackMeshRepair_PyMeshFix": "Mesh Repair PyMeshFix (backend)"}
