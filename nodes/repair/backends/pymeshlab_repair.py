# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""PyMeshLab mesh-repair backend node (degenerate / sliver / duplicate cleanup)."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class MeshRepairPyMeshLabNode(io.ComfyNode):
    """PyMeshLab cleanup backend: removes degenerate/sliver/duplicate elements."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackMeshRepair_PyMeshLab",
            display_name="Mesh Repair PyMeshLab (backend)",
            category="geompack/repair",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("mesh"),
                io.Combo.Input("remove_duplicate_vertices", options=["true", "false"], default="true", tooltip="Merge coincident (duplicate) vertices first, so the cleanups below see correct connectivity."),
                io.Combo.Input("remove_null_faces", options=["true", "false"], default="true", tooltip="Remove null / zero-area (degenerate) faces -- the core sliver remover."),
                io.Combo.Input("remove_folded_faces", options=["true", "false"], default="true", tooltip="Remove folded faces (triangles that flip back over a neighbor -- thin slivers / overlaps)."),
                io.Combo.Input("remove_duplicate_faces", options=["true", "false"], default="true", tooltip="Remove exact-duplicate faces (same 3 vertices)."),
                io.Combo.Input("remove_t_vertices", options=["true", "false"], default="false", tooltip="Remove T-vertices (a vertex lying on another face's edge -> cracks). Off by default; can change topology."),
                io.Combo.Input("repair_non_manifold_edges", options=["true", "false"], default="true", tooltip="Repair non-manifold edges (edges shared by >2 faces)."),
                io.Float.Input("remove_small_components_pct", default=0.0, min=0.0, max=100.0, step=0.5, display_mode="number", tooltip="Drop floating components whose bounding-box diagonal is below this PERCENT of the whole-mesh diagonal (0 = off). Great for scan/recon crumbs."),
                io.Combo.Input("remove_unreferenced_vertices", options=["true", "false"], default="true", tooltip="Drop orphan vertices not used by any face (run last)."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="repaired_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, mesh, remove_duplicate_vertices="true", remove_null_faces="true",
                remove_folded_faces="true", remove_duplicate_faces="true",
                remove_t_vertices="false", repair_non_manifold_edges="true",
                remove_small_components_pct=0.0, remove_unreferenced_vertices="true"):
        import pymeshlab as ml

        v0, f0 = len(mesh.vertices), len(mesh.faces)
        was_watertight = mesh.is_watertight

        ms = ml.MeshSet()
        ms.add_mesh(ml.Mesh(
            vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
            face_matrix=np.asarray(mesh.faces, dtype=np.int32),
        ))

        applied = []

        def step(enabled, label, fn):
            if not enabled:
                return
            try:
                fn()
                applied.append(label)
            except Exception as e:
                log.warning("Mesh Repair: %s failed: %s", label, e)

        # Order matters: merge verts -> drop bad faces -> repair topology ->
        # drop small components -> drop orphan verts.
        step(remove_duplicate_vertices == "true", "duplicate_vertices", ms.meshing_remove_duplicate_vertices)
        step(remove_null_faces == "true", "null_faces", ms.meshing_remove_null_faces)
        step(remove_folded_faces == "true", "folded_faces", ms.meshing_remove_folded_faces)
        step(remove_duplicate_faces == "true", "duplicate_faces", ms.meshing_remove_duplicate_faces)
        step(remove_t_vertices == "true", "t_vertices", ms.meshing_remove_t_vertices)
        step(repair_non_manifold_edges == "true", "non_manifold_edges", ms.meshing_repair_non_manifold_edges)
        if remove_small_components_pct and remove_small_components_pct > 0:
            step(True, f"small_components(<{remove_small_components_pct:g}%)",
                 lambda: ms.meshing_remove_connected_component_by_diameter(
                     mincomponentdiag=ml.PercentageValue(float(remove_small_components_pct))))
        step(remove_unreferenced_vertices == "true", "unreferenced_vertices", ms.meshing_remove_unreferenced_vertices)

        m = ms.current_mesh()
        repaired = trimesh_module.Trimesh(
            vertices=m.vertex_matrix(), faces=m.face_matrix(), process=False)
        if hasattr(mesh, "metadata") and mesh.metadata:
            repaired.metadata = mesh.metadata.copy()

        v1, f1 = len(repaired.vertices), len(repaired.faces)
        log.info("Mesh Repair (pymeshlab): applied [%s] | %dv/%df -> %dv/%df",
                 ", ".join(applied), v0, f0, v1, f1)

        info = (
            f"Mesh Repair (pymeshlab):\n"
            f"\n"
            f"Applied: {', '.join(applied) if applied else '(none enabled)'}\n"
            f"Vertices: {v0:,} -> {v1:,} ({v1 - v0:+,})\n"
            f"Faces: {f0:,} -> {f1:,} ({f1 - f0:+,})\n"
            f"Watertight: {was_watertight} -> {repaired.is_watertight}"
        )
        return io.NodeOutput(repaired, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackMeshRepair_PyMeshLab": MeshRepairPyMeshLabNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackMeshRepair_PyMeshLab": "Mesh Repair PyMeshLab (backend)"}
