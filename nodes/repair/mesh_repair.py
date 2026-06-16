# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Unified Mesh Repair node - single frontend with backend selector.

Cleans up degenerate / sliver / duplicate / non-manifold elements. Dispatches to
hidden backend nodes via node expansion (GraphBuilder), each in its own isolation
env. Designed to grow more backends (pymeshfix, trimesh, cgal, ...) over time.
"""

import logging
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class MeshRepairNode(io.ComfyNode):
    """Mesh Repair - unified degenerate/sliver cleanup with backend selection."""

    BACKEND_MAP = {
        "pymeshlab": "GeomPackMeshRepair_PyMeshLab",
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackMeshRepair",
            display_name="Mesh Repair",
            category="geompack/repair",
            description=(
                "Clean up bad geometry: degenerate / sliver / folded / duplicate faces, "
                "duplicate vertices, T-vertices, non-manifold edges, and small floating "
                "components. Pick a backend; each exposes its own toggles.\n"
                "\n"
                "pymeshlab: per-operation cleanup -- the real sliver/degenerate remover "
                "(MeshLab's meshing_remove_* filters). More backends (pymeshfix, ...) to come."
            ),
            enable_expand=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("mesh"),
                io.DynamicCombo.Input("backend", tooltip="Repair algorithm and backend", options=[
                    io.DynamicCombo.Option("pymeshlab", [
                        io.Combo.Input("remove_duplicate_vertices", options=["true", "false"], default="true", tooltip="Merge coincident (duplicate) vertices first."),
                        io.Combo.Input("remove_null_faces", options=["true", "false"], default="true", tooltip="Remove null / zero-area (degenerate) faces -- the core sliver remover."),
                        io.Combo.Input("remove_folded_faces", options=["true", "false"], default="true", tooltip="Remove folded faces (slivers that flip back over a neighbor)."),
                        io.Combo.Input("remove_duplicate_faces", options=["true", "false"], default="true", tooltip="Remove exact-duplicate faces."),
                        io.Combo.Input("remove_t_vertices", options=["true", "false"], default="false", tooltip="Remove T-vertices (vertex on another face's edge). Off by default; changes topology."),
                        io.Combo.Input("repair_non_manifold_edges", options=["true", "false"], default="true", tooltip="Repair non-manifold edges (shared by >2 faces)."),
                        io.Float.Input("remove_small_components_pct", default=0.0, min=0.0, max=100.0, step=0.5, display_mode="number", tooltip="Drop floating components below this PERCENT of the mesh's bbox diagonal (0 = off). Removes scan/recon crumbs."),
                        io.Combo.Input("remove_unreferenced_vertices", options=["true", "false"], default="true", tooltip="Drop orphan vertices not used by any face (last)."),
                    ]),
                ]),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="repaired_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, mesh, backend):
        from comfy_execution.graph_utils import GraphBuilder

        if cls.SCHEMA is None:
            cls.GET_SCHEMA()

        selected = backend["backend"]
        node_id = cls.BACKEND_MAP[selected]

        log.info("Mesh Repair dispatch: %s -> %s", selected, node_id)

        kwargs = {"mesh": mesh}
        for k, v in backend.items():
            if k == "backend":
                continue
            kwargs[k] = v

        graph = GraphBuilder()
        backend_node = graph.node(node_id, **kwargs)

        return {
            "result": (backend_node.out(0), backend_node.out(1)),
            "expand": graph.finalize(),
        }


NODE_CLASS_MAPPINGS = {"GeomPackMeshRepair": MeshRepairNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackMeshRepair": "Mesh Repair"}
