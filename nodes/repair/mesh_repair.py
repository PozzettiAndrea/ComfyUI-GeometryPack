# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Unified Mesh Repair node - single frontend with backend selector.

Cleans up degenerate / sliver / duplicate / non-manifold elements. Dispatches to
hidden backend nodes via node expansion (GraphBuilder), each in its own isolation
env. Self-intersection fixing lives in its own dispatcher (GeomPackFixSelfIntersections,
nodes/repair_cgal/fix_self_intersections.py) -- a distinct topology problem, not
degeneracy/duplication.
"""

import logging
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class MeshRepairNode(io.ComfyNode):
    """Mesh Repair - unified degenerate/sliver cleanup with backend selection."""

    BACKEND_MAP = {
        "trimesh": "GeomPackMeshRepair_Trimesh",
        "pymeshlab": "GeomPackMeshRepair_PyMeshLab",
        "pymeshfix": "GeomPackMeshRepair_PyMeshFix",
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
                "trimesh: lightweight, no extra deps -- merges duplicate vertices then drops "
                "degenerate/cap-sliver faces.\n"
                "pymeshlab: per-operation cleanup -- the real sliver/degenerate remover "
                "(MeshLab's meshing_remove_* filters).\n"
                "pymeshfix: small-component removal/joining plus PyTMesh's own clean() pass."
            ),
            enable_expand=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("mesh"),
                io.DynamicCombo.Input("backend", tooltip="Repair algorithm and backend", options=[
                    io.DynamicCombo.Option("trimesh", [
                        io.Float.Input("tolerance", default=1e-5, min=1e-8, max=1e-2, step=1e-6,
                                       tooltip="Distance tolerance for merging duplicate vertices (1e-5 recommended for CAD meshes)."),
                        io.Float.Input("min_area", default=1e-10, min=0.0, max=1.0, step=1e-10,
                                       tooltip="Faces with area below this are deleted (0 disables this test)."),
                        io.Float.Input("max_angle_deg", default=180.0, min=90.0, max=180.0, step=0.5,
                                       tooltip="Cap-sliver collapse threshold on largest interior angle. 180 = off."),
                    ]),
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
                    io.DynamicCombo.Option("pymeshfix", [
                        io.Combo.Input("remove_small_components", options=["true", "false"], default="true", tooltip="Remove small isolated mesh fragments before cleaning."),
                        io.Combo.Input("join_components", options=["true", "false"], default="false", tooltip="Attempt to join nearby disconnected components."),
                        io.Combo.Input("clean_mesh", options=["true", "false"], default="true", tooltip="Remove self-intersections and degenerate faces (PyTMesh.clean)."),
                        io.Int.Input("clean_iterations", default=10, min=1, max=100, step=1, tooltip="Max iterations for self-intersection removal."),
                        io.Int.Input("inner_loops", default=3, min=1, max=10, step=1, tooltip="Inner loops per clean iteration."),
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
