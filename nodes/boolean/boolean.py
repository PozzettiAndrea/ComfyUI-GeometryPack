# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Unified Boolean Node - Single frontend with backend selector.

Uses ComfyUI's node expansion (GraphBuilder) to dispatch to hidden
backend-specific nodes, each running in its own isolation env.
"""

import logging
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class BooleanNode(io.ComfyNode):
    """
    Boolean - Unified CSG operations with backend selection.

    Dispatches to hidden backend nodes via node expansion.
    Backends span multiple isolation envs (main, blender).
    """

    BACKEND_MAP = {
        "libigl_cgal":   "GeomPackBoolean_LibiglCGAL",
        "blender_exact": "GeomPackBoolean_BlenderExact",
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackBoolean",
            display_name="Boolean Operations",
            category="geompack/boolean",
            enable_expand=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("mesh_a"),
                io.Custom("TRIMESH").Input("mesh_b"),
                io.Combo.Input("operation", options=["union", "difference", "intersection"]),
                io.DynamicCombo.Input("backend", tooltip=(
                        "Boolean engine. "
                        "libigl_cgal=robust CGAL-based (best), "
                        "blender_exact=Blender EXACT solver"
                    ), options=[
                    io.DynamicCombo.Option("libigl_cgal", []),
                    io.DynamicCombo.Option("blender_exact", []),
                ]),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="result_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, mesh_a, mesh_b, operation, backend):
        from comfy_execution.graph_utils import GraphBuilder

        if cls.SCHEMA is None:
            cls.GET_SCHEMA()

        selected = backend["backend"]
        node_id = cls.BACKEND_MAP[selected]

        log.info("Boolean dispatch: %s -> %s", selected, node_id)

        kwargs = {
            "mesh_a": mesh_a,
            "mesh_b": mesh_b,
            "operation": operation,
        }
        # Add any backend-specific params (currently none)
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


NODE_CLASS_MAPPINGS = {
    "GeomPackBoolean": BooleanNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackBoolean": "Boolean Operations",
}
