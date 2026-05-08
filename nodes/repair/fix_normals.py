# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Unified Fix Normals Node - Single frontend with backend selector.

Uses ComfyUI's node expansion (GraphBuilder) to dispatch to hidden
backend-specific nodes, each running in its own isolation env.
"""

import logging
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class FixNormalsNode(io.ComfyNode):
    """
    Fix Normals - Unified normal fixing with backend selection.

    Dispatches to hidden backend nodes via node expansion.
    """

    BACKEND_MAP = {
        "trimesh":         "GeomPackFixNormals_Trimesh",
        "igl_bfs":         "GeomPackFixNormals_IglBfs",
        "igl_winding":     "GeomPackFixNormals_IglWinding",
        "igl_raycast":     "GeomPackFixNormals_IglRaycast",
        "igl_signed_dist": "GeomPackFixNormals_IglSignedDist",
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackFixNormals",
            display_name="Fix Normals",
            category="geompack/repair",
            enable_expand=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.DynamicCombo.Input("backend", tooltip="Normal fixing algorithm and backend", options=[
                    io.DynamicCombo.Option("trimesh", []),
                    io.DynamicCombo.Option("igl_bfs", []),
                    io.DynamicCombo.Option("igl_winding", []),
                    io.DynamicCombo.Option("igl_raycast", []),
                    io.DynamicCombo.Option("igl_signed_dist", []),
                ]),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="fixed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, backend):
        from comfy_execution.graph_utils import GraphBuilder

        if cls.SCHEMA is None:
            cls.GET_SCHEMA()

        selected = backend["backend"]
        node_id = cls.BACKEND_MAP[selected]

        log.info("Fix Normals dispatch: %s -> %s", selected, node_id)

        kwargs = {"trimesh": trimesh}
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
    "GeomPackFixNormals": FixNormalsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackFixNormals": "Fix Normals",
}
