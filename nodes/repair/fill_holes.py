# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Unified Fill Holes Node - Single frontend with backend selector.

Uses ComfyUI's node expansion (GraphBuilder) to dispatch to hidden
backend-specific nodes, each running in its own isolation env.
"""

import logging
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class FillHolesNode(io.ComfyNode):
    """
    Fill Holes - Unified hole filling with backend selection.

    Dispatches to hidden backend nodes via node expansion.
    """

    BACKEND_MAP = {
        "trimesh":    "GeomPackFillHoles_Trimesh",
        "pymeshlab":  "GeomPackFillHoles_PyMeshLab",
        "pymeshfix":  "GeomPackFillHoles_PyMeshFix",
        "igl_fan":    "GeomPackFillHoles_IglFan",
        "cgal":       "GeomPackFillHoles_CGAL",
        "gpu_cumesh": "GeomPackFillHoles_GPU",
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackFillHoles",
            display_name="Fill Holes",
            category="geompack/repair",
            enable_expand=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("mesh"),
                io.DynamicCombo.Input("backend", tooltip="Hole filling algorithm and backend", options=[
                    io.DynamicCombo.Option("trimesh", []),
                    io.DynamicCombo.Option("pymeshlab", [
                        io.Int.Input("maxholesize", default=0, min=0, max=100000,
                                     tooltip="Max number of boundary edges composing a hole. Only holes with this many edges or fewer are closed. 0 = close all."),
                    ]),
                    io.DynamicCombo.Option("pymeshfix", [
                        io.Int.Input("max_edges", default=0, min=0, max=100000,
                                     tooltip="Max boundary edges for a hole to be filled. 0 = fill all holes."),
                        io.Combo.Input("refine", options=["true", "false"], default="true",
                                       tooltip="Refine filled regions for better triangle quality."),
                    ]),
                    io.DynamicCombo.Option("igl_fan", []),
                    io.DynamicCombo.Option("cgal", [
                        io.Combo.Input("quality", options=["triangulate", "refine", "fair"],
                                       default="refine",
                                       tooltip="triangulate: fast, basic fill. refine: better triangle quality. fair: smooth fill to match surrounding surface."),
                    ]),
                    io.DynamicCombo.Option("gpu_cumesh", [
                        io.Float.Input("perimeter", default=0.03, min=0.001, max=100.0, step=0.001,
                                       tooltip="Maximum hole perimeter to fill. Relative to mesh scale."),
                    ]),
                ]),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="filled_mesh"),
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

        log.info("Fill Holes dispatch: %s -> %s", selected, node_id)

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


NODE_CLASS_MAPPINGS = {
    "GeomPackFillHoles": FillHolesNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackFillHoles": "Fill Holes",
}
