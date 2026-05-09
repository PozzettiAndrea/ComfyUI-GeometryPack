# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Unified Smooth Mesh Node - Single frontend with backend selector.

Uses ComfyUI's node expansion (GraphBuilder) to dispatch to hidden
backend-specific nodes.
"""

import logging
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class SmoothMeshNode(io.ComfyNode):
    """
    Smooth Mesh - Unified smoothing with backend selection.

    Dispatches to hidden backend nodes via node expansion.
    """

    BACKEND_MAP = {
        "taubin":            "GeomPackSmooth_Taubin",
        "laplacian":         "GeomPackSmooth_Laplacian",
        "hc_laplacian":      "GeomPackSmooth_HCLaplacian",
        "trimesh_laplacian": "GeomPackSmooth_TrimeshLaplacian",
        "trimesh_taubin":    "GeomPackSmooth_TrimeshTaubin",
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackSmoothMesh",
            display_name="Smooth Mesh",
            category="geompack/smoothing",
            enable_expand=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.DynamicCombo.Input("backend", tooltip=(
                        "Smoothing algorithm. "
                        "taubin=shrinkage-free (recommended), "
                        "laplacian=fast but shrinks, "
                        "hc_laplacian=low shrinkage, "
                        "trimesh_*=lightweight alternatives"
                    ), options=[
                    io.DynamicCombo.Option("taubin", [
                        io.Int.Input("iterations", default=5, min=1, max=200, step=1, tooltip="Number of smoothing passes. More = smoother but slower."),
                        io.Float.Input("lambda_", default=0.5, min=0.01, max=1.0, step=0.01, tooltip="Smoothing strength per step. Higher = more aggressive smoothing per iteration."),
                        io.Float.Input("mu", default=-0.53, min=-1.0, max=-0.01, step=0.01, tooltip="Inflation factor (negative). Counteracts shrinkage from lambda. Must satisfy |mu| > lambda for stability. Typical: -0.53 for lambda=0.5."),
                    ]),
                    io.DynamicCombo.Option("laplacian", [
                        io.Int.Input("iterations", default=5, min=1, max=200, step=1, tooltip="Number of smoothing passes. More = smoother but slower."),
                        io.Combo.Input("cotangent_weight", options=["true", "false"], default="true", tooltip="Use cotangent weights instead of uniform weights. Cotangent weights respect mesh geometry better but may be unstable on degenerate meshes."),
                    ]),
                    io.DynamicCombo.Option("hc_laplacian", []),
                    io.DynamicCombo.Option("trimesh_laplacian", [
                        io.Int.Input("iterations", default=5, min=1, max=200, step=1, tooltip="Number of smoothing passes. More = smoother but slower."),
                        io.Float.Input("lambda_", default=0.5, min=0.01, max=1.0, step=0.01, tooltip="Smoothing strength per step. Higher = more aggressive smoothing per iteration."),
                    ]),
                    io.DynamicCombo.Option("trimesh_taubin", [
                        io.Int.Input("iterations", default=5, min=1, max=200, step=1, tooltip="Number of smoothing passes. More = smoother but slower."),
                        io.Float.Input("lambda_", default=0.5, min=0.01, max=1.0, step=0.01, tooltip="Smoothing strength per step. Higher = more aggressive smoothing per iteration."),
                        io.Float.Input("mu", default=-0.53, min=-1.0, max=-0.01, step=0.01, tooltip="Inflation factor (negative). Counteracts shrinkage from lambda. Must satisfy |mu| > lambda for stability. Typical: -0.53 for lambda=0.5."),
                    ]),
                ]),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="smoothed_mesh"),
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

        log.info("Smooth dispatch: %s -> %s", selected, node_id)

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
    "GeomPackSmoothMesh": SmoothMeshNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackSmoothMesh": "Smooth Mesh",
}
