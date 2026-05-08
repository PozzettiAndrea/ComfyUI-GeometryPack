# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Unified Reconstruct Surface Node - Single frontend with method selector.

Uses ComfyUI's node expansion (GraphBuilder) to dispatch to hidden
backend-specific nodes, each running in its own isolation env.
"""

import logging
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class ReconstructSurfaceNode(io.ComfyNode):
    """
    Reconstruct Surface - Convert point cloud to mesh.

    Dispatches to hidden backend nodes via node expansion.
    """

    BACKEND_MAP = {
        "poisson": "GeomPackReconstruct_Poisson",
        "ball_pivoting": "GeomPackReconstruct_BallPivoting",
        "alpha_shape": "GeomPackReconstruct_AlphaShape",
        "convex_hull": "GeomPackReconstruct_ConvexHull",
    }

    # Remap frontend param names to backend param names where they differ
    PARAM_REMAP = {}

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackReconstructSurface",
            display_name="Reconstruct Surface",
            category="geompack/reconstruction",
            enable_expand=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("points"),
                io.DynamicCombo.Input("backend", tooltip=(
                    "Reconstruction algorithm. "
                    "poisson=smooth watertight, "
                    "ball_pivoting=preserves detail, "
                    "alpha_shape=controllable detail, "
                    "convex_hull=fast rough"
                ), options=[
                    io.DynamicCombo.Option("poisson", [
                        io.Int.Input("poisson_depth", default=8, min=1, max=12, step=1, tooltip="Octree depth for the Poisson solver. Higher values capture finer detail but use exponentially more memory and time. 6=coarse, 8=balanced, 10+=high detail."),
                        io.Float.Input("poisson_scale", default=1.1, min=1.0, max=2.0, step=0.1, tooltip="Scale factor for the reconstruction grid relative to the bounding box. Values >1.0 add padding to avoid boundary artifacts. 1.1 is usually sufficient."),
                        io.Combo.Input("estimate_normals", options=["true", "false"], default="true", tooltip="Re-estimate point normals using k-nearest neighbors. Poisson reconstruction requires oriented normals — enable this if the input has no normals or unreliable normals."),
                        io.Float.Input("normal_radius", default=0.1, min=0.001, max=10.0, step=0.01, tooltip="Search radius for normal estimation via k-nearest neighbors. Should be 2-3x the average point spacing. Only used when estimate_normals is true."),
                    ]),
                    io.DynamicCombo.Option("ball_pivoting", [
                        io.Float.Input("ball_radius", default=0.0, min=0.0, max=100.0, step=0.01, tooltip="Ball radius in absolute units (same as mesh coordinates). Smaller = finer detail but more holes, larger = smoother but loses detail. 0 = auto (PyMeshLab estimates from point spacing)."),
                        io.Combo.Input("estimate_normals", options=["true", "false"], default="true", tooltip="Re-estimate point normals using k-nearest neighbors. Enable if the input has no normals or unreliable normals."),
                        io.Float.Input("normal_radius", default=0.1, min=0.001, max=10.0, step=0.01, tooltip="Search radius for normal estimation via k-nearest neighbors. Should be 2-3x the average point spacing. Only used when estimate_normals is true."),
                    ]),
                    io.DynamicCombo.Option("alpha_shape", [
                        io.Float.Input("alpha", default=0.0, min=0.0, max=100.0, step=0.01, tooltip="Radius threshold controlling which Delaunay tetrahedra are kept. Only tetrahedra with longest edge < 2*alpha are included. Larger = coarser shape with more fill, smaller = tighter fit with more holes. 0 = auto (10%% of bounding box diagonal)."),
                    ]),
                    io.DynamicCombo.Option("convex_hull", []),
                ]),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="reconstructed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, points, backend):
        from comfy_execution.graph_utils import GraphBuilder

        if cls.SCHEMA is None:
            cls.GET_SCHEMA()

        selected = backend["backend"]
        node_id = cls.BACKEND_MAP[selected]

        log.info("Reconstruct dispatch: %s -> %s", selected, node_id)

        # Build kwargs, applying param remaps for cross-env backends
        remap = cls.PARAM_REMAP.get(selected, {})
        kwargs = {remap.get("points", "points"): points}
        for k, v in backend.items():
            if k == "backend":
                continue
            kwargs[remap.get(k, k)] = v

        graph = GraphBuilder()
        backend_node = graph.node(node_id, **kwargs)

        return {
            "result": (backend_node.out(0), backend_node.out(1)),
            "expand": graph.finalize(),
        }


NODE_CLASS_MAPPINGS = {
    "GeomPackReconstructSurface": ReconstructSurfaceNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackReconstructSurface": "Reconstruct Surface",
}
