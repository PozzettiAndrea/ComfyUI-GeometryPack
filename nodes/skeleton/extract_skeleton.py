# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Extract Skeleton Node - Unified frontend with backend selector.

Uses ComfyUI's node expansion (GraphBuilder) to dispatch to hidden
backend-specific nodes, each running in its own isolation env.
"""

import logging
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class ExtractSkeleton(io.ComfyNode):
    """
    Extract skeleton from 3D mesh using Skeletor library.

    Dispatches to hidden backend nodes via node expansion.
    """

    BACKEND_MAP = {
        "wavefront":        "GeomPackSkeleton_Wavefront",
        "vertex_clusters":  "GeomPackSkeleton_VertexClusters",
        "edge_collapse":    "GeomPackSkeleton_EdgeCollapse",
        "teasar":           "GeomPackSkeleton_Teasar",
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackExtractSkeleton",
            display_name="Extract Skeleton",
            category="geompack/skeleton",
            enable_expand=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Boolean.Input("fix_mesh", default=True, tooltip="Fix mesh issues before skeletonization"),
                io.Boolean.Input("normalize", default=False, tooltip="Normalize skeleton to [-1, 1] range (False preserves original mesh scale)"),
                io.DynamicCombo.Input("backend", tooltip=(
                        "Skeletonization algorithm. "
                        "wavefront=default, "
                        "vertex_clusters=fast clustering, "
                        "edge_collapse=shape-aware, "
                        "teasar=tree-structure"
                    ), options=[
                    io.DynamicCombo.Option("wavefront", [
                        io.Int.Input("waves", default=1, min=1, max=20, tooltip="Number of waves"),
                        io.Float.Input("step_size", default=1.0, min=0.1, max=20.0, tooltip="Step size (higher = coarser)"),
                    ]),
                    io.DynamicCombo.Option("vertex_clusters", [
                        io.Float.Input("sampling_dist", default=1.0, min=0.1, max=50.0, tooltip="Max distance for clustering"),
                        io.Combo.Input("cluster_pos", options=["median", "center"], default="median", tooltip="Cluster position method"),
                    ]),
                    io.DynamicCombo.Option("edge_collapse", [
                        io.Float.Input("shape_weight", default=1.0, min=0.0, max=10.0, tooltip="Shape preservation weight"),
                        io.Float.Input("sample_weight", default=0.1, min=0.0, max=10.0, tooltip="Sampling quality weight"),
                    ]),
                    io.DynamicCombo.Option("teasar", [
                        io.Float.Input("inv_dist", default=10.0, min=1.0, max=100.0, tooltip="Invalidation distance (lower = more detail)"),
                        io.Float.Input("min_length", default=0.0, min=0.0, max=100.0, tooltip="Minimum branch length to keep"),
                    ]),
                ]),
            ],
            outputs=[
                io.Custom("SKELETON").Output(display_name="skeleton"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, fix_mesh, normalize, backend):
        from comfy_execution.graph_utils import GraphBuilder

        if cls.SCHEMA is None:
            cls.GET_SCHEMA()

        selected = backend["backend"]
        node_id = cls.BACKEND_MAP[selected]

        log.info("Extract Skeleton dispatch: %s -> %s", selected, node_id)

        kwargs = {
            "trimesh": trimesh,
            "fix_mesh": fix_mesh,
            "normalize": normalize,
        }
        for k, v in backend.items():
            if k == "backend":
                continue
            kwargs[k] = v

        graph = GraphBuilder()
        backend_node = graph.node(node_id, **kwargs)

        return {
            "result": (backend_node.out(0),),
            "expand": graph.finalize(),
        }


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackExtractSkeleton": ExtractSkeleton,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackExtractSkeleton": "Extract Skeleton",
}
