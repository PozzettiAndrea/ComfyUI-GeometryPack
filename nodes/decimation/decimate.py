# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Unified Decimate Mesh Node - Single frontend with backend selector.

Uses ComfyUI's node expansion (GraphBuilder) to dispatch to hidden
backend-specific nodes, each running in its own isolation env.
"""

import logging
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class DecimateMeshNode(io.ComfyNode):
    """
    Decimate Mesh - Unified decimation with backend selection.

    Dispatches to hidden backend nodes via node expansion.
    """

    BACKEND_MAP = {
        "quadric_edge_collapse": "GeomPackDecimate_QuadricEdgeCollapse",
        "fast_simplification":   "GeomPackDecimate_FastSimplification",
        "vertex_clustering":     "GeomPackDecimate_VertexClustering",
        # cgal_edge_collapse disabled: CGAL Python bindings lack Surface_mesh_simplification module
        "decimate_pro":          "GeomPackDecimate_DecimatePro",
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackDecimateMesh",
            display_name="Decimate Mesh",
            category="geompack/decimation",
            enable_expand=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.DynamicCombo.Input("backend", tooltip=(
                        "Decimation algorithm. "
                        "quadric_edge_collapse=best quality, "
                        "fast_simplification=fastest, "
                        "vertex_clustering=aggressive, "
                        "decimate_pro=topology-preserving"
                    ), options=[
                    io.DynamicCombo.Option("quadric_edge_collapse", [
                        io.Int.Input("target_face_count", default=5000, min=4, max=10000000, step=100, tooltip="Target number of output faces."),
                        io.Float.Input("quality_threshold", default=0.3, min=0.0, max=1.0, step=0.05, tooltip="Quality threshold for edge collapse. Higher = more conservative, better triangle quality."),
                        io.Combo.Input("preserve_boundary", options=["true", "false"], default="true", tooltip="Preserve mesh boundary edges during decimation."),
                        io.Combo.Input("preserve_normal", options=["true", "false"], default="true", tooltip="Prevent face normal flips during decimation."),
                        io.Combo.Input("preserve_topology", options=["true", "false"], default="true", tooltip="Preserve mesh topology (genus) during decimation."),
                        io.Combo.Input("planar_quadric", options=["true", "false"], default="false", tooltip="Add penalty for non-planar faces. Helps preserve flat regions."),
                    ]),
                    io.DynamicCombo.Option("fast_simplification", [
                        io.Float.Input("target_reduction", default=0.5, min=0.01, max=0.99, step=0.01, tooltip="Fraction of faces to REMOVE. 0.5 = reduce to ~50%% of original faces, 0.9 = reduce to ~10%% of original."),
                        io.Int.Input("aggressiveness", default=7, min=1, max=15, step=1, tooltip="How aggressively to simplify. Higher = faster but lower quality. Default 7."),
                    ]),
                    io.DynamicCombo.Option("vertex_clustering", [
                        io.Float.Input("cluster_threshold", default=1.0, min=0.1, max=10.0, step=0.1, tooltip="Clustering cell size as percentage of bounding box diagonal. Larger = more aggressive reduction."),
                    ]),

                    io.DynamicCombo.Option("decimate_pro", [
                        io.Float.Input("target_reduction", default=0.5, min=0.01, max=0.99, step=0.01, tooltip="Fraction of faces to REMOVE. 0.5 = reduce to ~50%% of original faces, 0.9 = reduce to ~10%% of original."),
                        io.Float.Input("feature_angle", default=15.0, min=0.0, max=180.0, step=1.0, tooltip="Feature angle threshold (degrees). Edges with dihedral angle above this are preserved."),
                        io.Combo.Input("preserve_topology", options=["true", "false"], default="true", tooltip="Preserve mesh topology (genus) during decimation."),
                    ]),
                ]),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="decimated_mesh"),
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

        log.info("Decimate dispatch: %s -> %s", selected, node_id)

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
    "GeomPackDecimateMesh": DecimateMeshNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackDecimateMesh": "Decimate Mesh",
}
