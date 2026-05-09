# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Vertex clusters skeletonization backend node."""

import logging
import numpy as np
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class SkeletonVertexClustersNode(io.ComfyNode):
    """Vertex clusters skeletonization backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackSkeleton_VertexClusters",
            display_name="Skeleton Vertex Clusters (backend)",
            category="geompack/skeleton",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Boolean.Input("fix_mesh", default=True, tooltip="Fix mesh issues before skeletonization"),
                io.Boolean.Input("normalize", default=False, tooltip="Normalize skeleton to [-1, 1] range"),
                io.Float.Input("sampling_dist", default=1.0, min=0.1, max=50.0, tooltip="Max distance for clustering"),
                io.Combo.Input("cluster_pos", options=["median", "center"], default="median", tooltip="Cluster position method"),
            ],
            outputs=[
                io.Custom("SKELETON").Output(display_name="skeleton"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, fix_mesh=True, normalize=False, sampling_dist=1.0, cluster_pos="median"):
        import skeletor as sk

        log.info("Backend: vertex_clusters")

        if fix_mesh:
            mesh = sk.pre.fix_mesh(trimesh, remove_disconnected=5, inplace=False)
        else:
            mesh = trimesh

        skel = sk.skeletonize.by_vertex_clusters(mesh, sampling_dist=sampling_dist, cluster_pos=cluster_pos)

        vertices = np.array(skel.vertices)
        edges = np.array(skel.edges)

        skel_min = vertices.min(axis=0)
        skel_max = vertices.max(axis=0)
        original_scale = float((skel_max - skel_min).max() / 2)
        original_center = ((skel_min + skel_max) / 2).tolist()

        if normalize:
            from ._helpers import normalize_skeleton
            vertices = normalize_skeleton(vertices)

        skeleton = {
            "vertices": vertices,
            "edges": edges,
            "scale": original_scale,
            "center": original_center,
            "normalized": normalize,
        }

        return io.NodeOutput(skeleton)


NODE_CLASS_MAPPINGS = {"GeomPackSkeleton_VertexClusters": SkeletonVertexClustersNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackSkeleton_VertexClusters": "Skeleton Vertex Clusters (backend)"}
