# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""TEASAR skeletonization backend node."""

import logging
import numpy as np
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class SkeletonTeasarNode(io.ComfyNode):
    """TEASAR skeletonization backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackSkeleton_Teasar",
            display_name="Skeleton Teasar (backend)",
            category="geompack/skeleton",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Boolean.Input("fix_mesh", default=True, tooltip="Fix mesh issues before skeletonization"),
                io.Boolean.Input("normalize", default=False, tooltip="Normalize skeleton to [-1, 1] range"),
                io.Float.Input("inv_dist", default=10.0, min=1.0, max=100.0, tooltip="Invalidation distance (lower = more detail)"),
                io.Float.Input("min_length", default=0.0, min=0.0, max=100.0, tooltip="Minimum branch length to keep"),
            ],
            outputs=[
                io.Custom("SKELETON").Output(display_name="skeleton"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, fix_mesh=True, normalize=False, inv_dist=10.0, min_length=0.0):
        import skeletor as sk

        log.info("Backend: teasar")

        if fix_mesh:
            mesh = sk.pre.fix_mesh(trimesh, remove_disconnected=5, inplace=False)
        else:
            mesh = trimesh

        skel = sk.skeletonize.by_teasar(
            mesh,
            inv_dist=inv_dist,
            min_length=min_length if min_length > 0 else None,
        )

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


NODE_CLASS_MAPPINGS = {"GeomPackSkeleton_Teasar": SkeletonTeasarNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackSkeleton_Teasar": "Skeleton Teasar (backend)"}
