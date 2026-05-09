# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Wavefront skeletonization backend node."""

import logging
import numpy as np
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class SkeletonWavefrontNode(io.ComfyNode):
    """Wavefront skeletonization backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackSkeleton_Wavefront",
            display_name="Skeleton Wavefront (backend)",
            category="geompack/skeleton",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Boolean.Input("fix_mesh", default=True, tooltip="Fix mesh issues before skeletonization"),
                io.Boolean.Input("normalize", default=False, tooltip="Normalize skeleton to [-1, 1] range"),
                io.Int.Input("waves", default=1, min=1, max=20, tooltip="Number of waves"),
                io.Float.Input("step_size", default=1.0, min=0.1, max=20.0, tooltip="Step size (higher = coarser)"),
            ],
            outputs=[
                io.Custom("SKELETON").Output(display_name="skeleton"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, fix_mesh=True, normalize=False, waves=1, step_size=1.0):
        import skeletor as sk

        log.info("Backend: wavefront")

        if fix_mesh:
            mesh = sk.pre.fix_mesh(trimesh, remove_disconnected=5, inplace=False)
        else:
            mesh = trimesh

        skel = sk.skeletonize.by_wavefront(mesh, waves=waves, step_size=step_size)

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


NODE_CLASS_MAPPINGS = {"GeomPackSkeleton_Wavefront": SkeletonWavefrontNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackSkeleton_Wavefront": "Skeleton Wavefront (backend)"}
