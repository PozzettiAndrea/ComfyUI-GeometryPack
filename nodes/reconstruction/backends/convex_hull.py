# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Convex hull surface reconstruction backend node."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class ReconstructConvexHullNode(io.ComfyNode):
    """Simple convex hull reconstruction."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackReconstruct_ConvexHull",
            display_name="Reconstruct Convex Hull (backend)",
            category="geompack/reconstruction",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("points"),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="reconstructed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, points):
        log.info("Backend: convex_hull")
        vertices = np.asarray(points.vertices, dtype=np.float64)

        log.info("Computing convex hull...")

        result = trimesh_module.Trimesh(vertices=vertices).convex_hull

        info = f"""Reconstruct Surface Results (Convex Hull):

Input Points: {len(vertices):,}
Output Vertices: {len(result.vertices):,}
Output Faces: {len(result.faces):,}

Watertight: {result.is_watertight}
Volume: {result.volume:.6f}

Convex hull is fast but loses all concave features.
"""
        # Preserve metadata
        if hasattr(points, 'metadata') and points.metadata:
            result.metadata = points.metadata.copy()
        else:
            result.metadata = {}
        result.metadata['reconstruction'] = {
            'method': 'convex_hull',
            'input_points': len(vertices),
            'output_vertices': len(result.vertices),
            'output_faces': len(result.faces),
        }

        return io.NodeOutput(result, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackReconstruct_ConvexHull": ReconstructConvexHullNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackReconstruct_ConvexHull": "Reconstruct Convex Hull (backend)"}
