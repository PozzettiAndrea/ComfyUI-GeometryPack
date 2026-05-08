# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Trimesh Laplacian smoothing backend node."""

import logging
import numpy as np
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _trimesh_laplacian_smooth(mesh, iterations, lamb):
    """Laplacian smoothing via trimesh (uniform weights)."""
    result = mesh.copy()

    from trimesh.smoothing import filter_laplacian
    filter_laplacian(result, lamb=lamb, iterations=iterations)

    return result


class SmoothTrimeshLaplacianNode(io.ComfyNode):
    """Trimesh Laplacian smoothing backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackSmooth_TrimeshLaplacian",
            display_name="Smooth Trimesh Laplacian (backend)",
            category="geompack/smoothing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("iterations", default=5, min=1, max=200, step=1, tooltip="Number of smoothing passes. More = smoother but slower."),
                io.Float.Input("lambda_", default=0.5, min=0.01, max=1.0, step=0.01, tooltip="Smoothing strength per step. Higher = more aggressive smoothing per iteration."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="smoothed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, iterations=5, lambda_=0.5):
        log.info("Backend: trimesh_laplacian")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: iterations=%d, lambda=%.3f", iterations, lambda_)

        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        smoothed = _trimesh_laplacian_smooth(trimesh, iterations, lambda_)

        if hasattr(trimesh, "metadata") and trimesh.metadata:
            smoothed.metadata = trimesh.metadata.copy()
        smoothed.metadata["smoothing"] = {
            "algorithm": "trimesh_laplacian",
            "iterations": iterations,
            "original_vertices": initial_vertices,
            "original_faces": initial_faces,
        }

        disp = np.linalg.norm(
            np.asarray(smoothed.vertices) - np.asarray(trimesh.vertices), axis=1
        )
        avg_disp = float(np.mean(disp))
        max_disp = float(np.max(disp))

        log.info("Output: %d vertices, %d faces", len(smoothed.vertices), len(smoothed.faces))
        log.info("Avg vertex displacement: %.6f, max: %.6f", avg_disp, max_disp)

        info = f"""Smooth Mesh Results (trimesh_laplacian):

Iterations: {iterations}
Lambda: {lambda_}

Vertices: {initial_vertices:,} (unchanged)
Faces: {initial_faces:,} (unchanged)

Displacement:
  Average: {avg_disp:.6f}
  Maximum: {max_disp:.6f}
"""
        return io.NodeOutput(smoothed, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackSmooth_TrimeshLaplacian": SmoothTrimeshLaplacianNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackSmooth_TrimeshLaplacian": "Smooth Trimesh Laplacian (backend)"}
