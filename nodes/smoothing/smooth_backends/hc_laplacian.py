# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""PyMeshLab HC Laplacian smoothing backend node."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _pymeshlab_hc_laplacian_smooth(mesh):
    """HC Laplacian smoothing via PyMeshLab (low shrinkage)."""
    import pymeshlab

    ms = pymeshlab.MeshSet()
    pml_mesh = pymeshlab.Mesh(
        vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
        face_matrix=np.asarray(mesh.faces, dtype=np.int32),
    )
    ms.add_mesh(pml_mesh)

    try:
        ms.apply_coord_hc_laplacian_smoothing()
    except AttributeError:
        try:
            ms.hc_laplacian_smooth()
        except AttributeError:
            raise RuntimeError(
                "PyMeshLab HC Laplacian smoothing filter not available. "
                "Ensure pymeshlab >= 2023.12 is installed."
            )

    out = ms.current_mesh()
    return trimesh_module.Trimesh(
        vertices=out.vertex_matrix(),
        faces=out.face_matrix(),
        process=False,
    )


class SmoothHCLaplacianNode(io.ComfyNode):
    """PyMeshLab HC Laplacian smoothing backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackSmooth_HCLaplacian",
            display_name="Smooth HC Laplacian (backend)",
            category="geompack/smoothing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="smoothed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh):
        log.info("Backend: hc_laplacian")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))

        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        smoothed = _pymeshlab_hc_laplacian_smooth(trimesh)

        if hasattr(trimesh, "metadata") and trimesh.metadata:
            smoothed.metadata = trimesh.metadata.copy()
        smoothed.metadata["smoothing"] = {
            "algorithm": "hc_laplacian",
            "iterations": 1,
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

        info = f"""Smooth Mesh Results (hc_laplacian):

(single-pass HC correction)

Vertices: {initial_vertices:,} (unchanged)
Faces: {initial_faces:,} (unchanged)

Displacement:
  Average: {avg_disp:.6f}
  Maximum: {max_disp:.6f}
"""
        return io.NodeOutput(smoothed, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackSmooth_HCLaplacian": SmoothHCLaplacianNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackSmooth_HCLaplacian": "Smooth HC Laplacian (backend)"}
