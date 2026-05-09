# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Two-step bilateral normal sharpening backend node (pymeshlab)."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _pymeshlab_two_step_sharpen(mesh, smooth_steps, normal_threshold,
                                normal_iterations, fit_iterations, selected_only):
    """Two-step bilateral normal sharpening via PyMeshLab."""
    try:
        import pymeshlab
    except (ImportError, OSError):
        return None, "pymeshlab is not installed. Install with: pip install pymeshlab"

    ms = pymeshlab.MeshSet()
    pml_mesh = pymeshlab.Mesh(
        vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
        face_matrix=np.asarray(mesh.faces, dtype=np.int32),
    )
    ms.add_mesh(pml_mesh)

    try:
        ms.apply_coord_two_steps_smoothing(
            stepsmoothnum=smooth_steps,
            normalthr=normal_threshold,
            stepnormalnum=normal_iterations,
            stepfitnum=fit_iterations,
            selected=selected_only,
        )
    except AttributeError:
        try:
            ms.two_steps_smooth(
                stepsmoothnum=smooth_steps,
                normalthr=normal_threshold,
                stepnormalnum=normal_iterations,
                stepfitnum=fit_iterations,
                selected=selected_only,
            )
        except AttributeError:
            return None, (
                "PyMeshLab two-step smoothing filter not available. "
                "Ensure pymeshlab >= 2023.12 is installed."
            )

    out = ms.current_mesh()
    result = trimesh_module.Trimesh(
        vertices=out.vertex_matrix(),
        faces=out.face_matrix(),
        process=False,
    )
    return result, ""


class SharpenTwoStepNode(io.ComfyNode):
    """Two-step bilateral normal sharpening backend (pymeshlab)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackSharpen_TwoStep",
            display_name="Sharpen Two Step (backend)",
            category="geompack/smoothing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("smooth_steps", default=3, min=1, max=50, step=1, tooltip=(
                    "Number of two-step smoothing passes. "
                    "More steps = stronger sharpening effect."
                )),
                io.Float.Input("normal_threshold", default=60.0, min=0.0, max=180.0, step=0.5, tooltip=(
                    "Dihedral angle threshold in degrees. "
                    "Edges sharper than this angle are preserved as features. "
                    "Lower = more aggressive (more edges treated as creases). "
                    "60 is a good default for most CAD models."
                )),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="sharpened_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, smooth_steps=3, normal_threshold=60.0):
        log.info("Backend: two_step")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: smooth_steps=%d, normal_threshold=%.1f",
                 smooth_steps, normal_threshold)

        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        sharpened, error = _pymeshlab_two_step_sharpen(
            trimesh, smooth_steps, normal_threshold,
            normal_iterations=20, fit_iterations=20,
            selected_only=False,
        )

        if sharpened is None:
            raise ValueError(f"Sharpening failed (two_step): {error}")

        # Copy metadata
        if hasattr(trimesh, "metadata") and trimesh.metadata:
            sharpened.metadata = trimesh.metadata.copy()
        sharpened.metadata["sharpening"] = {
            "algorithm": "two_step",
            "original_vertices": initial_vertices,
            "original_faces": initial_faces,
        }

        # Compute displacement stats
        disp = np.linalg.norm(
            np.asarray(sharpened.vertices) - np.asarray(trimesh.vertices), axis=1
        )
        avg_disp = float(np.mean(disp))
        max_disp = float(np.max(disp))

        log.info("Output: %d vertices, %d faces",
                 len(sharpened.vertices), len(sharpened.faces))
        log.info("Avg vertex displacement: %.6f, max: %.6f", avg_disp, max_disp)

        param_text = (
            f"Smooth Steps: {smooth_steps}\n"
            f"Normal Threshold: {normal_threshold}\u00b0"
        )

        info = f"""Sharpen Mesh Results (two_step):

{param_text}

Vertices: {initial_vertices:,} (unchanged)
Faces: {initial_faces:,} (unchanged)

Displacement:
  Average: {avg_disp:.6f}
  Maximum: {max_disp:.6f}
"""
        return io.NodeOutput(sharpened, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackSharpen_TwoStep": SharpenTwoStepNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackSharpen_TwoStep": "Sharpen Two Step (backend)"}
