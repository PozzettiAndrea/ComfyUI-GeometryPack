# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Geometric unsharp mask sharpening backend node (pymeshlab)."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _pymeshlab_unsharp_mask_sharpen(mesh, weight, weight_original, iterations):
    """Geometric unsharp mask sharpening via PyMeshLab."""
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
        ms.apply_coord_unsharp_mask(
            weight=weight,
            weightorig=weight_original,
            iterations=iterations,
        )
    except AttributeError:
        try:
            ms.coord_unsharp_mask(
                weight=weight,
                weightorig=weight_original,
                iterations=iterations,
            )
        except AttributeError:
            return None, (
                "PyMeshLab coordinate unsharp mask filter not available. "
                "Ensure pymeshlab >= 2023.12 is installed."
            )

    out = ms.current_mesh()
    result = trimesh_module.Trimesh(
        vertices=out.vertex_matrix(),
        faces=out.face_matrix(),
        process=False,
    )
    return result, ""


class SharpenUnsharpMaskNode(io.ComfyNode):
    """Geometric unsharp mask sharpening backend (pymeshlab)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackSharpen_UnsharpMask",
            display_name="Sharpen Unsharp Mask (backend)",
            category="geompack/smoothing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("weight", default=0.3, min=0.0, max=3.0, step=0.01, tooltip=(
                    "Unsharp mask weight controlling sharpening strength. "
                    "Higher = more pronounced sharpening."
                )),
                io.Int.Input("iterations", default=5, min=1, max=50, step=1, tooltip=(
                    "Smoothing iterations for the reference smooth mesh. "
                    "More iterations = larger-scale sharpening."
                )),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="sharpened_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, weight=0.3, iterations=5):
        log.info("Backend: unsharp_mask")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: weight=%.3f, iterations=%d", weight, iterations)

        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        sharpened, error = _pymeshlab_unsharp_mask_sharpen(
            trimesh, weight, weight_original=1.0, iterations=iterations,
        )

        if sharpened is None:
            raise ValueError(f"Sharpening failed (unsharp_mask): {error}")

        # Copy metadata
        if hasattr(trimesh, "metadata") and trimesh.metadata:
            sharpened.metadata = trimesh.metadata.copy()
        sharpened.metadata["sharpening"] = {
            "algorithm": "unsharp_mask",
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
            f"Weight: {weight}\n"
            f"Iterations: {iterations}"
        )

        info = f"""Sharpen Mesh Results (unsharp_mask):

{param_text}

Vertices: {initial_vertices:,} (unchanged)
Faces: {initial_faces:,} (unchanged)

Displacement:
  Average: {avg_disp:.6f}
  Maximum: {max_disp:.6f}
"""
        return io.NodeOutput(sharpened, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackSharpen_UnsharpMask": SharpenUnsharpMaskNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackSharpen_UnsharpMask": "Sharpen Unsharp Mask (backend)"}
