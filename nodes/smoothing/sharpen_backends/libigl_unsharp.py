# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Cotangent-weighted geometric unsharp mask backend node (libigl)."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _libigl_unsharp_sharpen(mesh, weight, iterations):
    """Cotangent-weighted geometric unsharp mask via libigl.

    Uses implicit Laplacian smoothing (backward Euler) to compute a stable
    smooth reference, then amplifies the detail:
        V_smooth = solve (M - dt*L) V_smooth = M @ V  (repeated)
        V_sharp  = V + weight * (V - V_smooth)
    """
    try:
        import igl
    except ImportError:
        return None, "libigl is not installed. Install with: pip install libigl"
    from scipy.sparse.linalg import spsolve

    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)

    if len(V) == 0 or len(F) == 0:
        return None, "Empty mesh (no vertices or faces)."

    L = igl.cotmatrix(V, F)
    M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)

    # Implicit smoothing: solve (M - dt*L) V_smooth = M @ V
    # dt=1.0 gives strong smoothing per iteration; always stable.
    A = M - L
    V_smooth = V.copy()
    for _ in range(iterations):
        for dim in range(3):
            rhs = M @ V_smooth[:, dim]
            V_smooth[:, dim] = spsolve(A, rhs)

    # Unsharp mask: amplify the detail signal
    detail = V - V_smooth
    V_sharp = V + weight * detail

    result = trimesh_module.Trimesh(
        vertices=V_sharp,
        faces=np.asarray(mesh.faces, dtype=np.int32),
        process=False,
    )
    return result, ""


class SharpenLibiglUnsharpNode(io.ComfyNode):
    """Cotangent-weighted geometric unsharp mask backend (libigl)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackSharpen_LibiglUnsharp",
            display_name="Sharpen Libigl Unsharp (backend)",
            category="geompack/smoothing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("weight", default=0.5, min=0.01, max=5.0, step=0.01, tooltip=(
                    "How much detail to add back. 0.5 = subtle sharpening, "
                    "1.0 = double the detail, 2.0+ = aggressive."
                )),
                io.Int.Input("iterations", default=3, min=1, max=50, step=1, tooltip=(
                    "Smoothing iterations for the reference mesh. "
                    "More iterations = smoother reference = sharpens broader features. "
                    "Fewer iterations = sharpens fine detail."
                )),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="sharpened_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, weight=0.5, iterations=3):
        log.info("Backend: libigl_unsharp")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: weight=%.3f, iterations=%d", weight, iterations)

        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        sharpened, error = _libigl_unsharp_sharpen(
            trimesh, weight, iterations,
        )

        if sharpened is None:
            raise ValueError(f"Sharpening failed (libigl_unsharp): {error}")

        # Copy metadata
        if hasattr(trimesh, "metadata") and trimesh.metadata:
            sharpened.metadata = trimesh.metadata.copy()
        sharpened.metadata["sharpening"] = {
            "algorithm": "libigl_unsharp",
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

        info = f"""Sharpen Mesh Results (libigl_unsharp):

{param_text}

Vertices: {initial_vertices:,} (unchanged)
Faces: {initial_faces:,} (unchanged)

Displacement:
  Average: {avg_disp:.6f}
  Maximum: {max_disp:.6f}
"""
        return io.NodeOutput(sharpened, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackSharpen_LibiglUnsharp": SharpenLibiglUnsharpNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackSharpen_LibiglUnsharp": "Sharpen Libigl Unsharp (backend)"}
