# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Fast and Effective Feature-Preserving Mesh Denoising backend node
(Sun et al. TVCG 2007)."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io
from ._helpers import (
    _compute_face_geometry,
    _update_vertices_from_normals,
    _build_vertex_to_faces,
    _build_vertex_based_face_neighbors,
)

log = logging.getLogger("geometrypack")


def _fast_effective_sharpen(mesh, threshold_T, normal_iterations, vertex_iterations):
    """Fast and Effective Feature-Preserving Mesh Denoising.

    Ported from FastAndEffectiveFeaturePreservingMeshDenoising.cpp.

    Normal filtering weight: w = max(0, dot(ni, nj) - T)^2
    where T is a threshold controlling which normals contribute.
    Faces with normals more similar than T contribute with quadratic weight;
    faces with normals less similar than T contribute nothing. This produces
    sharp edges at dihedral angles corresponding to the threshold.
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)

    if len(V) == 0 or len(F) == 0:
        return None, "Empty mesh (no vertices or faces)."

    m = len(F)

    # Build vertex-based face neighbors (including central face, matching C++)
    vert_to_faces = _build_vertex_to_faces(len(V), F)
    all_face_neighbor = _build_vertex_based_face_neighbors(
        F, vert_to_faces, include_central=True)

    normals, _, _ = _compute_face_geometry(V, F)
    previous_normals = normals.copy()

    # Iterative normal filtering
    for it in range(normal_iterations):
        filtered = np.zeros((m, 3))
        for fi in range(m):
            ni = previous_normals[fi]
            temp_normal = np.zeros(3)
            for fj in all_face_neighbor[fi]:
                nj = previous_normals[fj]
                value = float(np.dot(ni, nj)) - threshold_T
                weight = value * value if value > 0.0 else 0.0
                temp_normal += nj * weight
            norm_len = np.linalg.norm(temp_normal)
            if norm_len > 1e-12:
                temp_normal /= norm_len
            filtered[fi] = temp_normal
        previous_normals = filtered
        log.debug("Fast effective normal iter %d/%d", it + 1, normal_iterations)

    # Update vertex positions (with fixed boundary matching C++ reference)
    V = _update_vertices_from_normals(V, F, previous_normals, vertex_iterations,
                                      fixed_boundary=True)

    result = trimesh_module.Trimesh(
        vertices=V,
        faces=np.asarray(mesh.faces, dtype=np.int32),
        process=False,
    )
    return result, ""


class SharpenFastEffectiveNode(io.ComfyNode):
    """Fast and Effective Feature-Preserving Mesh Denoising backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackSharpen_FastEffective",
            display_name="Sharpen Fast Effective (backend)",
            category="geompack/smoothing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("threshold_T", default=0.5, min=1e-10, max=1.0, step=0.01, tooltip=(
                    "Cosine similarity threshold (Sun et al. TVCG 2007). "
                    "Normals with dot(ni,nj) > T contribute with weight "
                    "(dot-T)^2; below T they contribute nothing. "
                    "Lower = more normals averaged (smoother), "
                    "higher = only very similar normals averaged (sharper). "
                    "0.5 is a good default."
                )),
                io.Int.Input("normal_iterations", default=20, min=1, max=500, step=1, tooltip=(
                    "Iterations for normal filtering. More iterations "
                    "produce stronger flattening of near-flat regions."
                )),
                io.Int.Input("vertex_iterations", default=50, min=1, max=500, step=1, tooltip=(
                    "Iterations for vertex position update from filtered "
                    "normals. Boundary vertices are kept fixed."
                )),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="sharpened_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, threshold_T=0.5, normal_iterations=20,
                vertex_iterations=50):
        log.info("Backend: fast_effective")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: threshold_T=%.3f, normal_iter=%d, vertex_iter=%d",
                 threshold_T, normal_iterations, vertex_iterations)

        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        sharpened, error = _fast_effective_sharpen(
            trimesh, threshold_T, normal_iterations, vertex_iterations,
        )

        if sharpened is None:
            raise ValueError(f"Sharpening failed (fast_effective): {error}")

        # Copy metadata
        if hasattr(trimesh, "metadata") and trimesh.metadata:
            sharpened.metadata = trimesh.metadata.copy()
        sharpened.metadata["sharpening"] = {
            "algorithm": "fast_effective",
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
            f"Threshold T: {threshold_T}\n"
            f"Normal Iterations: {normal_iterations}\n"
            f"Vertex Iterations: {vertex_iterations}"
        )

        info = f"""Sharpen Mesh Results (fast_effective):

{param_text}

Vertices: {initial_vertices:,} (unchanged)
Faces: {initial_faces:,} (unchanged)

Displacement:
  Average: {avg_disp:.6f}
  Maximum: {max_disp:.6f}
"""
        return io.NodeOutput(sharpened, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackSharpen_FastEffective": SharpenFastEffectiveNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackSharpen_FastEffective": "Sharpen Fast Effective (backend)"}
