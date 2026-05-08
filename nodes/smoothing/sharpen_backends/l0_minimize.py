# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""L0 normal minimization sharpening backend node (He & Schaefer 2013)."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io
from ._helpers import _compute_face_geometry, _update_vertices_from_normals

log = logging.getLogger("geometrypack")


def _l0_minimize_sharpen(mesh, alpha, beta, iterations):
    """L0 normal minimization for piecewise-flat sharpening.

    Iteratively thresholds face normal differences across edges: if the
    difference is below the current alpha threshold, adjacent normals are
    snapped to their area-weighted average. Alpha grows by factor beta each
    iteration, progressively eliminating small variations and forcing the
    mesh into piecewise-constant normal regions with sharp edges at
    boundaries.
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)

    if len(V) == 0 or len(F) == 0:
        return None, "Empty mesh (no vertices or faces)."

    adj_pairs = np.asarray(mesh.face_adjacency)
    if len(adj_pairs) == 0:
        return None, "Mesh has no face adjacency (disconnected or degenerate)."

    normals, centroids, areas = _compute_face_geometry(V, F)
    current_alpha = alpha

    for it in range(iterations):
        # Recompute normals from current vertex positions
        normals, centroids, areas = _compute_face_geometry(V, F)
        target_normals = normals.copy()

        # Threshold: snap adjacent normals whose difference^2 < current_alpha
        for ei in range(len(adj_pairs)):
            fi, fj = adj_pairs[ei]
            ni, nj = target_normals[fi], target_normals[fj]
            diff_sq = np.sum((ni - nj) ** 2)
            if diff_sq < current_alpha:
                ai, aj = areas[fi], areas[fj]
                avg = (ai * ni + aj * nj) / (ai + aj + 1e-12)
                norm_len = np.linalg.norm(avg)
                if norm_len > 1e-12:
                    avg /= norm_len
                target_normals[fi] = avg
                target_normals[fj] = avg

        # Update vertex positions to match target normals
        V = _update_vertices_from_normals(V, F, target_normals, vertex_iterations=1)

        current_alpha *= beta
        log.debug("L0 iteration %d: alpha=%.6f", it, current_alpha)

    result = trimesh_module.Trimesh(
        vertices=V,
        faces=np.asarray(mesh.faces, dtype=np.int32),
        process=False,
    )
    return result, ""


class SharpenL0MinimizeNode(io.ComfyNode):
    """L0 normal minimization sharpening backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackSharpen_L0Minimize",
            display_name="Sharpen L0 Minimize (backend)",
            category="geompack/smoothing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("alpha", default=0.001, min=0.0001, max=0.1, step=0.0001, tooltip=(
                    "Initial regularization weight for L0 minimization. "
                    "Controls the threshold below which normal differences "
                    "are snapped to zero. Smaller = gentler start, "
                    "larger = more aggressive initial flattening."
                )),
                io.Float.Input("beta", default=2.0, min=1.1, max=10.0, step=0.1, tooltip=(
                    "Growth rate for alpha each iteration. Alpha is multiplied "
                    "by beta each step. 2.0 doubles per iteration. "
                    "Higher = faster convergence to piecewise-flat."
                )),
                io.Int.Input("iterations", default=10, min=1, max=50, step=1, tooltip=(
                    "Number of L0 optimization iterations. The algorithm "
                    "gradually increases the threshold, snapping more normals "
                    "flat each step."
                )),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="sharpened_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, alpha=0.001, beta=2.0, iterations=10):
        log.info("Backend: l0_minimize")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: alpha=%.4f, beta=%.1f, iterations=%d",
                 alpha, beta, iterations)

        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        sharpened, error = _l0_minimize_sharpen(
            trimesh, alpha, beta, iterations,
        )

        if sharpened is None:
            raise ValueError(f"Sharpening failed (l0_minimize): {error}")

        # Copy metadata
        if hasattr(trimesh, "metadata") and trimesh.metadata:
            sharpened.metadata = trimesh.metadata.copy()
        sharpened.metadata["sharpening"] = {
            "algorithm": "l0_minimize",
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
            f"Alpha: {alpha}\n"
            f"Beta: {beta}\n"
            f"Iterations: {iterations}"
        )

        info = f"""Sharpen Mesh Results (l0_minimize):

{param_text}

Vertices: {initial_vertices:,} (unchanged)
Faces: {initial_faces:,} (unchanged)

Displacement:
  Average: {avg_disp:.6f}
  Maximum: {max_disp:.6f}
"""
        return io.NodeOutput(sharpened, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackSharpen_L0Minimize": SharpenL0MinimizeNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackSharpen_L0Minimize": "Sharpen L0 Minimize (backend)"}
