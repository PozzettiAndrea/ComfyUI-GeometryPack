# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Non-Iterative Feature-Preserving Mesh Smoothing backend node
(Jones et al. SIGGRAPH 2003)."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io
from ._helpers import (
    _compute_face_geometry,
    _build_vertex_to_faces,
    _build_vertex_based_face_neighbors,
)

log = logging.getLogger("geometrypack")


def _non_iterative_sharpen(mesh, sigma_f_ratio, sigma_g_ratio):
    """Non-Iterative Feature-Preserving Mesh Smoothing.

    Ported from NonIterativeFeaturePreservingMeshFiltering.cpp.

    1. Mollify normals: smooth vertex positions with Gaussian (sigma_m = sigma_f/2),
       then recompute face normals on the smoothed mesh.
    2. For each vertex, BFS-expand to find faces within radius 2*sigma_f.
    3. Project vertex onto each neighbor face plane (using mollified normal +
       original centroid). Weight by area * spatial_Gaussian(sigma_f) *
       influence_Gaussian(sigma_g). Average projections.
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)

    if len(V) == 0 or len(F) == 0:
        return None, "Empty mesh (no vertices or faces)."

    m = len(F)
    n = len(V)

    normals, centroids, areas = _compute_face_geometry(V, F)

    # Compute average edge length
    edge_lens = np.concatenate([
        np.linalg.norm(V[F[:, 1]] - V[F[:, 0]], axis=1),
        np.linalg.norm(V[F[:, 2]] - V[F[:, 1]], axis=1),
        np.linalg.norm(V[F[:, 0]] - V[F[:, 2]], axis=1),
    ])
    avg_edge_len = float(np.mean(edge_lens))

    sigma_f = sigma_f_ratio * avg_edge_len
    sigma_g = sigma_g_ratio * avg_edge_len
    sigma_m = sigma_f / 2.0
    radius = 2.0 * sigma_f

    # Build vertex-to-face adjacency
    vert_to_faces = _build_vertex_to_faces(n, F)

    # Build vertex-based face adjacency for BFS expansion (matching C++ kVertexBased)
    vbased_face_adj = _build_vertex_based_face_neighbors(
        F, vert_to_faces, include_central=False)

    def _bfs_faces_in_radius(vertex_pos, seed_faces, rad):
        """BFS from seed faces, expanding via vertex-based neighbors within radius."""
        visited = set()
        queue = []
        for fi in seed_faces:
            dist = float(np.linalg.norm(vertex_pos - centroids[fi]))
            if dist <= rad:
                queue.append(fi)
            visited.add(fi)
        result_faces = []
        head = 0
        while head < len(queue):
            fi = queue[head]
            result_faces.append(fi)
            head += 1
            for fj in vbased_face_adj[fi]:
                if fj not in visited:
                    visited.add(fj)
                    dist = float(np.linalg.norm(vertex_pos - centroids[fj]))
                    if dist <= rad:
                        queue.append(fj)
        return result_faces

    # --- Step 1: Mollified normals ---
    # Smooth vertex positions with area-weighted Gaussian (sigma_m)
    V_smooth = V.copy()
    for vi in range(n):
        pt = V[vi]
        seed_faces = vert_to_faces[vi]
        if not seed_faces:
            continue

        faces_in_range = _bfs_faces_in_radius(pt, seed_faces, radius)

        new_pt = np.zeros(3)
        w_sum = 0.0
        for fi in faces_in_range:
            c = centroids[fi]
            dist = float(np.linalg.norm(c - pt))
            w = np.exp(-0.5 * dist * dist / (sigma_m * sigma_m + 1e-12))
            a = areas[fi]
            new_pt += c * a * w
            w_sum += a * w

        if w_sum > 1e-12:
            V_smooth[vi] = new_pt / w_sum

    # Recompute normals on the smoothed mesh
    mollified_normals, _, _ = _compute_face_geometry(V_smooth, F)

    # --- Step 2: Single-pass bilateral vertex update ---
    V_new = V.copy()
    sigma_f_sq = sigma_f * sigma_f
    sigma_g_sq = sigma_g * sigma_g

    for vi in range(n):
        pt = V[vi]
        seed_faces = vert_to_faces[vi]
        if not seed_faces:
            continue

        faces_in_range = _bfs_faces_in_radius(pt, seed_faces, radius)
        if not faces_in_range:
            continue

        temp_pt = np.zeros(3)
        w_sum = 0.0
        for fi in faces_in_range:
            c = centroids[fi]
            mn = mollified_normals[fi]

            # Project vertex onto face plane (mollified normal + original centroid)
            proj = pt - mn * float(np.dot(pt - c, mn))

            dist_spatial = float(np.linalg.norm(c - pt))
            w_spatial = np.exp(-0.5 * dist_spatial * dist_spatial / (sigma_f_sq + 1e-12))

            dist_influence = float(np.linalg.norm(proj - pt))
            w_influence = np.exp(-0.5 * dist_influence * dist_influence / (sigma_g_sq + 1e-12))

            a = areas[fi]
            temp_pt += proj * a * w_spatial * w_influence
            w_sum += a * w_spatial * w_influence

        if w_sum > 1e-12:
            V_new[vi] = temp_pt / w_sum

    result = trimesh_module.Trimesh(
        vertices=V_new,
        faces=np.asarray(mesh.faces, dtype=np.int32),
        process=False,
    )
    return result, ""


class SharpenNonIterativeNode(io.ComfyNode):
    """Non-Iterative Feature-Preserving Mesh Smoothing backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackSharpen_NonIterative",
            display_name="Sharpen Non Iterative (backend)",
            category="geompack/smoothing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("sigma_f", default=1.0, min=0.001, max=10.0, step=0.1, tooltip=(
                    "Spatial sigma as multiple of average edge length "
                    "(Jones et al. SIGGRAPH 2003). Controls spatial extent "
                    "of the bilateral filter. Face neighbors are searched "
                    "within radius 2*sigma_f. Larger = smoother."
                )),
                io.Float.Input("sigma_g", default=1.0, min=0.001, max=10.0, step=0.1, tooltip=(
                    "Influence sigma as multiple of average edge length. "
                    "Controls sensitivity to projection distance (how far "
                    "the vertex moves toward each face plane). Smaller = "
                    "more feature-preserving."
                )),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="sharpened_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, sigma_f=1.0, sigma_g=1.0):
        log.info("Backend: non_iterative")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: sigma_f=%.3f, sigma_g=%.3f", sigma_f, sigma_g)

        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        sharpened, error = _non_iterative_sharpen(
            trimesh, sigma_f, sigma_g,
        )

        if sharpened is None:
            raise ValueError(f"Sharpening failed (non_iterative): {error}")

        # Copy metadata
        if hasattr(trimesh, "metadata") and trimesh.metadata:
            sharpened.metadata = trimesh.metadata.copy()
        sharpened.metadata["sharpening"] = {
            "algorithm": "non_iterative",
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
            f"Sigma F: {sigma_f}\n"
            f"Sigma G: {sigma_g}"
        )

        info = f"""Sharpen Mesh Results (non_iterative):

{param_text}

Vertices: {initial_vertices:,} (unchanged)
Faces: {initial_faces:,} (unchanged)

Displacement:
  Average: {avg_disp:.6f}
  Maximum: {max_disp:.6f}
"""
        return io.NodeOutput(sharpened, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackSharpen_NonIterative": SharpenNonIterativeNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackSharpen_NonIterative": "Sharpen Non Iterative (backend)"}
