# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Guided mesh normal filtering sharpening backend node (Zhang et al. 2015)."""

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


def _guided_normal_sharpen(mesh, normal_iterations, vertex_iterations,
                           sigma_s, sigma_r):
    """Guided mesh normal filtering with interleaved vertex update.

    Matches GuidedMeshNormalFiltering::updateFilteredNormalsLocalScheme from
    the C++ reference (Zhang et al. 2015).

    Per iteration:
    1. Recompute geometry from current vertices
    2. Compute guidance normals via min-range-metric: for each face's
       vertex-based 1-ring, compute a sharpness metric (maxdiff * max_tv /
       sum_tv). Pick the neighbor with minimum metric and use its area-weighted
       average normal as guidance.
    3. Bilateral filter normals using guidance for range weight
    4. Immediately update vertex positions (interleaved)
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)

    if len(V) == 0 or len(F) == 0:
        return None, "Empty mesh (no vertices or faces)."

    m = len(F)
    adj_pairs = np.asarray(mesh.face_adjacency)
    if len(adj_pairs) == 0:
        return None, "Mesh has no face adjacency (disconnected or degenerate)."

    # Build vertex-based face neighbor structures
    vert_to_faces = _build_vertex_to_faces(len(V), F)
    # Guided neighborhood: vertex-based including central face
    guided_neighbors = _build_vertex_based_face_neighbors(F, vert_to_faces, include_central=True)
    # Filtering neighborhood: same vertex-based including central
    filter_neighbors = guided_neighbors

    # Precompute face-to-adjacency mapping for inner edge computation
    face_to_adj = [[] for _ in range(m)]
    for ai in range(len(adj_pairs)):
        fi, fj = adj_pairs[ai]
        face_to_adj[fi].append(ai)
        face_to_adj[fj].append(ai)

    for normal_iter in range(normal_iterations):
        # Recompute geometry from current vertex positions
        normals, centroids, areas = _compute_face_geometry(V, F)

        # Compute sigma_s in absolute units (avg centroid-to-centroid distance * multiple)
        edge_lens = np.concatenate([
            np.linalg.norm(V[F[:, 1]] - V[F[:, 0]], axis=1),
            np.linalg.norm(V[F[:, 2]] - V[F[:, 1]], axis=1),
            np.linalg.norm(V[F[:, 0]] - V[F[:, 2]], axis=1),
        ])
        avg_edge_len = float(np.mean(edge_lens))
        sigma_s_abs = sigma_s * avg_edge_len
        sigma_s_sq2 = 2.0 * sigma_s_abs * sigma_s_abs
        sigma_r_sq2 = 2.0 * sigma_r * sigma_r

        # --- Step 1: Compute guidance normals via min-range-metric ---
        # For each face compute (metric, area_weighted_avg_normal) over its
        # guided neighborhood, matching getRangeAndMeanNormal in the C++ code.
        metrics = np.zeros(m)
        avg_normals = np.zeros((m, 3))

        for fi in range(m):
            patch = guided_neighbors[fi]
            n_patch = len(patch)

            # Area-weighted average normal
            patch_areas = areas[patch]
            patch_normals = normals[patch]
            avg_n = np.sum(patch_normals * patch_areas[:, None], axis=0)
            norm_len = np.linalg.norm(avg_n)
            if norm_len > 1e-12:
                avg_n /= norm_len
            avg_normals[fi] = avg_n

            # Max pairwise normal difference in patch
            if n_patch > 1:
                pn = normals[patch]
                diffs = np.linalg.norm(pn[:, None, :] - pn[None, :, :], axis=2)
                maxdiff = float(np.max(diffs))
            else:
                maxdiff = 0.0

            # Inner edges: adjacency pairs where both faces are in the patch
            patch_set = set(patch)
            max_tv = 0.0
            sum_tv = 0.0
            seen_edges = set()
            for pf in patch:
                for ai in face_to_adj[pf]:
                    if ai not in seen_edges:
                        seen_edges.add(ai)
                        fa, fb = adj_pairs[ai]
                        if fa in patch_set and fb in patch_set:
                            tv = float(np.linalg.norm(normals[fa] - normals[fb]))
                            if tv > max_tv:
                                max_tv = tv
                            sum_tv += tv

            metrics[fi] = maxdiff * max_tv / (sum_tv + 1e-9)

        # For each face, find the guided neighbor with minimum metric and use
        # that neighbor's area-weighted average normal as guidance.
        guided_normals = np.zeros((m, 3))
        for fi in range(m):
            patch = guided_neighbors[fi]
            min_metric = 1e8
            min_idx = patch[0]
            for pf in patch:
                if metrics[pf] < min_metric:
                    min_metric = metrics[pf]
                    min_idx = pf
            guided_normals[fi] = avg_normals[min_idx]

        # --- Step 2: Bilateral filter normals using guidance ---
        filtered = np.zeros((m, 3))
        for fi in range(m):
            patch = filter_neighbors[fi]
            if not patch:
                filtered[fi] = normals[fi]
                continue

            w_total = 0.0
            n_acc = np.zeros(3)
            for fj in patch:
                dist_sq = float(np.sum((centroids[fi] - centroids[fj]) ** 2))
                ws = np.exp(-dist_sq / (sigma_s_sq2 + 1e-12))

                # Range weight uses guidance normal distance (matching C++ reference)
                gdiff_sq = float(np.sum((guided_normals[fi] - guided_normals[fj]) ** 2))
                wr = np.exp(-gdiff_sq / (sigma_r_sq2 + 1e-12))

                w = areas[fj] * ws * wr
                n_acc += normals[fj] * w
                w_total += w

            if w_total > 1e-12:
                filtered[fi] = n_acc / w_total
            else:
                filtered[fi] = normals[fi]

        f_norms = np.linalg.norm(filtered, axis=1, keepdims=True)
        filtered_normals = filtered / (f_norms + 1e-12)

        # --- Step 3: Interleaved vertex update (matching C++ reference) ---
        V = _update_vertices_from_normals(V, F, filtered_normals, vertex_iterations)

        log.debug("Guided normal iteration %d/%d complete", normal_iter + 1, normal_iterations)

    result = trimesh_module.Trimesh(
        vertices=V,
        faces=np.asarray(mesh.faces, dtype=np.int32),
        process=False,
    )
    return result, ""


class SharpenGuidedNormalNode(io.ComfyNode):
    """Guided mesh normal filtering sharpening backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackSharpen_GuidedNormal",
            display_name="Sharpen Guided Normal (backend)",
            category="geompack/smoothing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("normal_iterations", default=5, min=1, max=50, step=1, tooltip=(
                    "Iterations for guided bilateral normal filtering. "
                    "More iterations produce smoother/flatter regions while "
                    "preserving sharp edges."
                )),
                io.Int.Input("vertex_iterations", default=10, min=1, max=100, step=1, tooltip=(
                    "Iterations for updating vertex positions to match filtered "
                    "normals. More iterations give better convergence."
                )),
                io.Float.Input("sigma_s", default=1.0, min=0.1, max=10.0, step=0.1, tooltip=(
                    "Spatial weight sigma as a multiple of average edge length. "
                    "Controls the neighborhood size for normal filtering. "
                    "Larger = smoother but may blur sharp features."
                )),
                io.Float.Input("sigma_r", default=0.35, min=0.01, max=1.0, step=0.01, tooltip=(
                    "Normal similarity threshold. Controls which normals are "
                    "averaged together. Smaller = more aggressive edge "
                    "preservation. 0.35 corresponds to roughly 40 degree "
                    "dihedral angle threshold."
                )),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="sharpened_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, normal_iterations=5, vertex_iterations=10,
                sigma_s=1.0, sigma_r=0.35):
        log.info("Backend: guided_normal")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: normal_iter=%d, vertex_iter=%d, sigma_s=%.2f, sigma_r=%.3f",
                 normal_iterations, vertex_iterations, sigma_s, sigma_r)

        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        sharpened, error = _guided_normal_sharpen(
            trimesh, normal_iterations, vertex_iterations, sigma_s, sigma_r,
        )

        if sharpened is None:
            raise ValueError(f"Sharpening failed (guided_normal): {error}")

        # Copy metadata
        if hasattr(trimesh, "metadata") and trimesh.metadata:
            sharpened.metadata = trimesh.metadata.copy()
        sharpened.metadata["sharpening"] = {
            "algorithm": "guided_normal",
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
            f"Normal Iterations: {normal_iterations}\n"
            f"Vertex Iterations: {vertex_iterations}\n"
            f"Sigma S: {sigma_s}\n"
            f"Sigma R: {sigma_r}"
        )

        info = f"""Sharpen Mesh Results (guided_normal):

{param_text}

Vertices: {initial_vertices:,} (unchanged)
Faces: {initial_faces:,} (unchanged)

Displacement:
  Average: {avg_disp:.6f}
  Maximum: {max_disp:.6f}
"""
        return io.NodeOutput(sharpened, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackSharpen_GuidedNormal": SharpenGuidedNormalNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackSharpen_GuidedNormal": "Sharpen Guided Normal (backend)"}
