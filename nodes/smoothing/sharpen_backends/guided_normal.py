# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Guided mesh normal filtering sharpening backend node (Zhang et al. 2015)."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io
from ._helpers import (
    _compute_face_geometry,
    _update_vertices_regularized,
    _build_vertex_to_faces,
    _build_vertex_based_face_neighbors,
)

log = logging.getLogger("geometrypack")


def _guided_normal_sharpen(mesh, normal_iterations, vertex_iterations,
                           sigma_s, sigma_r, neighborhood_rings=1, vertex_anchor=0.5):
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

    V_anchor = V.copy()                  # fixed original positions (drag-back data term)

    m = len(F)
    adj_pairs = np.asarray(mesh.face_adjacency)
    if len(adj_pairs) == 0:
        return None, "Mesh has no face adjacency (disconnected or degenerate)."

    # Build vertex-based face neighbor structures
    vert_to_faces = _build_vertex_to_faces(len(V), F)
    # Guided neighborhood: vertex-based including central face
    guided_neighbors = _build_vertex_based_face_neighbors(F, vert_to_faces, include_central=True, rings=neighborhood_rings)
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

        # --- Step 3: regularized foldless vertex update (drag-back to anchor) ---
        V = _update_vertices_regularized(V, F, filtered_normals, vertex_iterations,
                                         V_anchor, anchor=vertex_anchor)

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
                io.Int.Input("normal_iterations", default=5, min=1, max=1000, step=1, tooltip=(
                    "Iterations for guided bilateral normal filtering. "
                    "More iterations produce smoother/flatter regions while "
                    "preserving sharp edges. This is the main strength knob; "
                    "high values (50-200) progressively flatten."
                )),
                io.Int.Input("vertex_iterations", default=10, min=1, max=100, step=1, tooltip=(
                    "Iterations for updating vertex positions to match filtered "
                    "normals. More iterations give better convergence."
                )),
                io.Int.Input("neighborhood_rings", default=1, min=1, max=4, step=1, tooltip=(
                    "Size of the face neighborhood (k-ring) used for both guidance and "
                    "filtering. 1 = faces sharing a vertex (standard). Higher = wider "
                    "footprint, so each pass hits HARDER and reaches farther (stronger "
                    "than just adding iterations) -- but cost and memory grow quickly "
                    "(patch size ~ rings^2). Try 2 for a noticeably stronger effect."
                )),
                io.Float.Input("sigma_s", default=1.0, min=0.1, max=10.0, step=0.1, tooltip=(
                    "SPATIAL scale = neighborhood size as a multiple of the average edge "
                    "length. Controls how far (in surface distance) a neighbor face still "
                    "influences the filter. Larger = wider smoothing (can blur small "
                    "features); smaller = more local. Default 1.0 (~one ring)."
                )),
                io.Float.Input("sigma_r_degrees", default=20.0, min=1.0, max=120.0, step=1.0, tooltip=(
                    "RANGE scale, in DEGREES: the angle between two faces' normals at which "
                    "they stop being averaged together (the edge-preservation knob). "
                    "Internally converted to a unit-normal distance via 2*sin(theta/2). "
                    "SMALLER (e.g. 10 deg) = only near-parallel faces blend = sharper, "
                    "stronger edge preservation. LARGER (e.g. 45 deg) = more faces blend = "
                    "smoother, softer edges. Default 20 deg."
                )),
                io.Float.Input("vertex_anchor", default=0.5, min=0.01, max=10.0, step=0.01, display_mode="number", tooltip=(
                    "Drag-back strength of the foldless vertex update: each vertex is pulled "
                    "toward its ORIGINAL position while it moves to match the filtered normals. "
                    "This regularization is what makes strong smoothing STABLE -- it stops the "
                    "old projection sweep from overshooting at creases, collapsing triangles, "
                    "and folding. LOWER (e.g. 0.05) = stronger smoothing (moves more); HIGHER "
                    "(e.g. 2.0) = gentler, stays near the input. Default 0.5."
                )),
                io.Combo.Input("use_gpu", options=["false", "true"], default="false", tooltip=(
                    "Run the faithful vectorized torch port instead of the per-face Python "
                    "loops. Uses CUDA when available (else vectorized CPU torch) -- much "
                    "faster on large meshes. Topology (1-ring patches + inner edges) is "
                    "built once on CPU, then every iteration runs on the GPU. Same "
                    "min-range-metric guidance and bilateral filter as the CPU path; results "
                    "can differ slightly (float32 vs float64)."
                )),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="sharpened_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, normal_iterations=5, vertex_iterations=10,
                neighborhood_rings=1, sigma_s=1.0, sigma_r_degrees=20.0,
                vertex_anchor=0.5, use_gpu="false"):
        import math
        import time
        gpu = (use_gpu == "true")
        algorithm = "guided_normal_gpu" if gpu else "guided_normal"
        # Range weight works on unit-normal distances; a difference angle of theta
        # between two unit normals is a chord of length 2*sin(theta/2). Let the user
        # think in degrees and convert to that internal sigma here.
        sigma_r = 2.0 * math.sin(math.radians(sigma_r_degrees) / 2.0)
        log.info("Backend: %s", algorithm)
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: normal_iter=%d, vertex_iter=%d, rings=%d, sigma_s=%.2f, sigma_r=%.3f (%.1f deg), vertex_anchor=%.3f, use_gpu=%s",
                 normal_iterations, vertex_iterations, neighborhood_rings, sigma_s, sigma_r, sigma_r_degrees, vertex_anchor, use_gpu)

        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        device = "cpu"
        t0 = time.perf_counter()
        if gpu:
            from .guided_normal_gpu import _guided_normal_gpu
            sharpened, error, device = _guided_normal_gpu(
                trimesh, normal_iterations, vertex_iterations, sigma_s, sigma_r,
                neighborhood_rings=neighborhood_rings, vertex_anchor=vertex_anchor,
            )
        else:
            sharpened, error = _guided_normal_sharpen(
                trimesh, normal_iterations, vertex_iterations, sigma_s, sigma_r,
                neighborhood_rings=neighborhood_rings, vertex_anchor=vertex_anchor,
            )
        elapsed = time.perf_counter() - t0

        if sharpened is None:
            raise ValueError(f"Sharpening failed ({algorithm}): {error}")

        # Copy metadata
        if hasattr(trimesh, "metadata") and trimesh.metadata:
            sharpened.metadata = trimesh.metadata.copy()
        sharpened.metadata["sharpening"] = {
            "algorithm": algorithm,
            "device": device,
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
            f"Neighborhood Rings: {neighborhood_rings}\n"
            f"Sigma S: {sigma_s}\n"
            f"Sigma R: {sigma_r_degrees} deg (= {sigma_r:.3f} normal-dist)\n"
            f"Use GPU: {use_gpu}\n"
            f"Time: {elapsed:.2f}s"
        )

        info = f"""Sharpen Mesh Results ({algorithm}, device={device}):

{param_text}

Vertices: {initial_vertices:,} (unchanged)
Faces: {initial_faces:,} (unchanged)

Displacement:
  Average: {avg_disp:.6f}
  Maximum: {max_disp:.6f}
"""
        if len(sharpened.vertices) == len(trimesh.vertices):
            sharpened.vertex_attributes["sharpen_displacement_magnitude"] = disp.astype(np.float32)

        return io.NodeOutput(sharpened, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackSharpen_GuidedNormal": SharpenGuidedNormalNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackSharpen_GuidedNormal": "Sharpen Guided Normal (backend)"}
