# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Shared helper functions for sharpen backend nodes."""

import numpy as np


def _compute_face_geometry(V, F):
    """Compute face normals, centroids, and areas from vertex/face arrays.

    Returns:
        normals: (m, 3) unit face normals
        centroids: (m, 3) face centroids
        areas: (m,) face areas
    """
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    cross = np.cross(e1, e2)
    area_2x = np.linalg.norm(cross, axis=1, keepdims=True)
    normals = cross / (area_2x + 1e-12)
    areas = area_2x.ravel() * 0.5
    centroids = (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3.0
    return normals, centroids, areas


def _update_vertices_from_normals(V, F, target_normals, vertex_iterations,
                                  fixed_boundary=False):
    """Update vertex positions to match target face normals via iterative
    projection. For each iteration, projects each vertex onto the planes
    defined by the target normals of its adjacent faces, then averages.

    Matches MeshDenoisingBase::updateVertexPosition from the C++ reference:
    p += (1/N) * sum_j n_j * dot(n_j, c_j - p)

    Args:
        fixed_boundary: If True, boundary vertices are kept in place.

    Returns:
        V_new: (n, 3) updated vertex positions
    """
    V = V.copy()
    n_verts = len(V)
    m_faces = len(F)

    # Detect boundary vertices if needed
    if fixed_boundary:
        edge_face_count = {}
        for fi in range(m_faces):
            for i in range(3):
                e = (min(F[fi][i], F[fi][(i + 1) % 3]),
                     max(F[fi][i], F[fi][(i + 1) % 3]))
                edge_face_count[e] = edge_face_count.get(e, 0) + 1
        is_boundary = np.zeros(n_verts, dtype=bool)
        for (v0, v1), cnt in edge_face_count.items():
            if cnt == 1:
                is_boundary[v0] = True
                is_boundary[v1] = True
    else:
        is_boundary = None

    for _ in range(vertex_iterations):
        new_V = np.zeros_like(V)
        counts = np.zeros(n_verts)

        # Compute current centroids
        centroids = (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3.0

        for fi in range(m_faces):
            c = centroids[fi]
            n = target_normals[fi]
            for vi in F[fi]:
                if is_boundary is not None and is_boundary[vi]:
                    continue
                d = np.dot(V[vi] - c, n)
                new_V[vi] += V[vi] - d * n
                counts[vi] += 1

        mask = counts > 0
        new_V[mask] /= counts[mask, None]
        new_V[~mask] = V[~mask]
        V = new_V

    return V


def _build_vertex_to_faces(n_verts, F):
    """Build vertex-to-face adjacency list. Returns list of lists."""
    vtf = [[] for _ in range(n_verts)]
    for fi in range(len(F)):
        for vi in F[fi]:
            vtf[vi].append(fi)
    return vtf


def _build_vertex_based_face_neighbors(F, vert_to_faces, include_central=True):
    """Build vertex-based face neighbor lists.

    Two faces are neighbors if they share at least one vertex.
    Returns list of sorted index lists, one per face.
    """
    m = len(F)
    neighbors = []
    for fi in range(m):
        nbrs = set()
        for vi in F[fi]:
            for fj in vert_to_faces[vi]:
                nbrs.add(fj)
        if not include_central:
            nbrs.discard(fi)
        neighbors.append(sorted(nbrs))
    return neighbors
