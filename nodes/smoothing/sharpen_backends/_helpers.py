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


def _anti_flip_step(V, F, step, eta=0.1, max_rounds=20):
    """Cap a proposed per-vertex displacement so NO triangle folds (signed-area
    barrier line search, as in inversion-free geometric optimization / IPC).

    For the current orientation (winding normal of each face), a face is allowed
    only down to `eta` of its current area; any face that would drop below that
    (or invert) has the step of its incident vertices halved, repeated until none
    violate. Remaining offenders are frozen. Guarantees the returned step keeps
    every face's signed area >= eta * current > 0, i.e. no fold, for any input.

    V: (n,3) current positions. step: (n,3) proposed displacement. Returns the
    capped step (n,3).
    """
    a, b, c = F[:, 0], F[:, 1], F[:, 2]
    e1 = V[b] - V[a]
    e2 = V[c] - V[a]
    n0 = np.cross(e1, e2)
    A0 = np.linalg.norm(n0, axis=1)                     # 2 * current area
    n0u = n0 / (A0[:, None] + 1e-20)                    # current winding normal
    thresh = eta * A0
    s = step.copy()
    for _ in range(max_rounds):
        Vc = V + s
        sa = np.einsum('ij,ij->i', np.cross(Vc[b] - Vc[a], Vc[c] - Vc[a]), n0u)
        bad = sa < thresh
        if not bad.any():
            return s
        bv = np.unique(F[bad].ravel())
        s[bv] *= 0.5
    # backstop: freeze any vertex still touching a folded face
    Vc = V + s
    sa = np.einsum('ij,ij->i', np.cross(Vc[b] - Vc[a], Vc[c] - Vc[a]), n0u)
    bad = sa < thresh
    if bad.any():
        s[np.unique(F[bad].ravel())] = 0.0
    return s


def _update_vertices_regularized(V, F, target_normals, vertex_iterations,
                                 V_anchor, anchor=0.5, anti_flip=True):
    """Reliable foldless point-to-plane vertex update (drag-back regularized).

    The bare Jacobi projection sweep (`_update_vertices_from_normals`) is only
    marginally stable: iterated many times it overshoots at creases, oscillates,
    and collapses/inverts triangles. Instead solve, per vertex, the regularized
    point-to-plane least squares EXACTLY each pass:

        (sum_{f in i} n_f n_f^T + anchor*I) x_i = sum_f n_f (n_f . c_f) + anchor*x0_i

    where c_f is the current face centroid, n_f the filtered target normal, and x0
    the FIXED original position. anchor*I + the drag-back data term make each 3x3
    well-conditioned and stop the drift/collapse; a signed-area barrier then caps
    the step so no triangle folds. Numerically matches the GPU port.

    anchor: drag-back strength. Lower = stronger smoothing (moves more), higher =
    gentler (stays near input).
    """
    V = V.copy()
    n = len(V)
    nf = np.ascontiguousarray(target_normals, dtype=np.float64)
    nnT = nf[:, :, None] * nf[:, None, :]                    # (m,3,3)
    eye = np.eye(3)
    for _ in range(int(vertex_iterations)):
        cen = (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3.0
        rhs_face = nf * np.einsum('ij,ij->i', nf, cen)[:, None]   # (m,3)
        M = np.zeros((n, 3, 3))
        rhs = np.zeros((n, 3))
        for k in range(3):
            np.add.at(M, F[:, k], nnT)
            np.add.at(rhs, F[:, k], rhs_face)
        M += anchor * eye
        rhs += anchor * V_anchor
        new_V = np.linalg.solve(M, rhs[:, :, None])[:, :, 0]
        if anti_flip:
            new_V = V + _anti_flip_step(V, F, new_V - V)
        V = new_V
    return V


def _update_vertices_from_normals(V, F, target_normals, vertex_iterations,
                                  fixed_boundary=False, relax=1.0, anti_flip=True):
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

        # Strong point-to-plane direction, capped by a signed-area barrier so no
        # triangle can fold (the guided_normal update is orientation-blind and
        # overshoots/inverts at preserved creases without this guard).
        step = relax * (new_V - V)
        if anti_flip:
            step = _anti_flip_step(V, F, step)
        V = V + step

    return V


def _build_vertex_to_faces(n_verts, F):
    """Build vertex-to-face adjacency list. Returns list of lists."""
    vtf = [[] for _ in range(n_verts)]
    for fi in range(len(F)):
        for vi in F[fi]:
            vtf[vi].append(fi)
    return vtf


def _build_vertex_based_face_neighbors(F, vert_to_faces, include_central=True, rings=1):
    """Build vertex-based face neighbor lists (k-ring patches).

    Two faces are 1-ring neighbors if they share at least one vertex. With
    rings>1 the patch is grown outward `rings` times: each step adds every face
    touching a vertex of the faces added in the previous step. Larger rings =
    wider filtering footprint (stronger per-pass effect, more cost).
    Returns a list of sorted index lists, one per face.
    """
    m = len(F)
    rings = max(1, int(rings))
    neighbors = []
    for fi in range(m):
        face_set = {fi}
        frontier_verts = set(int(v) for v in F[fi])
        for _ in range(rings):
            new_faces = set()
            for vi in frontier_verts:
                for fj in vert_to_faces[vi]:
                    if fj not in face_set:
                        new_faces.add(fj)
            if not new_faces:
                break
            face_set |= new_faces
            # Next frontier = vertices of the newly added faces.
            frontier_verts = set()
            for fj in new_faces:
                for vi in F[fj]:
                    frontier_verts.add(int(vi))
        if not include_central:
            face_set.discard(fi)
        neighbors.append(sorted(face_set))
    return neighbors
